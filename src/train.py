# src/train.py

import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import time
# 使用相对导入，确保 utils.py 在同一目录下
from .utils import extract_hog_features, TRAIN_IMG_SIZE # 不直接用 preprocess，假设加载时已处理或HOG内部处理

# --- 路径定义 (相对于项目根目录) ---
DATA_DIR = 'data' # 改为相对路径，假设从根目录运行 run.sh
POSITIVE_DIR = os.path.join(DATA_DIR, 'positive')
NEGATIVE_DIR = os.path.join(DATA_DIR, 'negative')
MODEL_DIR = 'models' # 改为相对路径
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# --- 训练参数 ---
TEST_SPLIT_RATIO = 0.2 # 20% 的数据用作测试集
RANDOM_STATE = 42      # 随机种子，确保结果可复现
ENABLE_GRID_SEARCH = False # 是否进行网格搜索调优 (耗时，但可能提高精度)
GRID_SEARCH_CV_FOLDS = 3  # 网格搜索的交叉验证折数
GRID_SEARCH_PARAMS = {'C': [0.1, 1, 10]} # 网格搜索的 C 值范围 (可扩展)

def load_and_extract_features(image_dir, label, augment=False):
    """
    加载图像，调整大小，提取HOG特征，可选数据增强。
    Args:
        image_dir (str): 图像目录路径。
        label (int): 类别标签 (1 for positive, 0 for negative)。
        augment (bool): 是否对该类图像进行数据增强 (目前仅水平翻转)。
    Returns:
        tuple: (features_list, labels_list)
    """
    features_list = []
    labels_list = []
    print(f"加载标签 {label} 图像从: {image_dir} (增强: {augment})")
    if not os.path.isdir(image_dir):
        print(f"错误: 目录未找到 - {image_dir}")
        return [], []

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    try:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_extensions)]
    except FileNotFoundError:
        print(f"错误: 访问目录失败 - {image_dir}")
        return [], []

    if not image_files:
        print(f"警告: 在 {image_dir} 未找到支持的图像文件 {supported_extensions}")
        return [], []

    original_file_count = 0
    feature_vector_count = 0
    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        # 以灰度模式加载
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"警告: 无法读取图像 {filename}，跳过。")
            continue
        original_file_count += 1

        # 调整大小是 HOG 的前提
        try:
            img_resized = cv2.resize(img, TRAIN_IMG_SIZE)
        except Exception as e:
            print(f"警告: 调整图像 {filename} 大小失败: {e}，跳过。")
            continue

        # 提取原始图像特征
        features = extract_hog_features(img_resized)
        if features is not None:
            features_list.append(features)
            labels_list.append(label)
            feature_vector_count += 1
        else:
            print(f"警告: 无法从原始图像 {filename} 提取特征，跳过。")
            continue # 基础特征失败，不进行增强

        # --- 数据增强 ---
        if augment:
            # 水平翻转
            try:
                img_flipped = cv2.flip(img_resized, 1)
                features_flipped = extract_hog_features(img_flipped)
                if features_flipped is not None:
                    features_list.append(features_flipped)
                    labels_list.append(label)
                    feature_vector_count += 1
            except Exception as e:
                print(f"警告: 对 {filename} 水平翻转或提取特征时出错: {e}")

            # 在此添加更多增强... (例如轻微旋转, 亮度调整等)
            # 注意 HOG 对旋转敏感，角度不宜过大

    print(f"标签 {label}: 处理了 {original_file_count} 个文件, 生成了 {feature_vector_count} 个特征向量。")
    return features_list, labels_list

def train_svm_model():
    """主训练函数"""
    start_time = time.time()
    print("\n===================================")
    print(" 开始训练 HOG + SVM 模型")
    print("===================================")

    # --- 1. 加载数据 & 提取特征 ---
    print("\n--- [1/5] 加载数据 & 提取特征 ---")
    # 通常只对正样本做增强，或者按需对两者都做
    pos_features, pos_labels = load_and_extract_features(POSITIVE_DIR, 1, augment=True)
    neg_features, neg_labels = load_and_extract_features(NEGATIVE_DIR, 0, augment=False)

    if not pos_features or not neg_features:
        print("\n错误: 正样本或负样本特征列表为空。请检查 'data' 目录及内容。")
        print(f"  - 正样本目录: {POSITIVE_DIR}")
        print(f"  - 负样本目录: {NEGATIVE_DIR}")
        print("训练中止。")
        return False # 返回失败

    # --- 合并数据 ---
    X = np.array(pos_features + neg_features).astype(np.float32) # 确保类型正确
    y = np.array(pos_labels + neg_labels)

    if X.ndim != 2 or X.shape[0] != y.shape[0]:
         print("\n错误: 特征数据 X 或标签 y 维度不正确。")
         print(f"X shape: {X.shape}, y shape: {y.shape}")
         print("训练中止。")
         return False

    print(f"\n数据集信息:")
    print(f"  总样本数: {X.shape[0]}")
    print(f"  正样本数 (标签 1): {len(pos_features)}")
    print(f"  负样本数 (标签 0): {len(neg_features)}")
    print(f"  特征维度 (HOG): {X.shape[1]}")

    # --- 2. 数据划分 ---
    print("\n--- [2/5] 划分训练集和测试集 ---")
    if X.shape[0] * TEST_SPLIT_RATIO < 1 or X.shape[0] * (1-TEST_SPLIT_RATIO) < 1:
         print(f"\n错误: 数据集太小 ({X.shape[0]} 个样本)，无法按 {TEST_SPLIT_RATIO} 比例划分。")
         print("请增加更多训练数据。训练中止。")
         return False
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SPLIT_RATIO,
            random_state=RANDOM_STATE,
            stratify=y # 确保划分后类别比例大致不变
        )
        print(f"  训练集大小: {X_train.shape[0]} (正: {np.sum(y_train==1)}, 负: {np.sum(y_train==0)})")
        print(f"  测试集大小: {X_test.shape[0]} (正: {np.sum(y_test==1)}, 负: {np.sum(y_test==0)})")
    except ValueError as e:
        print(f"\n错误: 数据划分失败: {e}")
        print("这可能因为某个类别的样本数过少。")
        print("训练中止。")
        return False

    # --- 3. 特征标准化 ---
    print("\n--- [3/5] 标准化特征 (StandardScaler) ---")
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # 用训练集的均值/方差转换测试集
        print("  特征标准化完成。")
    except Exception as e:
        print(f"错误: 特征标准化失败: {e}")
        return False

    # --- 4. 训练 SVM 模型 ---
    print("\n--- [4/5] 训练 SVM 模型 ---")
    # LinearSVC 通常对高维数据有效，class_weight='balanced' 处理不平衡
    base_model = LinearSVC(random_state=RANDOM_STATE,
                           class_weight='balanced',
                           max_iter=5000, # 增加迭代次数防止不收敛警告
                           dual=False)    # 当 n_samples > n_features 时推荐 False

    if ENABLE_GRID_SEARCH:
        print(f"--- 启用 GridSearchCV (参数: {GRID_SEARCH_PARAMS}, CV={GRID_SEARCH_CV_FOLDS}) ---")
        grid_search = GridSearchCV(base_model, GRID_SEARCH_PARAMS,
                                   cv=GRID_SEARCH_CV_FOLDS, scoring='accuracy',
                                   n_jobs=-1, verbose=1)
        try:
            grid_search.fit(X_train_scaled, y_train)
            print(f"\n  GridSearchCV 完成.")
            print(f"  最佳参数: {grid_search.best_params_}")
            print(f"  最佳交叉验证准确率: {grid_search.best_score_:.4f}")
            model = grid_search.best_estimator_
        except Exception as e:
            print(f"\n错误: GridSearchCV 失败: {e}")
            print("  将使用默认参数 C=1.0 继续训练...")
            model = LinearSVC(C=1.0, random_state=RANDOM_STATE, class_weight='balanced', max_iter=5000, dual=False)
            try:
                 model.fit(X_train_scaled, y_train)
            except Exception as fit_e:
                 print(f"错误: 使用默认参数训练模型也失败了: {fit_e}")
                 return False
    else:
        # 不进行搜索，直接使用默认 C=1.0 训练
        default_C = 1.0
        print(f"--- 使用默认参数训练 LinearSVC (C={default_C}) ---")
        model = LinearSVC(C=default_C, random_state=RANDOM_STATE, class_weight='balanced', max_iter=5000, dual=False)
        try:
            model.fit(X_train_scaled, y_train)
            print("  模型训练完成。")
        except Exception as e:
            print(f"错误: 使用默认参数训练模型失败: {e}")
            return False

    # --- 5. 评估模型 ---
    print("\n--- [5/5] 评估模型在测试集上的表现 ---")
    try:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'], zero_division=0)
        matrix = confusion_matrix(y_test, y_pred)

        print(f"  测试集准确率 (Accuracy): {accuracy:.4f}")
        print("\n  测试集分类报告:")
        print(report)
        print("\n  测试集混淆矩阵:")
        # [[TN, FP], [FN, TP]]
        print(f"          预测为0 | 预测为1")
        print(f"  实际为0:  {matrix[0, 0]:^7d} | {matrix[0, 1]:^7d}")
        print(f"  实际为1:  {matrix[1, 0]:^7d} | {matrix[1, 1]:^7d}")

    except Exception as e:
        print(f"\n错误: 模型评估失败: {e}")
        print("  将尝试继续保存模型...") # 即使评估失败也尝试保存

    # --- 保存模型和标准化器 ---
    print("\n--- 保存模型和标准化处理器 ---")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True) # 确保目录存在
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"  模型已保存至: {MODEL_PATH}")
        print(f"  标准化器已保存至: {SCALER_PATH}")
    except Exception as e:
        print(f"\n错误: 保存模型或标准化器失败: {e}")
        return False # 保存失败，标记为失败

    # --- 结束 ---
    end_time = time.time()
    print("\n===================================")
    print(f" 训练过程成功完成，总耗时: {end_time - start_time:.2f} 秒")
    print("===================================")
    return True # 返回成功

if __name__ == "__main__":
    # 当直接运行此脚本时执行
    if train_svm_model():
        print("\n训练脚本执行成功。")
        print("提示: 检查 'models/' 目录确认模型文件已生成。")
        if not ENABLE_GRID_SEARCH:
            print("提示: 为了可能获得更高精度，考虑在脚本中设置 ENABLE_GRID_SEARCH = True 并重新运行 (会耗时更长)。")
    else:
        print("\n训练脚本执行失败。请检查上面的错误信息。")
        # 可以考虑退出码非0，以便 run.sh 捕获
        exit(1)

