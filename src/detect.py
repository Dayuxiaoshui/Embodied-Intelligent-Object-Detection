# src/detect.py

import cv2
import numpy as np
import json
import os
import joblib
import time
# 使用相对导入
from .utils import (preprocess_image, extract_hog_features, get_orientation_and_centroid,
                   nms, pyramid, sliding_window, TRAIN_IMG_SIZE)

# --- 路径定义 (相对于项目根目录) ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
INPUT_IMAGE_DIR = 'input_images'
OUTPUT_JSON = 'results.json' # 输出 JSON 文件名
OUTPUT_VIZ_DIR = 'detection_visualizations' # 可视化结果目录名

# --- 检测参数 (关键！需要根据模型和数据仔细调优) ---
PYRAMID_SCALE_FACTOR = 1.25 # 图像金字塔缩放比例 (>1), 越小越慢但可能检测到更多尺寸
SLIDING_WINDOW_STEP = 8    # 滑动窗口步长 (像素), 越小越密但也越慢
# 置信度阈值 (基于 LinearSVC 的 decision_function)
# 0 是决策边界，正值表示倾向于正类。需要根据实际效果调整。
# 初始可以尝试 0.5 或 1.0。漏检多则降低，误报多则提高。
CONFIDENCE_THRESHOLD = 0.8
NMS_IOU_THRESHOLD = 0.15   # 非极大值抑制 IoU 阈值, 越小抑制越多重叠框

# --- 可视化选项 ---
SAVE_VISUALIZATIONS = True # 是否保存带有检测框的可视化图像
VIZ_BOX_COLOR = (0, 255, 0) # BGR 颜色: 绿色
VIZ_CENTROID_COLOR = (0, 0, 255) # BGR 颜色: 红色
VIZ_TEXT_COLOR = (0, 255, 0) # BGR 颜色: 绿色

def detect_lock_holes(image_path, model, scaler):
    """
    使用多尺度滑窗检测单张图像中的锁孔。
    Args:
        image_path (str): 输入图像的完整路径。
        model: 加载的 SVM 模型对象。
        scaler: 加载的 StandardScaler 对象。
    Returns:
        tuple: (detection_results, visualized_image)
               detection_results (list): 包含检测到的每个锁孔信息的字典列表。
                                         如果未检测到，则为空列表。
               visualized_image (numpy.ndarray): 带有绘制结果的图像 (如果 SAVE_VISUALIZATIONS=True)
                                                  否则返回原始图像副本。
               如果处理失败，返回 (None, None)。
    """
    start_time = time.time()
    print(f"处理图像: {os.path.basename(image_path)} ...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"  错误: 无法读取图像 {image_path}")
        return None, None

    original_image = image.copy() # 保留原始图像用于可视化
    img_height, img_width = image.shape[:2]
    center_x, center_y = img_width / 2.0, img_height / 2.0

    # 1. 图像预处理 (灰度, 去噪, CLAHE)
    processed_image = preprocess_image(image)
    if processed_image is None:
        print(f"  错误: 图像预处理失败 {image_path}")
        return None, None

    # 存储原始检测结果 [(x, y, w, h), score]
    raw_detections = []
    # 记录金字塔的缩放比例，用于将坐标映射回原图
    scale_factors = []

    # 2. 图像金字塔 + 滑动窗口
    window_w, window_h = TRAIN_IMG_SIZE
    min_pyramid_size = (window_w, window_h) # 金字塔最小层不能小于窗口大小

    # 迭代图像金字塔的每一层
    current_scale = 1.0 # 初始比例为 1 (原图)
    for layer_index, scaled_img in enumerate(pyramid(processed_image, scale=PYRAMID_SCALE_FACTOR, minSize=min_pyramid_size)):
        scale_h, scale_w = scaled_img.shape
        # 当前层相对于原始图像的比例
        current_scale = img_height / float(scale_h) # 或 img_width / float(scale_w)
        scale_factors.append(current_scale)
        # print(f"  金字塔层 {layer_index}, 尺寸: {scaled_img.shape}, 缩放比例: {current_scale:.2f}")

        # 在当前缩放层上滑动窗口
        for (x, y, window) in sliding_window(scaled_img, SLIDING_WINDOW_STEP, TRAIN_IMG_SIZE):
            # 确保窗口是正确的尺寸 (边缘情况)
            if window.shape[0] != window_h or window.shape[1] != window_w:
                continue

            # a. 提取 HOG 特征
            features = extract_hog_features(window)
            if features is None: continue # 特征提取失败，跳过

            # b. 标准化特征 (使用训练时保存的 scaler)
            try:
                # scaler.transform 需要 2D 输入, [features] 将 1D 转为 (1, n_features)
                features_scaled = scaler.transform([features])
            except Exception as e:
                print(f"  警告: 特征标准化失败在 ({x},{y}), 比例 {current_scale:.2f} - {e}")
                continue

            # c. 使用 SVM 模型预测置信度
            try:
                # decision_function 返回样本到决策超平面的有符号距离
                confidence_score = model.decision_function(features_scaled)[0]
            except Exception as e:
                 print(f"  警告: 模型预测失败在 ({x},{y}), 比例 {current_scale:.2f} - {e}")
                 continue

            # d. 如果置信度高于阈值，记录检测结果
            if confidence_score > CONFIDENCE_THRESHOLD:
                # 将窗口坐标 (x, y) 和尺寸映射回原始图像坐标系
                orig_x = int(x * current_scale)
                orig_y = int(y * current_scale)
                orig_w = int(window_w * current_scale)
                orig_h = int(window_h * current_scale)
                # 保存格式: ( (x, y, w, h), score )
                raw_detections.append(((orig_x, orig_y, orig_w, orig_h), confidence_score))

    if not raw_detections:
        print(f"  未找到高于阈值 {CONFIDENCE_THRESHOLD} 的候选区域。")
        return [], original_image # 返回空列表和原图

    # 3. 应用非极大值抑制 (NMS)
    boxes = np.array([det[0] for det in raw_detections])
    scores = np.array([det[1] for det in raw_detections])
    final_boxes, final_scores = nms(boxes, scores, threshold=NMS_IOU_THRESHOLD)

    if len(final_boxes) == 0:
        print(f"  所有 {len(raw_detections)} 个候选区域都被 NMS (IoU > {NMS_IOU_THRESHOLD}) 抑制。")
        return [], original_image

    print(f"  初步检测到 {len(raw_detections)} 个候选, NMS 后保留 {len(final_boxes)} 个。")

    # 4. 处理最终检测结果
    detection_results = []
    viz_image = original_image # 开始在原图上绘制

    for i in range(len(final_boxes)):
        x, y, w, h = map(int, final_boxes[i])
        score = final_scores[i]

        # 确保 ROI 坐标有效且尺寸大于0
        x = max(0, x)
        y = max(0, y)
        w = min(img_width - x, w)  # 确保不超过图像右边界
        h = min(img_height - y, h) # 确保不超过图像下边界
        if w <= 0 or h <= 0:
            print(f"  警告：NMS 后的一个检测框尺寸无效 (w={w}, h={h})，跳过。")
            continue

        # 5. 提取最终 ROI 并计算方向和质心
        #    建议在预处理后的图像上计算，以获得更稳定的结果
        #    也可以尝试在原始灰度图上计算: roi_for_orientation = original_image[y:y+h, x:x+w]
        roi_for_orientation = processed_image[y:y+h, x:x+w]
        lock_angle, cx_roi, cy_roi = get_orientation_and_centroid(roi_for_orientation)

        # 计算绝对质心坐标 (相对于原始图像左上角)
        abs_cx = x + cx_roi
        abs_cy = y + cy_roi

        # 计算到图像中心的距离
        distance = np.sqrt((abs_cx - center_x)**2 + (abs_cy - center_y)**2)

        # 准备输出字典
        result = {
            "filename": os.path.basename(image_path),
            "box": {"x": x, "y": y, "w": w, "h": h},
            "centroid": {"x": abs_cx, "y": abs_cy},
            "confidence": round(float(score), 4),
            "distance_from_center": round(distance, 2),
            "orientation_angle": round(float(lock_angle), 2), # 保留两位小数
            # --- 以下为占位符，需要额外逻辑实现 ---
            "is_lock_original": True, # 假设是原装，需要额外判断逻辑
            "is_key_in": False,       # 假设没钥匙，需要额外判断或传感器
            "key_angle": 0            # 假设钥匙角度为0，需要额外判断或传感器
        }
        detection_results.append(result)

        # --- 可视化 (如果启用) ---
        if SAVE_VISUALIZATIONS:
            # 绘制边界框
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), VIZ_BOX_COLOR, 2)
            # 绘制质心
            cv2.circle(viz_image, (abs_cx, abs_cy), 5, VIZ_CENTROID_COLOR, -1) # 实心圆
            # 显示信息 (置信度和角度)
            info_text = f"C:{result['confidence']:.2f} A:{result['orientation_angle']:.1f}"
            # 调整文本位置，避免超出图像边界
            text_x = x
            text_y = y - 10 if y > 20 else y + h + 15
            cv2.putText(viz_image, info_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, VIZ_TEXT_COLOR, 1, cv2.LINE_AA)

    end_time = time.time()
    print(f"  图像处理完成, 耗时: {end_time - start_time:.2f} 秒.")

    return detection_results, viz_image

def main():
    """主函数：加载模型，处理输入目录中的所有图像，保存结果。"""
    print("--- 开始锁孔检测流程 ---")

    # 1. 检查并加载模型和标准化器
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"错误: 模型 ({MODEL_PATH}) 或 标准化器 ({SCALER_PATH}) 未找到。")
        print("请先成功运行训练脚本 (./run.sh 或 python src/train.py)。")
        return
    try:
        print("加载模型和标准化器...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("加载成功。")
    except Exception as e:
        print(f"错误: 加载模型或标准化器时出错: {e}")
        return

    # 2. 检查输入目录
    if not os.path.isdir(INPUT_IMAGE_DIR):
        print(f"错误: 输入图像目录未找到: {INPUT_IMAGE_DIR}")
        print("请创建该目录并将待检测图像放入其中。")
        return

    # 3. 获取图像文件列表
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    try:
        image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(supported_extensions)]
    except FileNotFoundError:
         print(f"错误: 访问输入目录失败: {INPUT_IMAGE_DIR}")
         return

    if not image_files:
        print(f"警告: 在目录 {INPUT_IMAGE_DIR} 中未找到支持的图像文件。")
        return

    print(f"\n发现 {len(image_files)} 张图像待处理...")

    # 4. 循环处理每张图像
    all_results_list = [] # 存储所有图像的检测结果
    total_start_time = time.time()
    processed_count = 0
    detection_count = 0

    for i, filename in enumerate(image_files):
        print(f"\n--- 处理第 {i+1}/{len(image_files)} 张图像: {filename} ---")
        image_path = os.path.join(INPUT_IMAGE_DIR, filename)
        # 调用检测函数
        results, visualized_image = detect_lock_holes(image_path, model, scaler)

        if results is None: # 表示处理过程中出现错误
             print(f"  处理图像 {filename} 时发生错误，跳过。")
             continue # 继续处理下一张图

        processed_count += 1
        if results: # 如果列表不为空，表示检测到了锁孔
             all_results_list.extend(results) # 将当前图像的检测结果加入总列表
             detection_count += len(results)

        # --- 保存可视化图像 (如果启用且有检测结果或需要保存原图) ---
        if SAVE_VISUALIZATIONS and visualized_image is not None:
            os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True) # 确保目录存在
            viz_filename = os.path.join(OUTPUT_VIZ_DIR, f"det_{filename}")
            try:
                cv2.imwrite(viz_filename, visualized_image)
                # print(f"  可视化结果已保存至: {viz_filename}")
            except Exception as e:
                print(f"  错误: 保存可视化图像失败: {e}")

    total_end_time = time.time()
    print(f"\n--- 检测流程结束 ---")
    print(f"总共处理了 {processed_count} 张图像。")
    print(f"总共检测到 {detection_count} 个锁孔实例。")
    total_time = total_end_time - total_start_time
    print(f"总耗时: {total_time:.2f} 秒。")
    if processed_count > 0:
        print(f"平均每张图像耗时: {total_time / processed_count:.2f} 秒。")

    # 5. 保存所有检测结果到 JSON 文件
    if all_results_list:
        output_path = os.path.join('.', OUTPUT_JSON) # 保存在项目根目录
        print(f"\n将 {len(all_results_list)} 条检测结果保存至 {output_path}...")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # 使用 indent=4 美化输出 JSON 格式
                json.dump(all_results_list, f, indent=4, ensure_ascii=False)
            print("结果保存成功。")
            if SAVE_VISUALIZATIONS:
                 print(f"可视化图像已保存至 '{OUTPUT_VIZ_DIR}/' 目录 (如果生成)。")
        except Exception as e:
            print(f"错误: 保存结果到 JSON 文件时出错: {e}")
    else:
        print("\n未检测到任何锁孔，不生成 JSON 文件。")

if __name__ == "__main__":
    main()
