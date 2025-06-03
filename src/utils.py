# src/utils.py

import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure # 用于 Gamma 校正 (可选)
import math

# --- HOG 和图像尺寸参数 (重要！根据实际锁孔大小和纹理调整) ---
# 训练和滑窗时使用的图像块尺寸 (像素)
TRAIN_IMG_SIZE = (64, 64) # 推荐与 data/ 目录下的样本尺寸一致
# HOG 特征提取参数
HOG_ORIENTATIONS = 9       # 梯度方向数 (通常 9)
HOG_PIXELS_PER_CELL = (8, 8) # 每个细胞单元的像素大小 (常用 8x8 或 6x6)
HOG_CELLS_PER_BLOCK = (2, 2) # 每个块包含的细胞单元数 (常用 2x2 或 3x3)
HOG_BLOCK_NORM = 'L2-Hys'  # 块归一化方法 (L2-Hys 较常用且鲁棒)

def preprocess_image(image):
    """
    图像预处理流程：灰度化 -> 去噪 -> 对比度增强。
    Args:
        image: 输入图像 (BGR 或 灰度)。
    Returns:
        预处理后的单通道灰度图像，或在失败时返回 None。
    """
    if image is None:
        print("错误：输入的图像为空 (preprocess_image)。")
        return None

    # 1. 转换为灰度图
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image # 已经是灰度图
    else:
        print(f"错误：不支持的图像形状 {image.shape} (preprocess_image)。")
        return None

    # 2. 高斯模糊降噪 (核大小 (5,5) 是常用值，可调)
    #    sigmaX=0 表示根据核大小自动计算高斯核标准差
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 应用 CLAHE (对比度受限的自适应直方图均衡化)
    #    clipLimit: 对比度限制阈值，防止噪声过度放大 (2.0-4.0 常用)
    #    tileGridSize: 网格大小，在其内进行直方图均衡 (8x8 常用)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(denoised)
    except Exception as e:
        print(f"错误：应用 CLAHE 失败: {e}。返回去噪后的图像。")
        clahe_image = denoised # 如果 CLAHE 失败，至少返回去噪图

    # 可选：Gamma 校正 (调整亮度)
    # gamma_corrected = exposure.adjust_gamma(clahe_image, gamma=1.0)

    return clahe_image

def extract_hog_features(image):
    """
    从给定图像块中提取 HOG 特征。
    重要：输入图像应已被调整到 TRAIN_IMG_SIZE。
    Args:
        image: 输入的单通道图像块 (应为 TRAIN_IMG_SIZE 大小)。
    Returns:
        一维 HOG 特征向量 (numpy array)，或在失败时返回 None。
    """
    if image is None:
        print("错误：输入图像为空 (extract_hog_features)")
        return None
    if image.shape[0] != TRAIN_IMG_SIZE[1] or image.shape[1] != TRAIN_IMG_SIZE[0]:
         # print(f"警告: HOG 输入图像尺寸 ({image.shape}) 与 TRAIN_IMG_SIZE ({TRAIN_IMG_SIZE}) 不符。将尝试调整。")
         try:
             image = cv2.resize(image, TRAIN_IMG_SIZE)
         except Exception as e:
             print(f"错误: 调整 HOG 输入图像大小失败: {e}")
             return None

    # 确保图像是适合 HOG 的类型 (通常是灰度图)
    if len(image.shape) > 2:
        print("警告：HOG 输入图像不是单通道，将尝试转为灰度图。")
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"错误: 转换 HOG 输入图像为灰度图失败: {e}")
            return None

    # 提取 HOG 特征
    try:
        features = hog(image,
                       orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK,
                       block_norm=HOG_BLOCK_NORM,
                       visualize=False,        # 我们只需要特征向量
                       feature_vector=True,    # 返回一维特征向量
                       transform_sqrt=False)   # Gamma 校正，有时需要，有时不需要
    except Exception as e:
        print(f"错误: HOG 特征提取失败: {e}")
        return None

    return features

def get_orientation_and_centroid(roi):
    """
    计算二值化 ROI 中最大轮廓的方向和质心。
    使用轮廓点的 PCA (主成分分析) 来确定主方向。
    Args:
        roi: 感兴趣区域 (建议传入单通道灰度图)。
    Returns:
        tuple: (angle, cx, cy)
               angle (float): 轮廓主方向与水平轴的夹角 (0-180 度)。
               cx (int): 轮廓质心的 x 坐标 (相对于 roi 左上角)。
               cy (int): 轮廓质心的 y 坐标 (相对于 roi 左上角)。
               如果失败则返回 (0, roi中心x, roi中心y)。
    """
    default_cx = roi.shape[1] // 2
    default_cy = roi.shape[0] // 2

    if roi is None or roi.size == 0:
        print("警告：用于计算方向的 ROI 为空。")
        return 0.0, default_cx, default_cy

    # 1. 确保 ROI 是 8 位单通道灰度图
    if len(roi.shape) > 2 or roi.dtype != np.uint8:
        try:
            if len(roi.shape) > 2:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                # 尝试转换类型，例如从 float 转为 uint8
                if roi.max() <= 1.0 and roi.min() >= 0: # 假设是 [0, 1] 范围的 float
                     roi_gray = (roi * 255).astype(np.uint8)
                else: # 否则直接转换
                     roi_gray = roi.astype(np.uint8)
        except Exception as e:
            print(f"警告：无法将 ROI 转换为 8 位灰度图: {e}。返回默认值。")
            return 0.0, default_cx, default_cy
    else:
        roi_gray = roi

    # 2. 二值化：使用 Otsu's 方法自动确定阈值
    # THRESH_BINARY_INV 适用于目标比背景暗的情况（例如黑色锁孔在亮背景上）
    # THRESH_BINARY 适用于目标比背景亮的情况
    # 需要根据实际锁孔和背景调整
    threshold_value, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 可选：形态学操作去噪（开运算去小噪点，闭运算连接断裂）
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. 查找轮廓
    # RETR_EXTERNAL: 只查找最外层轮廓
    # CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段，只保留端点
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # print("警告：在 ROI 中未找到轮廓。计算二值图像质心作为替代。")
        M = cv2.moments(binary)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else default_cx
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else default_cy
        return 0.0, cx, cy # 返回默认角度

    # 4. 找到面积最大的轮廓
    try:
        max_contour = max(contours, key=cv2.contourArea)
    except Exception as e:
         print(f"警告：寻找最大轮廓时出错: {e}")
         return 0.0, default_cx, default_cy

    # 5. 计算最大轮廓的质心
    M = cv2.moments(max_contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else default_cx
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else default_cy

    # 6. 使用 PCA 计算方向 (需要至少 5 个点)
    if len(max_contour) < 5:
        # print("警告：最大轮廓点数过少 (<5)，无法进行 PCA 方向计算。")
        return 0.0, cx, cy # 点太少，无法确定方向，返回默认角度

    try:
        # 将轮廓点转换为 PCA 需要的格式 (Nx2 的 float32 数组)
        data_pts = max_contour.reshape(-1, 2).astype(np.float32)
        # 执行 PCA，计算均值和特征向量
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=None)

        # 主方向（方差最大的方向）对应于第一个特征向量 eigenvectors[0]
        # 特征向量表示方向 (dx, dy)
        vec = eigenvectors[0]
        # 计算主轴与水平轴的夹角 (atan2 返回弧度，覆盖 -pi 到 pi)
        angle_rad = math.atan2(vec[1], vec[0])
        # 转换为角度
        angle_deg = math.degrees(angle_rad)

        # 将角度归一化到 0-180 度范围 (表示无方向的对称轴)
        # 例如，30 度和 210 度表示同一条线
        orientation_angle = abs(angle_deg) % 180.0

    except Exception as e:
        print(f"警告：PCA 计算失败: {e}")
        return 0.0, cx, cy # PCA 出错，返回默认角度

    return orientation_angle, cx, cy

def pyramid(image, scale=1.5, minSize=(30, 30)):
    """
    构建图像金字塔。
    Args:
        image: 输入图像。
        scale (float): 金字塔的缩放因子 (> 1)。每次缩小的比例。
        minSize (tuple): 金字塔层的最小尺寸 (width, height)。低于此尺寸停止。
    Yields:
        缩放后的图像层。从原始图像开始，逐渐变小。
    """
    # 产生原始图像
    yield image
    # 循环构建金字塔
    while True:
        # 计算新图像的尺寸并进行缩放
        new_w = int(image.shape[1] / scale)
        new_h = int(image.shape[0] / scale)
        # 提供明确的插值方法，例如 INTER_AREA 适合缩小
        try:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"错误：图像金字塔缩放失败: {e}")
            break # 停止生成

        # 如果缩放后的图像小于最小尺寸，停止构建
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # 产生下一层图像
        yield image

def sliding_window(image, stepSize, windowSize):
    """
    在图像上滑动窗口。
    Args:
        image: 输入图像。
        stepSize (int): 窗口滑动的步长 (像素)。
        windowSize (tuple): 窗口的尺寸 (width, height)。
    Yields:
        tuple: (x, y, window)
               x, y: 当前窗口左上角的坐标。
               window: 当前窗口区域的图像切片。
    """
    # 沿 y 方向滑动
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize):
        # 沿 x 方向滑动
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
            # 产生当前窗口
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def nms(boxes, scores, threshold=0.3):
    """
    非极大值抑制 (Non-Maximum Suppression)。
    Args:
        boxes (numpy.ndarray): 边界框数组，形状为 (N, 4)，格式为 [x, y, w, h]。
        scores (numpy.ndarray): 每个边界框对应的置信度分数，形状为 (N,)。
        threshold (float): IoU (Intersection over Union) 阈值。
                           重叠超过此阈值的框会被抑制。
    Returns:
        tuple: (picked_boxes, picked_scores)
               picked_boxes: 经过 NMS 后保留下来的边界框。
               picked_scores: 对应保留下来的边界框的分数。
               如果输入为空，返回空的 numpy 数组。
    """
    if boxes is None or len(boxes) == 0:
        return np.array([]), np.array([])

    # 确保输入是 NumPy 数组
    boxes = np.array(boxes)
    scores = np.array(scores)

    # 获取坐标: x1, y1 (左上角), x2, y2 (右下角)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    # 注意：这里假设 boxes 是 [x, y, w, h] 格式
    x2 = x1 + boxes[:, 2] # w -> x2
    y2 = y1 + boxes[:, 3] # h -> y2

    # 计算每个框的面积
    areas = (boxes[:, 2]) * (boxes[:, 3]) # area = w * h

    # 按分数降序排序，获取索引
    order = scores.argsort()[::-1]

    keep_indices = [] # 用于存储最终保留的框的索引
    while order.size > 0:
        # 1. 选择当前分数最高的框
        i = order[0]
        keep_indices.append(i) # 保留这个框

        # 2. 获取剩余框的索引 (order[1:])
        remaining_indices = order[1:]
        if remaining_indices.size == 0:
            break # 没有剩余的框了

        # 3. 计算当前框 i 与所有剩余框的交集区域坐标
        #    找到交集矩形的左上角 (xx1, yy1) 和右下角 (xx2, yy2)
        xx1 = np.maximum(x1[i], x1[remaining_indices])
        yy1 = np.maximum(y1[i], y1[remaining_indices])
        xx2 = np.minimum(x2[i], x2[remaining_indices])
        yy2 = np.minimum(y2[i], y2[remaining_indices])

        # 4. 计算交集区域的宽和高
        #    确保宽高不为负（无交集时为0）
        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)

        # 5. 计算交集面积
        inter_area = w_inter * h_inter

        # 6. 计算并集面积: Area(A) + Area(B) - Area(A∩B)
        #    添加一个小的 epsilon 防止除以零
        union_area = areas[i] + areas[remaining_indices] - inter_area + 1e-6

        # 7. 计算 IoU (Intersection over Union)
        iou = inter_area / union_area

        # 8. 找到与当前框 i 的 IoU 小于等于阈值的那些框的索引
        #    (这些框与当前框重叠不大，需要保留下来继续比较)
        #    `np.where` 返回满足条件的索引 (相对于 iou 数组，也就是相对于 remaining_indices)
        inds_to_keep = np.where(iou <= threshold)[0]

        # 9. 更新 order，只保留 IoU 小于阈值的框的索引
        #    inds_to_keep 是相对于 remaining_indices (即 order[1:]) 的索引
        #    所以要用 order[1:] 来选择这些索引对应的原始 order 中的值
        order = order[inds_to_keep + 1] # +1 是因为 inds_to_keep 是从 order[1:] 开始的

    # 返回保留下来的框和它们的分数
    return boxes[keep_indices], scores[keep_indices]

