# 锁孔识别系统 (HOG + SVM) - 完整版

## 项目概述
本项目使用经典的计算机视觉技术实现了一个锁孔识别系统。主要采用方向梯度直方图 (HOG) 作为特征描述子，并结合线性支持向量机 (LinearSVC) 进行分类训练。检测阶段使用图像金字塔和滑动窗口策略来定位不同大小和位置的锁孔。系统能够输出检测到的锁孔的边界框、估计的质心位置、相对于图像中心点的距离以及估计的主方向角度。

**核心技术栈:**
*   特征提取: HOG (Histogram of Oriented Gradients)
*   分类器: LinearSVC (线性支持向量机)
*   检测策略: 图像金字塔 + 滑动窗口 + 非极大值抑制 (NMS)
*   主要库: OpenCV, Scikit-learn, Scikit-image, NumPy

**精度提示:** 本代码提供了一个完整的框架。最终的识别精度**高度依赖**于以下因素：
1.  **训练数据的质量与数量:** 需要大量、多样化、清晰标注的正样本（锁孔）和负样本（背景）。
2.  **参数调优:** 包括 HOG 参数、SVM 参数 (如 C 值)、检测参数（置信度阈值、NMS 阈值、窗口步长、金字塔比例）等，都需要根据实际数据进行仔细调整。
3.  **数据增强:** 合理的数据增强可以显著提升模型的泛化能力。

## 项目结构
├── data/ # 存放训练数据
│ ├── positive/ # 正样本锁孔图像 (例如, 64x64px)
│ └── negative/ # 负样本背景图像 (例如, 64x64px)
├── input_images/ # 存放待检测的输入图像
├── models/ # 训练后保存的模型 (.pkl) 和标准化器 (.pkl)
├── src/ # 源代码目录
│ ├── init.py # 包标记文件
│ ├── utils.py # 工具函数 (预处理, HOG, 方向, NMS, 滑窗, 金字塔)
│ ├── train.py # SVM 模型训练脚本
│ └── detect.py # 锁孔检测脚本
├── detection_visualizations/ # (可选) 检测结果可视化图片输出目录
├── run.sh # 主运行脚本 (训练 -> 检测)
├── requirements.txt # Python 依赖列表
└── README.md # 本说明文档

## 环境设置与运行

1.  **克隆或下载项目:** 获取所有文件并按上述结构组织。
2.  **安装 Python:** 建议使用 Python 3.8 或更高版本。
3.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
4.  **安装依赖:** 在项目根目录 (`lockhole_recognition/`) 运行：
    ```bash
    pip install -r requirements.txt
    ```
5.  **准备训练数据 (关键步骤!):**
    *   在 `data/positive/` 目录下放入**大量且多样化**的正样本锁孔图像。图像应裁剪到大致只包含锁孔本身，尺寸建议与 `src/utils.py` 中的 `TRAIN_IMG_SIZE` (默认为 64x64) 一致。覆盖不同光照、角度、类型的锁孔。
    *   在 `data/negative/` 目录下放入**大量且多样化**的负样本背景图像。尺寸与正样本一致。包含各种非锁孔的纹理、物体、场景，特别是容易与锁孔混淆的区域（如螺丝孔、阴影、标记点等）。
    *   **数据量建议:** 正负样本数量级最好在几百到几千张，质量优先。

6.  **准备测试图像:**
    *   将您想要检测锁孔的图像放入 `input_images/` 目录。

7.  **运行系统:**
    *   在项目根目录 (`lockhole_recognition/`) 打开终端。
    *   **赋予 `run.sh` 执行权限:** `chmod +x run.sh`
    *   **执行脚本:** `./run.sh`
    *   该脚本将自动执行以下步骤：
        *   调用 `src/train.py` 训练模型，并将模型 (`svm_model.pkl`) 和标准化器 (`scaler.pkl`) 保存到 `models/` 目录。
        *   调用 `src/detect.py` 对 `input_images/` 目录中的所有图像进行锁孔检测。
        *   将所有检测结果汇总并保存到项目根目录下的 `results.json` 文件中。
        *   如果 `src/detect.py` 中的 `SAVE_VISUALIZATIONS` 设置为 `True`，检测的可视化图像将保存在 `detection_visualizations/` 目录。

## 结果说明 (`results.json`)

输出的 JSON 文件是一个包含多个检测结果对象的列表。每个对象代表一个检测到的锁孔，结构如下：

```json
[
    {
        "filename": "image_01.jpg", // 检测到的图像文件名
        "box": {                   // 边界框坐标和尺寸 (相对于图像左上角)
            "x": 152,              // 左上角 x 坐标
            "y": 210,              // 左上角 y 坐标
            "w": 68,               // 宽度
            "h": 71                // 高度
        },
        "centroid": {              // 估计的质心坐标 (相对于图像左上角)
             "x": 186,
             "y": 245
        },
        "confidence": 1.8765,      // 检测置信度 (来自 SVM decision_function)
        "distance_from_center": 55.8, // 质心到图像中心的像素距离
        "orientation_angle": 88.5,   // 估计的主方向角度 (0-180度)
        "is_lock_original": true,  // 占位符 - 是否为原装锁 (需额外逻辑)
        "is_key_in": false,        // 占位符 - 是否有钥匙插入 (需额外逻辑/传感器)
        "key_angle": 0             // 占位符 - 钥匙角度 (需额外逻辑/传感器)
    },
    // ... 可能有更多检测结果来自同一图像或其他图像
]

