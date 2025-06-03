#!/bin/bash

# --- run.sh ---
# 用于自动化执行训练和检测流程的脚本

# 确保脚本从其所在目录执行，保证相对路径正确
cd "$(dirname "$0")" || exit 1

echo "=========================================="
echo "  锁孔识别系统 - 自动化运行脚本"
echo "=========================================="
echo ""

# --- 检查 Python 环境 ---
if ! command -v python &> /dev/null
then
    echo "错误: 未找到 'python' 命令。请确保 Python 已安装并配置在 PATH 中。"
    exit 1
fi
echo "使用 Python 版本: $(python --version)"
echo ""

# --- 检查依赖 ---
echo "--- 检查 Python 依赖 (requirements.txt) ---"
# 可以在这里添加一个简单的检查，例如尝试导入一个关键库
if python -c "import cv2, sklearn, skimage, joblib" &> /dev/null; then
    echo "关键依赖似乎已安装。"
else
    echo "警告: 似乎缺少必要的 Python 库。"
    echo "请在项目根目录下运行 'pip install -r requirements.txt' 进行安装。"
    # 可以选择在这里退出，或者继续执行并让 Python 脚本报错
    # exit 1
fi
echo ""

# --- 步骤 1: 训练模型 ---
echo "--- [步骤 1/2] 开始训练 SVM 模型 (执行 src/train.py) ---"
# 执行训练脚本
python src/train.py

# 检查训练脚本的退出状态码
# $? 存储的是上一个命令的退出状态，0 表示成功，非 0 表示失败
if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 模型训练失败 (src/train.py 返回非零退出码)。"
    echo "请检查上面的训练日志输出以获取详细错误信息。"
    echo "脚本执行中止。"
    exit 1 # 训练失败则退出
fi
echo "--- [步骤 1/2] 模型训练似乎已完成 ---"
echo "(请检查日志确认是否成功保存模型文件到 models/ 目录)"
echo ""
# 短暂暂停，给用户看日志的时间
sleep 1

# --- 步骤 2: 检测锁孔 ---
echo "--- [步骤 2/2] 开始检测锁孔 (执行 src/detect.py) ---"
# 检查输入目录是否存在
if [ ! -d "input_images" ]; then
    echo "错误: 'input_images' 目录未找到。"
    echo "请创建该目录并将需要检测的图像放入其中。"
    echo "脚本执行中止。"
    exit 1
fi

# 检查输入目录中是否有图像文件 (可选，但推荐)
if ! ls input_images/*.{png,jpg,jpeg,bmp,tif,tiff} &> /dev/null; then
    echo "警告: 在 'input_images' 目录中未找到支持的图像文件。"
    echo "检测脚本将继续运行，但可能不会处理任何文件。"
fi

# 执行检测脚本
python src/detect.py

# 检查检测脚本的退出状态码
if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 锁孔检测失败 (src/detect.py 返回非零退出码)。"
    echo "请检查上面的检测日志输出以获取详细错误信息。"
    echo "脚本执行中止。"
    exit 1 # 检测失败则退出
fi
echo "--- [步骤 2/2] 锁孔检测似乎已完成 ---"
echo "(请检查日志和生成的 results.json 文件及 detection_visualizations/ 目录)"
echo ""

# --- 完成 ---
echo "=========================================="
echo "  所有步骤执行完毕。"
if [ -f "results.json" ]; then
    echo "  检测结果 (JSON) 已保存至: results.json"
fi
if [ -d "detection_visualizations" ] && [ "$(ls -A detection_visualizations)" ]; then
    echo "  可视化图像已保存至: detection_visualizations/"
fi
echo "=========================================="
echo ""

exit 0 # 所有步骤成功完成
