import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim

# Sample script to calculate PSNR and SSIM metrics from saved images in two directories
# using calculate_psnr and calculate_ssim functions from: https://github.com/JingyunLiang/SwinIR

# 指定保存图像的路径（模型输出和真实标签）
gt_path = '/PATH/TO/GROUND_TRUTH/'
results_path = '/PATH/TO/MODEL_OUTPUTS/'

# 获取模型输出和真实标签的图像文件名列表
imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))

# 确保模型输出和真实标签图像文件数量一致
assert len(imgsName) == len(gtsName)

# 初始化累积的 PSNR 和 SSIM
cumulative_psnr, cumulative_ssim = 0, 0

# 遍历每一对模型输出和真实标签图像
for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    
    # 读取模型输出和真实标签图像
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
    # 计算当前图像的 PSNR 和 SSIM
    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
    # 输出当前图像的 PSNR 和 SSIM
    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    # 累积 PSNR 和 SSIM
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)))
print(results_path)

# PSNR（Peak Signal-to-Noise Ratio）和 SSIM（Structural Similarity Index）是用于评估图像质量的两个常见指标。
# PSNR (Peak Signal-to-Noise Ratio):
# 定义： PSNR是通过测量图像的峰值信噪比来评估图像质量的指标。峰值信噪比越高，图像质量越好。

# SSIM (Structural Similarity Index):
# 定义： SSIM是一种结构相似性指数，用于测量两个图像之间的结构相似性，即它们的结构和内容有多相似。
# 计算方式： 其计算包括亮度相似性（Luminance Similarity）、对比度相似性（Contrast Similarity）和结构相似性（Structure Similarity）。SSIM的取值范围为[-1, 1]，越接近1表示两个图像越相似。