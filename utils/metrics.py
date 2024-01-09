import cv2
import numpy as np
import torch

# This script is adapted from the following repository: https://github.com/JingyunLiang/SwinIR
# 这个文件用来计算检测指标的函数，psnr，ssim

def calculate_psnr(img1, img2, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    函数名:calculate_psnr
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    参数：
    img1:第一个输入图像,类型为ndarray(NumPy数组),像素值的范围应在[0, 255]之间。
    img2:第二个输入图像,同样为ndarray类型,像素值范围也在[0, 255]之间。
    test_y_channel:一个布尔值,表示是否在Y通道上进行测试。默认为False。

    Returns:
        float: psnr result.
    返回值:计算得到的PSNR值,返回类型为float
    """
    
    # 确保输入的两个图像形状相同，否则会触发断言错误并输出图像形状的不同信息。
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    # 确保输入的图像具有3个通道（RGB图像），否则会触发断言错误。
    assert img1.shape[2] == 3
    # 转换输入图像的数据类型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Y通道测试
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    # 计算两个图像之间的均方误差
    mse = np.mean((img1 - img2) ** 2)
    
    # 如果均方误差为零，说明两个图像完全相同，此时PSNR被定义为正无穷。
    if mse == 0:
        return float('inf')
    # 根据PSNR的定义，计算最终的PSNR值，其中255是像素值的范围。返回计算得到的PSNR值。
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    用于计算两个单通道图像之间的结构相似性指数(SSIM)
    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 生成高斯滤波器窗口
    # 使用OpenCV的getGaussianKernel函数生成一个大小为11×11的高斯滤波器核。
    kernel = cv2.getGaussianKernel(11, 1.5)
    # 生成高斯滤波器窗口，通过对高斯核的外积得到
    window = np.outer(kernel, kernel.transpose())

    # 图像均值的计算
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    # 其他中间结果的计算
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # SSIM值计算
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=False):
    """Calculate SSIM (structural similarity).
    用于计算结构相似性指数(SSIM)的平均值,可适用于单通道或三通道图像。
    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    # 确保输入的两个图像形状相同，否则会触发断言错误并输出图像形状的不同信息。
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    
    # 确保输入的图像为三通道的彩色图像，否则会触发断言错误
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    # 单通道SSIM计算
    # 创建一个空列表，用于存储每个通道的SSIM值
    ssims = []
    for i in range(img1.shape[2]):
        # 调用之前定义的_ssim函数，计算每个通道的SSIM值，并将结果添加到列表中。
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    
    # 将每个通道的SSIM值存储在NumPy数组中，然后计算这些值的平均值，最终返回平均的SSIM值。
    return np.array(ssims).mean()


def to_y_channel(img):
    """Change to Y channel of YCbCr.
    # 将输入图像转换为Y通道(亮度通道)的YCbCr颜色空间
    
    Args:
        img (ndarray): Images with range [0, 255].

    返回值:转换后的图像,数据类型为ndarray,像素值范围在[0, 255]（浮点数类型），且不进行四舍五入。
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """

    # 数据类型转换
    img = img.astype(np.float32) / 255.

    #检查图像是否为3通道的彩色图像，如果是，进行以下操作
    if img.ndim == 3 and img.shape[2] == 3:
        # 将BGR彩色图像转换为YCbCr颜色空间，仅保留亮度通道（Y通道）
        img = bgr2ycbcr(img, y_only=True)
        # 添加一个新的维度，将图像从2D转换为3D，以匹配原始图像的形状。
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """Convert the type and range of the input image.
    将输入图像的数据类型和像素值范围进行转换
    
    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.
    它将输入图像转换为 np.float32 类型和范围 [0, 1]。
    它主要用于在色彩空间中对输入图像进行预处理
    转换函数，例如 RGB2YCBCR 和 YCBCR2RGB。

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    返回值:转换后的图像,数据类型为np.float32,像素值范围在[0, 1]之间。
    """

    # 保存输入图像的数据类型
    img_type = img.dtype

    # 将输入图像的数据类型转换为np.float32。
    img = img.astype(np.float32)
    
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].
        目标类型,可以是np.uint8或np.float32。
        如果是np.uint8,则将图像转换为np.uint8类,像素值范围为[0, 255];
        如果是np.float32,则将图像转换为np.float32类型,像素值范围为[0, 1]。
    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        # 将图像的浮点数像素值四舍五入为最接近的整数，以确保在转换为np.uint8类型时不会丢失信息。
        img = img.round()
    else:
        # 将图像像素值范围缩放到[0, 1]之间，以确保在转换为np.float32类型时符合该类型的范围要求。
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.
    将BGR格式的图像转换为YCbCr颜色空间

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        
        y_only (bool): Whether to only return Y channel. Default: False.
        一个布尔值,表示是否仅返回Y通道。默认为False,即返回完整的YCbCr图像

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    返回值:转换后的YCbCr图像,其数据类型和像素值范围与输入图像相同。
    """

    # 保存输入图像的数据类型
    img_type = img.dtype

    # 将输入图像的数据类型和像素值范围进行转换，使其符合处理的要求
    img = _convert_input_type_range(img)
    if y_only:
        # 通过乘以权重矩阵并加上偏移，将BGR图像转换为Y通道。
        # 这里使用的权重和偏移值是基于ITU-R BT.601标准的转换公式。
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        # 要返回完整的YCbCr图像
        # 通过矩阵相乘和加法，将BGR图像转换为完整的YCbCr图像，包括Y、Cb和Cr通道。
        # 这里的矩阵和偏移值同样基于ITU-R BT.601标准。
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    
    # 将输出图像的数据类型和像素值范围转换为与输入图像相同。
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img
