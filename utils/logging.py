import torch
import shutil
import os
import torchvision.utils as tvu

# 保存图片
def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        # 如果不存在该路径，则创造多层目录
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)

# 保存模型
def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        # 如果不存在该路径，则创造多层目录
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')

# 加载模型
# device 表示模型加载后将在哪个计算设备上运行，例如CPU或GPU。
def load_checkpoint(path, device):
    # 如果device为None，则表示在原始保存时使用的设备上运行；
    # 如果指定了设备，那么模型将被加载到该设备上。
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
