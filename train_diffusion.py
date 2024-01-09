import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion

# 解析命令行参数和配置文件
def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    # 配置文件的路径
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    # 恢复训练时的检查点路径
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    # 用于验证图像补丁的隐式采样步骤数
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    # 保存还原的验证图像的补丁的文件夹路径
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored validation image patches")
    # 初始化训练的随机种子
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    # 从配置文件中加载配置
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

# 将配置字典转换为命名空间
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    # 解析命令行参数和配置
    args, config = parse_args_and_config()

    # 设置设备
    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # 设置随机种子
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # 加载数据
    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)

    # 创建模型
    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(DATASET)



if __name__ == "__main__":
    main()


# 这段代码主要实现了以下功能：

# 通过命令行参数指定配置文件的路径，配置文件中包含了训练模型的各种参数。
# 创建命名空间以保存命令行参数和配置信息。
# 设置设备（GPU或CPU）、随机种子，并加载数据集。
# 创建并训练基于补丁的去噪扩散模型。
    
# 使用方法：
# python train_diffusion.py --config E:\deeplearning\last\WeatherDiffusion\configs\allweather.yml
# 在命令行中执行脚本，并通过 --config 参数指定配置文件的路径。
# 具体的配置信息可以在配置文件中进行设置，包括数据集类型、模型参数等。
# 可以选择是否从先前保存的检查点处继续训练，通过 --resume 参数指定检查点路径。
# 训练过程中，模型的训练日志和结果图像将会保存在指定的文件夹中。