import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration

# 解析命令行参数和配置文件
def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    # 配置文件的路径
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    # 添加命令行参数，用于指定预训练模型的检查点路径，可选，默认为空字符串
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    # 添加命令行参数，指定图像块之间的重叠的网格单元宽度 r，可选，默认为 16
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    # 添加命令行参数，指定隐式采样的步数，可选，默认为 25
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    # 添加命令行参数，指定恢复测试集的选项，可选，默认为 'raindrop'
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['raindrop', 'snow', 'rainfog']")
    # 添加命令行参数，指定保存恢复图像的位置，默认为 'results/images/'
    parser.add_argument("--image_folder", default='/gemini/output', type=str,
                        help="Location to save restored images")
    # 添加命令行参数，指定用于初始化训练的随机数生成的种子，默认为 61
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    # 解析命令行参数并将其存储在 args 对象中
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

# 测试只能用一个GPU
    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # 设置随机种子
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    # 通过字符串索引从 datasets 模块中选择指定的数据集类，然后创建该数据集的实例 DATASET
    DATASET = datasets.__dict__[config.data.dataset](config)
    # 调用 DATASET 对象的 get_loaders 方法，获取训练集和验证集的数据加载器
    # 参数 parse_patches 设置为 False，表示不进行图像块的解析
    # 参数 validation 设置为 args.test_set，表示使用指定的测试集（如 'raindrop', 'snow', 'rainfog'）
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    # 创建 DenoisingDiffusion 类的实例，传入命令行参数和配置参数
    diffusion = DenoisingDiffusion(args, config)
    # 创建 DiffusiveRestoration 类的实例，传入 DenoisingDiffusion 实例、命令行参数和配置参数
    model = DiffusiveRestoration(diffusion, args, config)
    # 使用 DiffusiveRestoration 类中的 restore 方法对验证集进行图像恢复
    model.restore(val_loader, validation=args.test_set, r=args.grid_r)


if __name__ == '__main__':
    main()
