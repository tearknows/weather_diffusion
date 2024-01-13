import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class AllWeather:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='snow'):
        if validation == 'raindrop':
            print("=> evaluating raindrop test set...")
            path = os.path.join(self.config.data.data_dir, 'data', 'raindrop', 'test')
            filename = 'raindroptesta.txt'
        elif validation == 'rainfog':
            print("=> evaluating outdoor rain-fog test set...")
            path = os.path.join(self.config.data.data_dir, 'data', 'outdoor-rain')
            filename = 'test1.txt'
        else:
            # 测试集，路径需要修改
            print("=> evaluating snowtest100K-L...")
            path = os.path.join(self.config.data.data_dir, 'data', 'snow')
            filename = 'snowtest100k_L.txt'

        # 训练集，路径需要修改
        train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, 'data', 'allweather'),
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          filelist='allweather.txt',
                                          parse_patches=parse_patches)
        val_dataset = AllWeatherDataset(path, n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=filename,
                                        parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()

        self.dir = dir
        train_list = os.path.join(dir, filelist)
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)


    #get_images 函数根据给定索引加载输入图像和目标图像，并返回它们的张量表示
    def get_images(self, index):
        # 获取输入图像和目标图像的文件名
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        # 从文件名中提取图像的标识符
        img_id = re.split('/', input_name)[-1][:-4]
        # 打开输入图像，如果指定了文件夹路径，则拼接路径
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            # 尝试打开目标图像，如果失败则将其转换为RGB模式再次尝试
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        # 如果 parse_patches 为真，表示要解析图像补丁，将图像切割为多个部分，然后将输入和目标张量拼接成列表。

        if self.parse_patches:
            # 如果要解析图像补丁，进行图像切割
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            # 将图像转换为张量，并拼接成包含输入和目标的列表
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        # 如果 parse_patches 为假，表示进行整体图像恢复，将整个图像调整为16的倍数。
        else:
            # 对整个图像进行整体恢复，将图像大小调整为16的倍数
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            # 调整图像大小
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            # 将图像转换为张量，并拼接成包含输入和目标的列表
            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

