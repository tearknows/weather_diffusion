import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
import torch.nn.functional as F
from datetime import datetime
from models.unet import DiffusionUNet


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm

# 数据预处理函数：将输入数据进行转换
def data_transform(X):
    return 2 * X - 1.0

# 数据逆预处理函数：将数据进行反转换，保证在 [0, 1] 范围内
def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

# 指数移动平均（EMA）辅助类，用于模型参数的平均更新
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    # 注册模型，用于获取模型的参数
    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    # 更新EMA，通过模型的参数进行更新
    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    # 使用EMA更新模型参数
    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    # 复制一个具有相同EMA参数的模型
    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    # 返回EMA的当前状态字典
    def state_dict(self):
        return self.shadow

    # 加载EMA的状态字典
    def load_state_dict(self, state_dict):
        self.shadow = state_dict

# 获取beta值的调度表
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# 原计算噪声估计的损失MSE
# def noise_estimation_loss(model, x0, t, e, b):
#     a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
#     x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
#     output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float())
#     return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

#KL损失（可能有误，效果不佳，不知原因）
def noise_estimation_loss(model, x0, t, e, b):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float())
    # 检查这一行之前的代码，确保没有语法错误
    model_probs = F.softmax(output.view(-1), dim=0)
    e_probs = F.softmax(e.view(-1), dim=0)
    kl_loss = F.kl_div(model_probs.log(), e_probs, reduction='batchmean')
    return kl_loss

# DenoisingDiffusion 类，用于训练模型和生成样本
class DenoisingDiffusion(object):
    #初始化模型、EMAHelper、优化器
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        # 创建DiffusionUNet模型
        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        # 创建EMAHelper对象
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        # 获取优化器
        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        # 获取beta值的调度表
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    #加载预训练的模型和优化器
    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    #训练模型的主要逻辑，包括数据加载、模型训练、EMA更新
    def train(self, DATASET):
        # 启用 cuDNN 的自动调整功能，提高训练速度
        cudnn.benchmark = True

        # 获取训练和验证数据的加载器
        train_loader, val_loader = DATASET.get_loaders()

        # 如果存在模型的断点文件，则加载模型的权重和优化器状态
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        # 循环迭代训练数据集的每个 epoch
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0

            # 遍历训练数据集中的每个批次
            for i, (x, y) in enumerate(train_loader):
                # 如果输入数据的维度为5，则展平
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                
                # 设置模型为训练模式
                self.model.train()
                self.step += 1

                # 将输入数据移动到设备（GPU）
                x = x.to(self.device)
                x = data_transform(x)

                # 生成与输入数据相同大小的随机张量
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                # 对时间步进行对称采样
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                # 计算损失
                loss = noise_estimation_loss(self.model, x, t, e, b)

                # 每 10 步输出损失信息
                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                # 梯度清零，执行反向传播，更新权重
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 使用 EMA Helper 更新模型的移动平均值
                self.ema_helper.update(self.model)
                data_start = time.time()

                # 如果达到验证的频率，则在验证集上执行模型的评估
                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)
                
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S") # 加时间
                # 加入时间，分别保存训练后的模型
                # 如果达到快照的频率或是第一步，则保存当前模型的权重、优化器状态、EMA Helper 状态以及其他相关信息
                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.save_dir, 'ckpts', self.config.data.dataset + '_ddpm',timestamp))
                fn=os.path.join(self.args.train_dir, 'ckpts', self.config.data.dataset + '_ddpm',timestamp)
                print(fn,"已保存")
    #生成样本图像
    '''
        x_cond:条件输入，是噪声图像的条件部分。在 DDM 中，通常是输入图像的前三个通道。

        x:生成样本的初始噪声图像。在 DDM 中，通常是一个与输入图像相同形状的噪声张量。

        last:一个布尔值，指示是否返回生成序列的最后一帧。

        patch_locs:如果你使用了 utils.sampling.generalized_steps_overlapping,这是一个指定生成图像局部补丁位置的参数。

        patch_size:如果你使用了 utils.sampling.generalized_steps_overlapping,这是一个指定生成图像局部补丁大小的参数。
    '''
    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        # 计算时间步的间隔，用于采样生成样本时的时间步
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        # 创建时间步序列，从0到num_diffusion_timesteps，间隔为skip
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        
        # 根据是否指定了patch_locs选择不同的采样方法
        if patch_locs is not None:
            # 使用局部重叠采样方法生成样本序列
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            # 使用常规采样方法生成样本序列
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        
        # 根据last参数决定是否返回生成序列的最后一帧
        if last:
            xs = xs[0][-1]
        return xs   # 返回生成的样本序列或最后一帧
    
    #在验证集上生成图像样本
    '''
        val_loader:验证集的数据加载器，用于遍历验证集中的样本。

        step:当前训练的步数，用于构造保存生成图像的文件夹路径。
    '''
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
