import torch
import torch.nn as nn
import utils
import torchvision
import os

# 将输入数据规范化到[-1, 1]的范围
def data_transform(X):
    return 2 * X - 1.0

# 将规范化后的数据逆转回[0, 1]的范围
def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

# 差分扩散图像恢复（Diffusive Restoration）类,用于在给定预训练的差分扩散模型的情况下，对输入图像进行恢复。
class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion
        
        # 如果提供了预训练的差分扩散模型路径，则加载模型
        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    # 对输入的验证集进行图像恢复
    def restore(self, val_loader, validation='snow', r=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                print(f"starting processing from image {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                
                # 保存恢复的图像
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}_output.png"))

    # 差分扩散图像恢复函数
    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    # 计算重叠网格的索引
    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
# 代码主要实现了以下功能：

# 将输入数据进行规范化和逆规范化，确保在[-1, 1]和[0, 1]的范围内。
# 创建了一个 DiffusiveRestoration 类，用于加载预训练的差分扩散模型，并在给定的验证集上进行图像恢复。
# restore 函数用于遍历验证集中的图像并保存差分扩散恢复的结果。
# diffusive_restoration 函数实际执行差分扩散图像恢复的过程，使用提供的差分扩散模型。
# overlapping_grid_indices 函数用于计算重叠网格的索引，以便将图像分割为重叠的块，用于差分扩散的采样。
# 使用方法：

# 创建一个 DiffusiveRestoration 实例，传入差分扩散模型、命令行参数和配置。
# 调用 restore 函数，传入验证集的数据加载器和选择的验证类型（例如，'snow'）。
# 在 restore 函数中，差分扩散模型将用于恢复验证集中的每个图像，并保存在指定的输出文件夹中。