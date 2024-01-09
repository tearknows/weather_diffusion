import math
import torch
import torch.nn as nn

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm

# 生成时间步长的嵌入
# 目的是将时间步长的信息编码成模型可以理解的形式，以便在模型中使用
# timesteps: 输入的时间步长，是一个一维的 PyTorch 张量。
# embedding_dim: 嵌入的维度，即每个时间步长被编码成的特征的维度。
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    # 断言，确保输入的 timesteps 是一维的，即一个一维张量
    assert len(timesteps.shape) == 1

    # embedding_dim 被用于指定嵌入的维度。
    # half_dim 表示嵌入维度的一半，用于后续的计算。
    half_dim = embedding_dim // 2

    # emb 是一个对数间隔的指数项，通过 torch.exp 计算，形成一个指数增长的序列。
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    
    # 将 timesteps 乘以 emb，生成一个形状为 (时间步长数, embedding_dim/2) 的嵌入矩阵
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]

    # 使用正弦和余弦函数对嵌入矩阵进行转换，最终得到一个形状为 (时间步长数, embedding_dim) 的嵌入矩阵
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    # 如果 embedding_dim 是奇数，通过在矩阵的最后一列进行零填充，使其成为偶数维度。
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

# 非线性激活函数，实现swish激活函数
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

# 执行组归一化的函数
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# 上采样类
# Upsample 类的目的是在 U-Net 架构中的上采样阶段使用。
# 在 U-Net 中，上采样的操作被用来逐步扩大特征图的空间维度。
# 在上采样过程中，可以选择是否使用卷积。
# 卷积的加入有助于引入非线性变换，从而提高模型的表达能力。
class Upsample(nn.Module):
    # in_channels: 输入数据的通道数，即输入张量的深度。
    # with_conv: 一个布尔值，指示是否在上采样后应用卷积。
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    # 接受一个输入张量 x，表示需要上采样的特征图
    def forward(self, x):
        x = torch.nn.functional.interpolate(
            # 将输入张量的大小乘以2（scale_factor=2.0）
            # 上采样模式使用 "nearest"，表示使用最近邻插值。
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

# 下采样类
# Downsample 类的目的是在 U-Net 架构中的下采样阶段使用。
# 在 U-Net 中，下采样的操作被用来逐步减小特征图的空间维度，同时增加特征图的通道数。这有助于模型学习更高级别、全局的特征。
# 在下采样过程中，可以选择是否使用卷积。
# 卷积的加入可以用来捕获更复杂的特征信息。
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            # 由于 PyTorch 默认的卷积操作不支持非对称填充
            # 因此在这个类中使用了自定义填充来实现下采样卷积。
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # 如果 with_conv 为 False，则使用平均池化进行下采样。
            # 池化操作的作用是在每个区域中取平均值，从而降低特征图的维度
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

# 残差块类，包含两个卷积层和一些归一化
class ResnetBlock(nn.Module):
    '''
    接受以下参数：
        in_channels: 输入数据的通道数，即输入张量的深度。
        out_channels (可选): 输出数据的通道数，即输出张量的深度。如果未提供，则默认为输入通道数，这样就保持通道数不变。
        conv_shortcut: 一个布尔值，指示是否使用卷积进行残差连接。
            如果为 True,则使用卷积进行残差连接；
            如果为 False,则使用 1x1 卷积(NIN,Network in Network)进行残差连接。
        dropout: 在残差块中使用的丢弃率。
        temb_channels: 时间步嵌入的通道数。
    '''
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    # 接受两个输入张量 x 和 temb，分别表示残差块的输入和时间步嵌入。
    def forward(self, x, temb):
        h = x
        
        # 对归一化后的输入进行非线性变换
        h = self.norm1(h)
        h = nonlinearity(h)

        # 第一个卷积层 (self.conv1)，使用3x3的卷积核，stride为1，padding为1，以保持特征图的大小不变
        h = self.conv1(h)

        # 将时间步嵌入与卷积结果相加，通过一个线性映射 (self.temb_proj)，并在最后两个维度上添加维度 ([:, :, None, None])。
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # 再次使用激活函数进行非线性变换
        h = self.norm2(h)
        h = nonlinearity(h)

        # 使用丢弃层 (self.dropout) 随机置零一些元素，以防止过拟合。
        h = self.dropout(h)

        # 应用第二个卷积层 (self.conv2)，同样使用3x3的卷积核，stride为1，padding为1
        h = self.conv2(h)

        # 如果输入通道数和输出通道数不相等，则进行残差连接。
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # 使用卷积进行残差连接
                x = self.conv_shortcut(x)
            else:
                # 使用 1x1 卷积进行连接 
                x = self.nin_shortcut(x)

        return x+h

# 自注意力机制
'''
原理：
    自注意力机制允许输入序列中的每个元素都注意到其他元素的信息，其核心思想是通过计算权重，使得每个元素对输出的贡献不同。
    在这个模块中，通过计算 Q 和 K 的点积，然后进行缩放（乘以 1/sqrt(c)）以获得注意力权重。这确保了注意力的稳定性和可靠性。
    最后，利用注意力权重对值 V 进行加权求和，得到自注意力机制的输出。
    
'''
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        # 通过三个卷积层（self.q，self.k，self.v）分别计算查询（Q）、键（K）和值（V）的表示
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        
        # 计算注意力权重矩阵 w_。采用 Q 和 K 的点积
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        # 进行缩放（除以 sqrt(c)）
        w_ = w_ * (int(c)**(-0.5))

        # 通过 softmax 函数获得注意力权重
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        # 通过一个卷积层 (self.proj_out) 处理自注意力机制的输出。
        h_ = self.proj_out(h_)

        return x+h_

# 主要UNet模型实现
    #由下采样、中间层、上采样组合，其中包括残差块和自注意力块
class DiffusionUNet(nn.Module):
    # 接受一个配置对象 config，其中包含有关模型的各种参数，
    # 如通道数 (ch)、输出通道数 (out_ch)、残差块数量 (num_res_blocks) 等
    def __init__(self, config):
        super().__init__()
        self.config = config

        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # 初始化时间嵌入部分
        # timestep embedding
        # 通过 nn.Module 创建一个包含两个线性层的模块，用于生成时间步长的嵌入。
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # 初始化下采样部分
        # downsampling
        # 使用 nn.ModuleList 创建一个包含多个下采样模块的列表，
        # 每个模块由多个残差块 (ResnetBlock) 组成，其中可能包含注意力机制 (AttnBlock)。
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        '''
        对于每个下采样级别，都执行以下操作：
            初始化一个空的模块列表 block,用于存放下采样层的残差块。
            初始化一个空的模块列表 attn,用于存放下采样层的注意力机制。
            计算当前下采样层的输入通道数 block_in,为上一级别的 block_out * ch。
            计算当前下采样层的输出通道数 block_out,为当前级别的 ch * ch_mult[i_level]。
            对于每个下采样层的残差块，执行以下操作：
                添加一个 ResnetBlock 模块到 block,使用当前的 block_in 和 block_out,并指定注意力机制的通道数为 temb_channels。
                更新 block_in 为当前的 block_out。
                如果当前分辨率在配置的注意力分辨率列表 (attn_resolutions) 中，添加一个 AttnBlock 模块到 attn。
            初始化一个 nn.Module 对象 down,用于存放当前级别的下采样模块。
            将 block 添加为 down 的属性 block。
            将 attn 添加为 down 的属性 attn。
            如果不是最后一个下采样级别，添加一个 Downsample 模块到 down.downsample,该模块使用当前的 block_in 和 resamp_with_conv。
            更新 curr_res 为当前的一半，以便下一级别的下采样。
        '''
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # 初始化中间部分
        # middle
        # 由两个残差块和一个注意力机制组成，用于处理 U-Net 结构的中间部分。
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # 初始化上采样
        # upsampling
        # 使用 nn.ModuleList 创建一个包含多个上采样模块的列表，
        # 每个模块由多个残差块组成，其中可能包含注意力机制。
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # 最终输出层
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        # 使用 get_timestep_embedding 函数获取时间嵌入，将其应用于每个残差块。
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
