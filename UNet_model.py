""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np
import kornia.filters as KF

def normalize_per_sample(x, eps=0):
    mean = x.mean(dim=[2, 3], keepdim=True)  # 计算每个样本的均值
    std = x.std(dim=[2, 3], keepdim=True)    # 计算每个样本的标准差
    return (x - mean) / (std + eps)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.LeakyReLU(inplace=True),

#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


# class UNet_multimodal(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet_multimodal, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.base_channel = 8

#         self.inc = (DoubleConv(n_channels, self.base_channel))
#         self.down1 = (Down(self.base_channel, self.base_channel * 2))
#         self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
#         self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

#         self.inc_mri = (DoubleConv(n_channels, self.base_channel))
#         self.down1_mri = (Down(self.base_channel, self.base_channel * 2))
#         self.down2_mri = (Down(self.base_channel * 2, self.base_channel * 4))
#         self.down3_mri = (Down(self.base_channel * 4, self.base_channel * 8))

#         factor = 2 if bilinear else 1
#         self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
#         self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
#         self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
#         self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
#         self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
#         self.outc = (OutConv(self.base_channel, n_classes))

#         self.fusion1 = CGAFusion(self.base_channel, 2)
#         self.fusion2 = CGAFusion(self.base_channel * 2, 2)
#         self.fusion3 = CGAFusion(self.base_channel * 4, 4)
#         self.fusion4 = CGAFusion(self.base_channel * 8, 8)

#     def forward(self, x_mpi, x_mri):
#         x1_mpi = self.inc(x_mpi)
#         x2_mpi = self.down1(x1_mpi)
#         x3_mpi = self.down2(x2_mpi)
#         x4_mpi = self.down3(x3_mpi)

#         x1_mri = self.inc_mri(x_mri)
#         x2_mri = self.down1_mri(x1_mri)
#         x3_mri = self.down2_mri(x2_mri)
#         x4_mri = self.down3_mri(x3_mri)

#         x4 = self.fusion4(x4_mpi, x4_mri)
#         x5 = self.down4(x4)
#         x3 = self.fusion3(x3_mpi, x3_mri)
#         x2 = self.fusion2(x2_mpi, x2_mri)
#         x1 = self.fusion1(x1_mpi, x1_mri)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.down4 = torch.utils.checkpoint(self.down4)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.up4 = torch.utils.checkpoint(self.up4)
#         self.outc = torch.utils.checkpoint(self.outc)

class LocalFusion(nn.Module):
    def __init__(self):
        super(LocalFusion, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x_mri, x_mpi):
        x = x_mri + x_mpi
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = torch.sigmoid(self.sa(x2))
        out = sattn * x_mri + (1 - sattn) * x_mpi
        return out


class GlobalFusion(nn.Module):
    def __init__(self, in_channels_start, in_channels_mri, embed_dim, num_heads, patch_size=4):
        """
        实现 x_mri 和 x_start 的交叉注意力融合（支持指定 patch size）。
        :param in_channels_start: x_start 的通道数。
        :param in_channels_mri: x_mri 的通道数。
        :param embed_dim: 交叉注意力模块的嵌入维度。
        :param num_heads: 多头注意力的头数。
        :param patch_size: 将特征图分块的 patch 大小。
        """
        super(GlobalFusion, self).__init__()
        
        self.patch_size = patch_size
        

        # 分块后对每个 patch 投影到嵌入维度
        self.start_patch_projection = nn.Linear(in_channels_start * patch_size * patch_size, embed_dim)
        self.mri_patch_projection = nn.Linear(in_channels_mri * patch_size * patch_size, embed_dim)
        
        
        # 多头注意力机制
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 3),
            nn.ReLU(),
            nn.Linear(embed_dim * 3, embed_dim),
        ) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 输出融合特征
        self.output_projection = nn.Conv2d(embed_dim // (self.patch_size * self.patch_size), in_channels_start, kernel_size=1)

    def patchify(self, x):
        """
        将输入特征图分块（patchify）。
        :param x: 输入特征图，形状为 (B, C, H, W)。
        :return: 展平的 patch 特征，形状为 (B, num_patches, patch_dim)。
        """
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        # 确保特征图大小可以整除 patch size
        assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size."
        
        # 分块并展平
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # (B, C, num_patches_H, num_patches_W, patch_size, patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, num_patches_H, num_patches_W, C, patch_size, patch_size)
        x = x.view(B, -1, C * patch_size * patch_size)  # (B, num_patches, patch_dim)
        return x

    def unpatchify(self, x_token, H, W):
        """
        将分块后的特征图恢复为原始特征图。
        :param x_token: 展平的 patch 特征，形状为 (B, num_patches, patch_dim)。
        :param patch_size: 分块大小 (patch_size, patch_size)。
        :param H: 原始特征图的高度。
        :param W: 原始特征图的宽度。
        :return: 恢复后的特征图，形状为 (B, C, H, W)。
        """
        B, num_patches, patch_dim = x_token.shape
        num_patches_H = H // self.patch_size
        num_patches_W = W // self.patch_size
        C = patch_dim // (self.patch_size * self.patch_size)
        
        # 恢复为分块形式
        x_token = x_token.view(B, num_patches_H, num_patches_W, C, self.patch_size, self.patch_size)  # (B, num_patches_H, num_patches_W, C, patch_size, patch_size)
        x_token = x_token.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, num_patches_H, patch_size, num_patches_W, patch_size)
        
        # 合并分块
        x = x_token.view(B, C, H, W)  # (B, C, H, W)
        return x

    def forward(self, x_start, x_mri):
        """
        前向计算
        :param x_start: 主模态输入 (B, C_start, H, W)。
        :param x_mri: 条件模态输入 (B, C_mri, H, W)。
        :return: 融合后的特征 (B, C_start, H, W)。
        """
        B, _, H, W = x_start.shape
        
        # 分块
        x_start_patches = self.patchify(x_start)  # (B, num_patches, patch_dim)
        x_mri_patches = self.patchify(x_mri)  # (B, num_patches, patch_dim)
        
        # 投影到嵌入空间
        x_start_embed = self.start_patch_projection(x_start_patches)  # (B, num_patches, embed_dim)
        x_mri_embed = self.mri_patch_projection(x_mri_patches)  # (B, num_patches, embed_dim)
        
        # 交叉注意力：x_start 是 Query，x_mri 是 Key 和 Value
        fused_features, _ = self.cross_attention(query=x_start_embed, key=x_mri_embed, value=x_mri_embed)
        fused_features = self.norm1(fused_features + x_mri_embed)
        ffn_output = self.ffn(fused_features)
        fused_features = self.norm2(ffn_output + x_start_embed)
        
        # 恢复形状
        fused_features = self.unpatchify(fused_features, H, W)
        # num_patches = H // self.patch_size
        # fused_features = fused_features.view(B, num_patches, num_patches, -1)  # (B, num_patches_H, num_patches_W, embed_dim)
        # fused_features = fused_features.permute(0, 3, 1, 2).contiguous()  # (B, embed_dim, num_patches_H, num_patches_W)
        
        # 映射回原始通道数
        output = self.output_projection(fused_features)  # (B, C_start, H, W)
        return output


def beta_distribution_normalized(T, alpha=2, beta=5):
    """
    计算长度为 T 的 Beta 分布归一化权重数组，时间范围为 1-T。
    :param T: 总时间步数（数组长度）。
    :param alpha: Beta 分布的形状参数 alpha。
    :param beta: Beta 分布的形状参数 beta。
    :return: 长度为 T 的归一化权重数组。
    """
    # 时间步数组：1 到 T
    t = np.arange(1, T + 1)

    # 将时间步归一化到 [0, 1]
    t_normalized = t / T

    # Gamma 函数的实现 (用 NumPy 实现)
    def gamma_func(x):
        return np.exp(np.log(np.arange(1, x)).sum()) if x > 1 else 1.0

    # Beta 函数的计算
    def beta_func(a, b):
        return gamma_func(a) * gamma_func(b) / gamma_func(a + b)

    # Beta 分布值
    beta_value = beta_func(alpha, beta)
    f_t = (t_normalized ** (alpha - 1)) * ((1 - t_normalized) ** (beta - 1)) / beta_value

    # 计算最大值
    if alpha > 1 and beta > 1:
        t_max = (alpha - 1) / (alpha + beta - 2)  # 最大值的位置
        f_max = (t_max ** (alpha - 1)) * ((1 - t_max) ** (beta - 1)) / beta_value
    else:
        f_max = np.max(f_t)  # 如果 alpha 或 beta <= 1，直接用计算结果

    # 归一化
    return f_t / f_max

class DynamicFusion(nn.Module):
    def __init__(self, 
                 T,
                 in_channels_mpi,
                 in_channels_mri,
                 embed_dim,
                 num_heads=4,
                 patch_size=4 
                 ):
        super(DynamicFusion, self).__init__()
        self.global_fusion = GlobalFusion(in_channels_start=in_channels_mpi, 
                                          in_channels_mri=in_channels_mri, 
                                          embed_dim=embed_dim,
                                          num_heads=num_heads, patch_size=patch_size)
        self.local_fusion = LocalFusion()
        self.T = T
        alphas = beta_distribution_normalized(T, 2, 5)
        # self.alphas = torch.from_numpy(alphas).float()
        self.register_buffer("alphas", torch.from_numpy(alphas).float())

    def forward(self, x_mpi, x_mri, t):
        """
        :param x_start: 主模态输入。
        :param x_mri: 条件模态输入。
        :param t: 当前时间步。
        """
        # print(self.alphas.device)
        alpha_t = self.alphas[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 动态权重
        global_fused = self.global_fusion(x_mpi, x_mri)
        local_fused = self.local_fusion(x_mpi, x_mri)
        return alpha_t * local_fused + (1 - alpha_t) * global_fused

class UNet_multimodal(nn.Module):
    def __init__(self, n_channels, n_classes, T, bilinear=False):
        super(UNet_multimodal, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 8

        self.inc = (DoubleConv(n_channels, self.base_channel))
        self.down1 = (Down(self.base_channel, self.base_channel * 2))
        self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

        self.inc_mri = (DoubleConv(n_channels, self.base_channel))
        self.down1_mri = (Down(self.base_channel, self.base_channel * 2))
        self.down2_mri = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3_mri = (Down(self.base_channel * 4, self.base_channel * 8))

        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
        self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
        self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
        self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
        self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
        self.outc = (OutConv(self.base_channel, n_classes))

        # self.fusion1 = CGAFusion(self.base_channel, 2)
        # self.fusion2 = CGAFusion(self.base_channel * 2, 2)
        # self.fusion3 = CGAFusion(self.base_channel * 4, 4)
        # self.fusion4 = CGAFusion(self.base_channel * 8, 8)

        self.fusion1 = DynamicFusion(T, self.base_channel, self.base_channel, embed_dim=128, num_heads=4, patch_size=8) # (64, 64)
        self.fusion2 = DynamicFusion(T, self.base_channel * 2, self.base_channel * 2, embed_dim=128, num_heads=4, patch_size=4)  # (32, 32)
        self.fusion3 = DynamicFusion(T, self.base_channel * 4, self.base_channel * 4, embed_dim=128, num_heads=4, patch_size=2) # (16, 16)
        self.fusion4 = DynamicFusion(T, self.base_channel * 8, self.base_channel * 8, embed_dim=128, num_heads=4, patch_size=1) # (8, 8)

    def forward(self, x_mpi, x_mri, t):
        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        x1_mri = self.inc_mri(x_mri)
        x2_mri = self.down1_mri(x1_mri)
        x3_mri = self.down2_mri(x2_mri)
        x4_mri = self.down3_mri(x3_mri)

        x4 = self.fusion4(x4_mpi, x4_mri, t)
        x5 = self.down4(x4)
        x3 = self.fusion3(x3_mpi, x3_mri, t)
        x2 = self.fusion2(x2_mpi, x2_mri, t)
        x1 = self.fusion1(x1_mpi, x1_mri, t)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def get_feature_maps(self, x_mpi, x_mri, t):
        MPI_feats = []
        MRI_feats = []
        fusion_feats = []

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)
        MPI_feats.extend([x1_mpi, x2_mpi, x3_mpi, x4_mpi])

        x1_mri = self.inc_mri(x_mri)
        x2_mri = self.down1_mri(x1_mri)
        x3_mri = self.down2_mri(x2_mri)
        x4_mri = self.down3_mri(x3_mri)
        MRI_feats.extend([x1_mri, x2_mri, x3_mri, x4_mri])

        x4 = self.fusion4(x4_mpi, x4_mri, t)
        x5 = self.down4(x4)
        x3 = self.fusion3(x3_mpi, x3_mri, t)
        x2 = self.fusion2(x2_mpi, x2_mri, t)
        x1 = self.fusion1(x1_mpi, x1_mri, t)

        fusion_feats.extend([x1, x2, x3, x4, x5])

        x = self.up1(x5, x4)
        fusion_feats.append(x)
        x = self.up2(x, x3)
        fusion_feats.append(x)
        x = self.up3(x, x2)
        fusion_feats.append(x)
        x = self.up4(x, x1)
        fusion_feats.append(x)
        logits = self.outc(x)
        return logits, MPI_feats, MRI_feats, fusion_feats
    

# class UNet_multimodal_distill(nn.Module):
#     def __init__(self, n_channels, n_classes, T, sig_dim=9000, bilinear=False):
#         super(UNet_multimodal_distill, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.base_channel = 8
#         self.sig_dim = sig_dim

#         self.sig2img = nn.Sequential(
#                 nn.Linear(sig_dim, 4096),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Linear(4096, 4096)
#             )
        
#         self.reco = nn.Sequential(
#                 DoubleConv(2, 16),
#                 DoubleConv(16, 32),
#                 DoubleConv(32, 1),
#             )


#         self.inc = (DoubleConv(n_channels, self.base_channel))
#         self.down1 = (Down(self.base_channel, self.base_channel * 2))
#         self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
#         self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

#         self.inc_mri = DoubleConv(self.base_channel, self.base_channel)
#         self.down1_mri = DoubleConv(self.base_channel * 2, self.base_channel * 2)
#         self.down2_mri = DoubleConv(self.base_channel * 4, self.base_channel * 4)
#         self.down3_mri = DoubleConv(self.base_channel * 8, self.base_channel * 8)

#         factor = 2 if bilinear else 1
#         self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
#         self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
#         self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
#         self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
#         self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
#         self.outc = (OutConv(self.base_channel, n_classes))

#         self.fusion1 = DoubleConv(self.base_channel * 2, self.base_channel)
#         self.fusion2 = DoubleConv(self.base_channel * 4, self.base_channel * 2)
#         self.fusion3 = DoubleConv(self.base_channel * 8, self.base_channel * 4)
#         self.fusion4 = DoubleConv(self.base_channel * 16, self.base_channel * 8)

#         self.track_prediction_head_x1_mpi = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
#         self.track_prediction_head_x4_out = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])

#     def forward(self, x_mpi, sig=None):
#         if sig is not None:
#             reco = self.sig2img(sig)
#             reco = reco.reshape(reco.shape[0], 1, 64, 64)
#             x_mpi = torch.cat([x_mpi, reco], 1)
#             x_mpi = self.reco(x_mpi)

#         x1_mpi = self.inc(x_mpi)
#         x2_mpi = self.down1(x1_mpi)
#         x3_mpi = self.down2(x2_mpi)
#         x4_mpi = self.down3(x3_mpi)

#         x1_mri = self.inc_mri(x1_mpi)
#         x2_mri = self.down1_mri(x2_mpi)
#         x3_mri = self.down2_mri(x3_mpi)
#         x4_mri = self.down3_mri(x4_mpi)

#         x4 = self.fusion4(torch.cat([x4_mri, x4_mpi], 1))
#         x5 = self.down4(x4)
#         x3 = self.fusion3(torch.cat([x3_mri, x3_mpi], 1))
#         x2 = self.fusion2(torch.cat([x2_mri, x2_mpi], 1))
#         x1 = self.fusion1(torch.cat([x1_mri, x1_mpi], 1))

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
    
#     def track_prediction(self, feat, track_operation_list):
#         track_feat = [conv(feat).unsqueeze(1) for conv in track_operation_list]
#         track_feat = torch.cat(track_feat, 1)
#         return track_feat

#     def forward_feats(self, x_mpi, sig=None):
#         feats = {}

#         if sig is not None:
#             reco = self.sig2img(sig)
#             reco = reco.reshape(reco.shape[0], 1, 64, 64)
#             x_mpi = torch.cat([x_mpi, reco], 1)
#             x_mpi = self.reco(x_mpi)

#         x1_mpi = self.inc(x_mpi)
#         x2_mpi = self.down1(x1_mpi)
#         x3_mpi = self.down2(x2_mpi)
#         x4_mpi = self.down3(x3_mpi)

#         feats['x1_mpi_feats'] = x1_mpi
#         feats['x2_mpi_feats'] = x2_mpi
#         feats['x3_mpi_feats'] = x3_mpi
#         # feats['x4_mpi_feats'] = x4_mpi

#         feats['x1_mpi_track_feats'] = self.track_prediction(x1_mpi, self.track_prediction_head_x1_mpi)

#         x1_mri = self.inc_mri(x1_mpi)
#         x2_mri = self.down1_mri(x2_mpi)
#         x3_mri = self.down2_mri(x3_mpi)
#         x4_mri = self.down3_mri(x4_mpi)

#         x4_fusion = self.fusion4(torch.cat([x4_mri, x4_mpi], 1))
#         x5 = self.down4(x4_fusion)
#         x3_fusion = self.fusion3(torch.cat([x3_mri, x3_mpi], 1))
#         x2_fusion = self.fusion2(torch.cat([x2_mri, x2_mpi], 1))
#         x1_fusion = self.fusion1(torch.cat([x1_mri, x1_mpi], 1))

#         feats['x1_fusion_feats'] = x1_fusion
#         feats['x2_fusion_feats'] = x2_fusion
#         feats['x3_fusion_feats'] = x3_fusion
#         # feats['x4_fusion_feats'] = x4_fusion
#         # feats['x5_fusion_feats'] = x5

#         x = self.up1(x5, x4_fusion)
#         # feats['x1_out_feats'] = x

#         x = self.up2(x, x3_fusion)
#         feats['x2_out_feats'] = x

#         x = self.up3(x, x2_fusion)
#         feats['x3_out_feats'] = x

#         x = self.up4(x, x1_fusion)
#         feats['x4_out_feats'] = x

#         feats['x4_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x4_out)

#         logits = self.outc(x)

#         return logits, feats   


class GateHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)  # output: [B, 1, H, W]


class UNet_multimodal_distill(nn.Module):
    def __init__(self, n_channels, n_classes, T, sig_dim=9000, bilinear=False):
        super(UNet_multimodal_distill, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 8
        self.sig_dim = sig_dim

        self.sig2img = nn.Sequential(
                nn.Linear(sig_dim, 4096),
                nn.LeakyReLU(inplace=True),
                nn.Linear(4096, 4096)
            )
        
        self.reco = nn.Sequential(
                DoubleConv(2, 16),
                DoubleConv(16, 32),
                DoubleConv(32, 1),
            )

        self.inc = (DoubleConv(n_channels, self.base_channel))
        self.down1 = (Down(self.base_channel, self.base_channel * 2))
        self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

        self.delta_head1 = DoubleConv(self.base_channel, self.base_channel)
        self.delta_head2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        self.delta_head3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        self.delta_head4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        # self.fusion1 = DoubleConv(self.base_channel, self.base_channel)
        # self.fusion2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        # self.fusion3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        # self.fusion4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        self.gate = GateHead(self.base_channel)

        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
        self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
        self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
        self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
        self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
        self.outc = (OutConv(self.base_channel, n_classes))
        # self.seg_out = (OutConv(self.base_channel, 2))

        self.track_prediction_head_x1_mpi = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

        self.track_prediction_head_x1_fusion = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

        self.track_prediction_head_x4_out = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_out = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_out = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

    def forward(self, x_mpi, sig=None):
        None
    
    def track_prediction(self, feat, track_operation_list):
        track_feat = [conv(feat).unsqueeze(1) for conv in track_operation_list]
        track_feat = torch.cat(track_feat, 1)
        return track_feat

    def forward_feats(self, x_mpi, sig=None):
        feats = {}

        if sig is not None:
            reco = self.sig2img(sig)
            reco = reco.reshape(reco.shape[0], 1, 64, 64)
            x_mpi = torch.cat([x_mpi, reco], 1)
            x_mpi = self.reco(x_mpi)

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        M1 = self.gate(x1_mpi)
        M2 = F.interpolate(M1, scale_factor=0.5)
        M3 = F.interpolate(M1, scale_factor=0.25)

        feats['x1_mpi_feats'] = x1_mpi
        feats['x2_mpi_feats'] = x2_mpi
        feats['x3_mpi_feats'] = x3_mpi
        # feats['x4_mpi_feats'] = x4_mpi

        feats['x1_mpi_track_feats'] = self.track_prediction(x1_mpi, self.track_prediction_head_x1_mpi)
        feats['x2_mpi_track_feats'] = self.track_prediction(x2_mpi, self.track_prediction_head_x2_mpi)
        feats['x3_mpi_track_feats'] = self.track_prediction(x3_mpi, self.track_prediction_head_x3_mpi)

        delta1_mri = self.delta_head1(x1_mpi)
        delta2_mri = self.delta_head2(x2_mpi)
        delta3_mri = self.delta_head3(x3_mpi)
        delta4_mri = self.delta_head4(x4_mpi)

        feats['delta1_mri'] = delta1_mri
        feats['delta2_mri'] = delta2_mri
        feats['delta3_mri'] = delta3_mri

        feats['M1'] = M1
        feats['M2'] = M2
        feats['M3'] = M3

        x4_fusion = x4_mpi + delta4_mri
        x5 = self.down4(x4_fusion)
        x3_fusion = x3_mpi + delta3_mri * M3
        x2_fusion = x2_mpi + delta2_mri * M2
        x1_fusion = x1_mpi + delta1_mri * M1

        # x4_fusion = self.fusion4(x4_mpi + delta4_mri)
        # x5 = self.down4(x4_fusion)
        # x3_fusion = self.fusion3(x3_mpi + delta3_mri * M3)
        # x2_fusion = self.fusion2(x2_mpi + delta2_mri * M2)
        # x1_fusion = self.fusion1(x1_mpi + delta1_mri * M1)

        # x4_fusion = self.fusion4(torch.cat([x4_mri, x4_mpi], 1))
        # x5 = self.down4(x4_fusion)
        # x3_fusion = self.fusion3(torch.cat([x3_mri, x3_mpi], 1))
        # x2_fusion = self.fusion2(torch.cat([x2_mri, x2_mpi], 1))
        # x1_fusion = self.fusion1(torch.cat([x1_mri, x1_mpi], 1))

        feats['x1_fusion_feats'] = x1_fusion
        feats['x2_fusion_feats'] = x2_fusion
        feats['x3_fusion_feats'] = x3_fusion

        feats['x1_fusion_track_feats'] = self.track_prediction(x1_fusion, self.track_prediction_head_x1_fusion)
        feats['x2_fusion_track_feats'] = self.track_prediction(x2_fusion, self.track_prediction_head_x2_fusion)
        feats['x3_fusion_track_feats'] = self.track_prediction(x3_fusion, self.track_prediction_head_x3_fusion)

        # feats['x4_fusion_feats'] = x4_fusion
        # feats['x5_fusion_feats'] = x5

        x = self.up1(x5, x4_fusion)
        # feats['x1_out_feats'] = x

        x = self.up2(x, x3_fusion)
        feats['x2_out_feats'] = x
        feats['x2_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x2_out)

        x = self.up3(x, x2_fusion)
        feats['x3_out_feats'] = x
        feats['x3_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x3_out)

        x = self.up4(x, x1_fusion)
        feats['x4_out_feats'] = x
        feats['x4_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x4_out)
        
        

        logits = self.outc(x)
        # seg_logits = self.seg_out(x)

        return logits, feats

class UNet_multimodal_distill(nn.Module):
    def __init__(self, n_channels, n_classes, T, sig_dim=9000, bilinear=False):
        super(UNet_multimodal_distill, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 8
        self.sig_dim = sig_dim

        self.sig2img = nn.Sequential(
                nn.Linear(sig_dim, 4096),
                nn.LeakyReLU(inplace=True),
                nn.Linear(4096, 4096)
            )
        
        self.reco = nn.Sequential(
                DoubleConv(2, 16),
                DoubleConv(16, 32),
                DoubleConv(32, 1),
            )

        self.inc = (DoubleConv(n_channels, self.base_channel))
        self.down1 = (Down(self.base_channel, self.base_channel * 2))
        self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

        self.delta_head1 = DoubleConv(self.base_channel, self.base_channel)
        self.delta_head2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        self.delta_head3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        self.delta_head4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        # self.fusion1 = DoubleConv(self.base_channel, self.base_channel)
        # self.fusion2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        # self.fusion3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        # self.fusion4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        self.gate = GateHead(self.base_channel)

        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
        self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
        self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
        self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
        self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
        self.outc = (OutConv(self.base_channel, n_classes))
        # self.seg_out = (OutConv(self.base_channel, 2))

        self.track_prediction_head_x1_mpi = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

        self.track_prediction_head_x1_fusion = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

        self.track_prediction_head_x4_out = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_out = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_out = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

    def forward(self, x_mpi, sig=None):
        if sig is not None:
            reco = self.sig2img(sig)
            reco = reco.reshape(reco.shape[0], 1, 64, 64)
            x_mpi = torch.cat([x_mpi, reco], 1)
            x_mpi = self.reco(x_mpi)

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        M1 = self.gate(x1_mpi)
        M2 = F.interpolate(M1, scale_factor=0.5)
        M3 = F.interpolate(M1, scale_factor=0.25)

        delta1_mri = self.delta_head1(x1_mpi)
        delta2_mri = self.delta_head2(x2_mpi)
        delta3_mri = self.delta_head3(x3_mpi)
        delta4_mri = self.delta_head4(x4_mpi)



        x4_fusion = x4_mpi + delta4_mri
        x5 = self.down4(x4_fusion)
        x3_fusion = x3_mpi + delta3_mri * M3
        x2_fusion = x2_mpi + delta2_mri * M2
        x1_fusion = x1_mpi + delta1_mri * M1


        x = self.up1(x5, x4_fusion)

        x = self.up2(x, x3_fusion)


        x = self.up3(x, x2_fusion)


        x = self.up4(x, x1_fusion)

        
    
        logits = self.outc(x)

        return logits, None
    
    def track_prediction(self, feat, track_operation_list):
        track_feat = [conv(feat).unsqueeze(1) for conv in track_operation_list]
        track_feat = torch.cat(track_feat, 1)
        return track_feat

    def forward_feats(self, x_mpi, sig=None):
        feats = {}

        if sig is not None:
            reco = self.sig2img(sig)
            reco = reco.reshape(reco.shape[0], 1, 64, 64)
            x_mpi = torch.cat([x_mpi, reco], 1)
            x_mpi = self.reco(x_mpi)

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        M1 = self.gate(x1_mpi)
        M2 = F.interpolate(M1, scale_factor=0.5)
        M3 = F.interpolate(M1, scale_factor=0.25)

        feats['x1_mpi_feats'] = x1_mpi
        feats['x2_mpi_feats'] = x2_mpi
        feats['x3_mpi_feats'] = x3_mpi
        # feats['x4_mpi_feats'] = x4_mpi

        feats['x1_mpi_track_feats'] = self.track_prediction(x1_mpi, self.track_prediction_head_x1_mpi)
        feats['x2_mpi_track_feats'] = self.track_prediction(x2_mpi, self.track_prediction_head_x2_mpi)
        feats['x3_mpi_track_feats'] = self.track_prediction(x3_mpi, self.track_prediction_head_x3_mpi)

        delta1_mri = self.delta_head1(x1_mpi)
        delta2_mri = self.delta_head2(x2_mpi)
        delta3_mri = self.delta_head3(x3_mpi)
        delta4_mri = self.delta_head4(x4_mpi)

        feats['delta1_mri'] = delta1_mri
        feats['delta2_mri'] = delta2_mri
        feats['delta3_mri'] = delta3_mri

        feats['M1'] = M1
        feats['M2'] = M2
        feats['M3'] = M3

        x4_fusion = x4_mpi + delta4_mri
        x5 = self.down4(x4_fusion)
        x3_fusion = x3_mpi + delta3_mri * M3
        x2_fusion = x2_mpi + delta2_mri * M2
        x1_fusion = x1_mpi + delta1_mri * M1

        # x4_fusion = self.fusion4(x4_mpi + delta4_mri)
        # x5 = self.down4(x4_fusion)
        # x3_fusion = self.fusion3(x3_mpi + delta3_mri * M3)
        # x2_fusion = self.fusion2(x2_mpi + delta2_mri * M2)
        # x1_fusion = self.fusion1(x1_mpi + delta1_mri * M1)

        # x4_fusion = self.fusion4(torch.cat([x4_mri, x4_mpi], 1))
        # x5 = self.down4(x4_fusion)
        # x3_fusion = self.fusion3(torch.cat([x3_mri, x3_mpi], 1))
        # x2_fusion = self.fusion2(torch.cat([x2_mri, x2_mpi], 1))
        # x1_fusion = self.fusion1(torch.cat([x1_mri, x1_mpi], 1))

        feats['x1_fusion_feats'] = x1_fusion
        feats['x2_fusion_feats'] = x2_fusion
        feats['x3_fusion_feats'] = x3_fusion

        feats['x1_fusion_track_feats'] = self.track_prediction(x1_fusion, self.track_prediction_head_x1_fusion)
        feats['x2_fusion_track_feats'] = self.track_prediction(x2_fusion, self.track_prediction_head_x2_fusion)
        feats['x3_fusion_track_feats'] = self.track_prediction(x3_fusion, self.track_prediction_head_x3_fusion)

        # feats['x4_fusion_feats'] = x4_fusion
        # feats['x5_fusion_feats'] = x5

        x = self.up1(x5, x4_fusion)
        # feats['x1_out_feats'] = x

        x = self.up2(x, x3_fusion)
        feats['x2_out_feats'] = x
        feats['x2_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x2_out)

        x = self.up3(x, x2_fusion)
        feats['x3_out_feats'] = x
        feats['x3_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x3_out)

        x = self.up4(x, x1_fusion)
        feats['x4_out_feats'] = x
        feats['x4_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x4_out)
        
        

        logits = self.outc(x)
        # seg_logits = self.seg_out(x)

        return logits, feats
    


class UNet_multimodal_distill03(nn.Module):
    def __init__(self, n_channels, n_classes, T, sig_dim=9000, bilinear=False):
        super(UNet_multimodal_distill03, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 8
        self.sig_dim = sig_dim

        self.sig2img = nn.Sequential(
                nn.Linear(sig_dim, 4096),
                nn.LeakyReLU(inplace=True),
                nn.Linear(4096, 4096)
            )
        
        self.reco = nn.Sequential(
                DoubleConv(2, 16),
                DoubleConv(16, 32),
                DoubleConv(32, 1),
            )

        self.inc = (DoubleConv(n_channels, self.base_channel))
        self.down1 = (Down(self.base_channel, self.base_channel * 2))
        self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

        self.delta_head1 = DoubleConv(self.base_channel, self.base_channel)
        self.delta_head2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        self.delta_head3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        self.delta_head4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        # self.fusion1 = DoubleConv(self.base_channel, self.base_channel)
        # self.fusion2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        # self.fusion3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        # self.fusion4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        self.gate = GateHead(self.base_channel)

        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
        self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
        self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
        self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
        self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
        self.outc = (OutConv(self.base_channel, n_classes))
        # self.seg_out = (OutConv(self.base_channel, 2))

        self.track_prediction_head_x1_mpi = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

        self.track_prediction_head_x1_fusion = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

        self.track_prediction_head_x4_out = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_out = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_out = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

    def forward(self, x_mpi, sig=None):
        None
    
    def track_prediction(self, feat, track_operation_list):
        track_feat = [conv(feat).unsqueeze(1) for conv in track_operation_list]
        track_feat = torch.cat(track_feat, 1)
        return track_feat

    def forward_feats(self, x_mpi, sig=None):
        feats = {}

        if sig is not None:
            reco = self.sig2img(sig)
            reco = reco.reshape(reco.shape[0], 1, 64, 64)
            x_mpi = torch.cat([x_mpi, reco], 1)
            x_mpi = self.reco(x_mpi)

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        M1 = self.gate(x1_mpi)
        M2 = F.interpolate(M1, scale_factor=0.5)
        M3 = F.interpolate(M1, scale_factor=0.25)

        feats['x1_mpi_feats'] = x1_mpi
        feats['x2_mpi_feats'] = x2_mpi
        feats['x3_mpi_feats'] = x3_mpi
        # feats['x4_mpi_feats'] = x4_mpi

        feats['x1_mpi_track_feats'] = self.track_prediction(x1_mpi, self.track_prediction_head_x1_mpi)
        feats['x2_mpi_track_feats'] = self.track_prediction(x2_mpi, self.track_prediction_head_x2_mpi)
        feats['x3_mpi_track_feats'] = self.track_prediction(x3_mpi, self.track_prediction_head_x3_mpi)

        delta1_mri = self.delta_head1(x1_mpi)
        delta2_mri = self.delta_head2(x2_mpi)
        delta3_mri = self.delta_head3(x3_mpi)
        delta4_mri = self.delta_head4(x4_mpi)

        # feats['delta1_mri'] = delta1_mri
        # feats['delta2_mri'] = delta2_mri
        # feats['delta3_mri'] = delta3_mri

        # feats['M1'] = M1
        # feats['M2'] = M2
        # feats['M3'] = M3

        x4_fusion = x4_mpi + delta4_mri
        x5 = self.down4(x4_fusion)
        x3_fusion = x3_mpi + delta3_mri * M3
        x2_fusion = x2_mpi + delta2_mri * M2
        x1_fusion = x1_mpi + delta1_mri * M1

        feats['x1_fusion_feats'] = x1_fusion
        feats['x2_fusion_feats'] = x2_fusion
        feats['x3_fusion_feats'] = x3_fusion

        feats['x1_fusion_track_feats'] = self.track_prediction(x1_fusion, self.track_prediction_head_x1_fusion)
        feats['x2_fusion_track_feats'] = self.track_prediction(x2_fusion, self.track_prediction_head_x2_fusion)
        feats['x3_fusion_track_feats'] = self.track_prediction(x3_fusion, self.track_prediction_head_x3_fusion)

        # feats['x4_fusion_feats'] = x4_fusion
        # feats['x5_fusion_feats'] = x5

        x = self.up1(x5, x4_fusion)
        # feats['x1_out_feats'] = x

        x = self.up2(x, x3_fusion)
        feats['x2_out_feats'] = x
        feats['x2_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x2_out)

        x = self.up3(x, x2_fusion)
        feats['x3_out_feats'] = x
        feats['x3_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x3_out)

        x = self.up4(x, x1_fusion)
        feats['x4_out_feats'] = x
        feats['x4_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x4_out)
        
        

        logits = self.outc(x)
        # seg_logits = self.seg_out(x)

        return logits, feats
    

class UNet_multimodal_distill02(nn.Module):
    def __init__(self, n_channels, n_classes, T, sig_dim=9000, bilinear=False):
        super(UNet_multimodal_distill02, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 8
        self.sig_dim = sig_dim

        self.sig2img = nn.Sequential(
                nn.Linear(sig_dim, 4096),
                nn.LeakyReLU(inplace=True),
                nn.Linear(4096, 4096)
            )
        
        self.reco = nn.Sequential(
                DoubleConv(2, 16),
                DoubleConv(16, 32),
                DoubleConv(32, 1),
            )

        self.inc = (DoubleConv(n_channels, self.base_channel))
        self.down1 = (Down(self.base_channel, self.base_channel * 2))
        self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

        self.delta_head1 = DoubleConv(self.base_channel, self.base_channel)
        self.delta_head2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        self.delta_head3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        self.delta_head4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        self.gate = GateHead(self.base_channel)

        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
        self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
        self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
        self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
        self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
        self.outc = (OutConv(self.base_channel, n_classes))

    def forward(self, x_mpi, sig=None):
        None
    
    def track_prediction(self, feat, track_operation_list):
        track_feat = [conv(feat).unsqueeze(1) for conv in track_operation_list]
        track_feat = torch.cat(track_feat, 1)
        return track_feat

    def forward_feats(self, x_mpi, sig=None):
        feats = {}

        if sig is not None:
            reco = self.sig2img(sig)
            reco = reco.reshape(reco.shape[0], 1, 64, 64)
            x_mpi = torch.cat([x_mpi, reco], 1)
            x_mpi = self.reco(x_mpi)

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        M1 = self.gate(x1_mpi)
        M2 = F.interpolate(M1, scale_factor=0.5)
        M3 = F.interpolate(M1, scale_factor=0.25)

        feats['x1_mpi_feats'] = x1_mpi
        feats['x2_mpi_feats'] = x2_mpi
        feats['x3_mpi_feats'] = x3_mpi
        # feats['x4_mpi_feats'] = x4_mpi

        delta1_mri = self.delta_head1(x1_mpi)
        delta2_mri = self.delta_head2(x2_mpi)
        delta3_mri = self.delta_head3(x3_mpi)
        delta4_mri = self.delta_head4(x4_mpi)

        feats['delta1_mri'] = delta1_mri
        feats['delta2_mri'] = delta2_mri
        feats['delta3_mri'] = delta3_mri

        feats['M1'] = M1
        feats['M2'] = M2
        feats['M3'] = M3

        x4_fusion = x4_mpi + delta4_mri
        x5 = self.down4(x4_fusion)
        x3_fusion = x3_mpi + delta3_mri * M3
        x2_fusion = x2_mpi + delta2_mri * M2
        x1_fusion = x1_mpi + delta1_mri * M1


        feats['x1_fusion_feats'] = x1_fusion
        feats['x2_fusion_feats'] = x2_fusion
        feats['x3_fusion_feats'] = x3_fusion

        x = self.up1(x5, x4_fusion)
        # feats['x1_out_feats'] = x

        x = self.up2(x, x3_fusion)
        feats['x2_out_feats'] = x

        x = self.up3(x, x2_fusion)
        feats['x3_out_feats'] = x

        x = self.up4(x, x1_fusion)
        feats['x4_out_feats'] = x
        
        

        logits = self.outc(x)
        # seg_logits = self.seg_out(x)

        return logits, feats
    

class UNet_multimodal_distill04(nn.Module):
    def __init__(self, n_channels, n_classes, T, sig_dim=9000, bilinear=False):
        super(UNet_multimodal_distill04, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 8
        self.sig_dim = sig_dim

        self.sig2img = nn.Sequential(
                nn.Linear(sig_dim, 4096),
                nn.LeakyReLU(inplace=True),
                nn.Linear(4096, 4096)
            )
        
        self.reco = nn.Sequential(
                DoubleConv(2, 16),
                DoubleConv(16, 32),
                DoubleConv(32, 1),
            )

        self.inc = (DoubleConv(n_channels, self.base_channel))
        self.down1 = (Down(self.base_channel, self.base_channel * 2))
        self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
        self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
        self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
        self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
        self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
        self.outc = (OutConv(self.base_channel, n_classes))

    def forward(self, x_mpi, sig=None):
        None
    
    def track_prediction(self, feat, track_operation_list):
        track_feat = [conv(feat).unsqueeze(1) for conv in track_operation_list]
        track_feat = torch.cat(track_feat, 1)
        return track_feat

    def forward_feats(self, x_mpi, sig=None):
        feats = {}

        if sig is not None:
            reco = self.sig2img(sig)
            reco = reco.reshape(reco.shape[0], 1, 64, 64)
            x_mpi = torch.cat([x_mpi, reco], 1)
            x_mpi = self.reco(x_mpi)

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        x5 = self.down4(x4_mpi)



        x = self.up1(x5, x4_mpi)
        x = self.up2(x, x3_mpi)
        x = self.up3(x, x2_mpi)
        x = self.up4(x, x1_mpi)
        logits = self.outc(x)

        return logits, feats
    
class UNet_multimodal_distill00(nn.Module):
    def __init__(self, n_channels, n_classes, T, bilinear=False):
        super(UNet_multimodal_distill00, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 8

        self.inc = (DoubleConv(n_channels, self.base_channel))
        self.down1 = (Down(self.base_channel, self.base_channel * 2))
        self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

        self.inc_mri = (DoubleConv(n_channels, self.base_channel))
        self.down1_mri = (Down(self.base_channel, self.base_channel * 2))
        self.down2_mri = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3_mri = (Down(self.base_channel * 4, self.base_channel * 8))

        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
        self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
        self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
        self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
        self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
        self.outc = (OutConv(self.base_channel, n_classes))

        self.fusion1 = DynamicFusion(T, self.base_channel, self.base_channel, embed_dim=128, num_heads=4, patch_size=8) # (64, 64)
        self.fusion2 = DynamicFusion(T, self.base_channel * 2, self.base_channel * 2, embed_dim=128, num_heads=4, patch_size=4)  # (32, 32)
        self.fusion3 = DynamicFusion(T, self.base_channel * 4, self.base_channel * 4, embed_dim=128, num_heads=4, patch_size=2) # (16, 16)
        self.fusion4 = DynamicFusion(T, self.base_channel * 8, self.base_channel * 8, embed_dim=128, num_heads=4, patch_size=1) # (8, 8)

        self.track_prediction_head_x1_mpi = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

        self.track_prediction_head_x1_fusion = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

        self.track_prediction_head_x4_out = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x3_out = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(7)])
        self.track_prediction_head_x2_out = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(7)])

    def forward(self):
        None


    def track_prediction(self, feat, track_operation_list):
        track_feat = [conv(feat).unsqueeze(1) for conv in track_operation_list]
        track_feat = torch.cat(track_feat, 1)
        return track_feat
    
    def forward_feats(self, x_mpi, x_mri, t):
        feats = {}

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        feats['x1_mpi_feats'] = x1_mpi
        feats['x2_mpi_feats'] = x2_mpi
        feats['x3_mpi_feats'] = x3_mpi
        # feats['x4_mpi_feats'] = x4_mpi

        feats['x1_mpi_track_feats'] = self.track_prediction(x1_mpi, self.track_prediction_head_x1_mpi)
        feats['x2_mpi_track_feats'] = self.track_prediction(x2_mpi, self.track_prediction_head_x2_mpi)
        feats['x3_mpi_track_feats'] = self.track_prediction(x3_mpi, self.track_prediction_head_x3_mpi)

        x1_mri = self.inc_mri(x_mri)
        x2_mri = self.down1_mri(x1_mri)
        x3_mri = self.down2_mri(x2_mri)
        x4_mri = self.down3_mri(x3_mri)

        x4 = self.fusion4(x4_mpi, x4_mri, t)
        x5 = self.down4(x4)
        x3 = self.fusion3(x3_mpi, x3_mri, t)
        x2 = self.fusion2(x2_mpi, x2_mri, t)
        x1 = self.fusion1(x1_mpi, x1_mri, t)

        feats['x1_fusion_feats'] = x1
        feats['x2_fusion_feats'] = x2
        feats['x3_fusion_feats'] = x3

        feats['x1_fusion_track_feats'] = self.track_prediction(x1, self.track_prediction_head_x1_fusion)
        feats['x2_fusion_track_feats'] = self.track_prediction(x2, self.track_prediction_head_x2_fusion)
        feats['x3_fusion_track_feats'] = self.track_prediction(x3, self.track_prediction_head_x3_fusion)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        feats['x2_out_feats'] = x
        feats['x2_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x2_out)
        x = self.up3(x, x2)
        feats['x3_out_feats'] = x
        feats['x3_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x3_out)
        x = self.up4(x, x1)
        feats['x4_out_feats'] = x
        feats['x4_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x4_out) 

        logits = self.outc(x)
        return logits, feats
    

class UNet_multimodal_distill_OpenMPI(nn.Module):
    def __init__(self, n_channels, n_classes, T, bilinear=False):
        super(UNet_multimodal_distill_OpenMPI, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channel = 8

        self.inc = (DoubleConv(n_channels, self.base_channel))
        self.down1 = (Down(self.base_channel, self.base_channel * 2))
        self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
        self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

        self.delta_head1 = DoubleConv(self.base_channel, self.base_channel)
        self.delta_head2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        self.delta_head3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        self.delta_head4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        # self.fusion1 = DoubleConv(self.base_channel, self.base_channel)
        # self.fusion2 = DoubleConv(self.base_channel * 2, self.base_channel * 2)
        # self.fusion3 = DoubleConv(self.base_channel * 4, self.base_channel * 4)
        # self.fusion4 = DoubleConv(self.base_channel * 8, self.base_channel * 8)

        self.gate = GateHead(self.base_channel)

        factor = 2 if bilinear else 1
        self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
        self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
        self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
        self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
        self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
        self.outc = (OutConv(self.base_channel, n_classes))

        # self.track_prediction_head_x1_mpi = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(10)])
        # self.track_prediction_head_x2_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(10)])
        # self.track_prediction_head_x3_mpi = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(10)])

        # self.track_prediction_head_x1_fusion = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(10)])
        # self.track_prediction_head_x2_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(10)])
        # self.track_prediction_head_x3_fusion = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(10)])

        # self.track_prediction_head_x4_out = nn.ModuleList([nn.Conv2d(self.base_channel, self.base_channel, kernel_size=1, bias=False) for _ in range(10)])
        # self.track_prediction_head_x3_out = nn.ModuleList([nn.Conv2d(self.base_channel * 2, self.base_channel * 2, kernel_size=1, bias=False) for _ in range(10)])
        # self.track_prediction_head_x2_out = nn.ModuleList([nn.Conv2d(self.base_channel * 4, self.base_channel * 4, kernel_size=1, bias=False) for _ in range(10)])

        # self.residual_conv = nn.Sequential(DoubleConv(1, 16), OutConv(16, 1)) # 试试 indentity
        self.residual_conv = nn.Sequential(DoubleConv(1, 1), OutConv(1, 1))
        # self.residual_conv = nn.Identity()
    def forward(self, x_mpi):
        None
    
    def track_prediction(self, feat, track_operation_list):
        track_feat = [conv(feat).unsqueeze(1) for conv in track_operation_list]
        track_feat = torch.cat(track_feat, 1)
        return track_feat
    
    def forward_feats(self, x_mpi):
        x_mpi, resi = x_mpi

        residual = resi


        residual[residual < -0.65] = 0

        # residual[residual < -0.0] = 0

        # residual = KF.gaussian_blur2d(residual, kernel_size=(3, 3), sigma=(1.25, 1.25))
        residual = KF.gaussian_blur2d(residual, kernel_size=(3, 3), sigma=(1.25, 1.25))


        residual = self.residual_conv(residual)

        feats = {}

        x1_mpi = self.inc(x_mpi)
        x2_mpi = self.down1(x1_mpi)
        x3_mpi = self.down2(x2_mpi)
        x4_mpi = self.down3(x3_mpi)

        M1 = self.gate(x1_mpi)
        M2 = F.interpolate(M1, scale_factor=0.5)
        M3 = F.interpolate(M1, scale_factor=0.25)

        # feats['x1_mpi_feats'] = x1_mpi
        # feats['x2_mpi_feats'] = x2_mpi
        # feats['x3_mpi_feats'] = x3_mpi
        # feats['x4_mpi_feats'] = x4_mpi

        # feats['x1_mpi_track_feats'] = self.track_prediction(x1_mpi, self.track_prediction_head_x1_mpi)
        # feats['x2_mpi_track_feats'] = self.track_prediction(x2_mpi, self.track_prediction_head_x2_mpi)
        # feats['x3_mpi_track_feats'] = self.track_prediction(x3_mpi, self.track_prediction_head_x3_mpi)

        delta1_mri = self.delta_head1(x1_mpi)
        delta2_mri = self.delta_head2(x2_mpi)
        delta3_mri = self.delta_head3(x3_mpi)
        delta4_mri = self.delta_head4(x4_mpi)

        # feats['delta1_mri'] = delta1_mri
        # feats['delta2_mri'] = delta2_mri
        # feats['delta3_mri'] = delta3_mri

        # feats['M1'] = M1
        # feats['M2'] = M2
        # feats['M3'] = M3

        x4_fusion = x4_mpi + delta4_mri
        x5 = self.down4(x4_fusion)
        x3_fusion = x3_mpi + delta3_mri * M3
        x2_fusion = x2_mpi + delta2_mri * M2
        x1_fusion = x1_mpi + delta1_mri * M1


        # feats['x1_fusion_feats'] = x1_fusion
        # feats['x2_fusion_feats'] = x2_fusion
        # feats['x3_fusion_feats'] = x3_fusion

        # feats['x1_fusion_track_feats'] = self.track_prediction(x1_fusion, self.track_prediction_head_x1_fusion)
        # feats['x2_fusion_track_feats'] = self.track_prediction(x2_fusion, self.track_prediction_head_x2_fusion)
        # feats['x3_fusion_track_feats'] = self.track_prediction(x3_fusion, self.track_prediction_head_x3_fusion)

        # feats['x4_fusion_feats'] = x4_fusion
        # feats['x5_fusion_feats'] = x5

        x = self.up1(x5, x4_fusion)
        # feats['x1_out_feats'] = x

        x = self.up2(x, x3_fusion)
        # feats['x2_out_feats'] = x
        # feats['x2_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x2_out)

        x = self.up3(x, x2_fusion)
        # feats['x3_out_feats'] = x
        # feats['x3_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x3_out)

        x = self.up4(x, x1_fusion)
        # feats['x4_out_feats'] = x
        # feats['x4_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x4_out)
        
        logits = self.outc(x) + residual

        # seg_logits = self.seg_out(x)

        return logits, feats

    # def forward_feats(self, x_mpi):
    #     feats = {}

    #     x1_mpi = self.inc(x_mpi)
    #     x2_mpi = self.down1(x1_mpi)
    #     x3_mpi = self.down2(x2_mpi)
    #     x4_mpi = self.down3(x3_mpi)

    #     M1 = self.gate(x1_mpi)
    #     M2 = F.interpolate(M1, scale_factor=0.5)
    #     M3 = F.interpolate(M1, scale_factor=0.25)

    #     feats['x1_mpi_feats'] = x1_mpi
    #     feats['x2_mpi_feats'] = x2_mpi
    #     feats['x3_mpi_feats'] = x3_mpi
    #     # feats['x4_mpi_feats'] = x4_mpi

    #     feats['x1_mpi_track_feats'] = self.track_prediction(x1_mpi, self.track_prediction_head_x1_mpi)
    #     feats['x2_mpi_track_feats'] = self.track_prediction(x2_mpi, self.track_prediction_head_x2_mpi)
    #     feats['x3_mpi_track_feats'] = self.track_prediction(x3_mpi, self.track_prediction_head_x3_mpi)

    #     delta1_mri = self.delta_head1(x1_mpi)
    #     delta2_mri = self.delta_head2(x2_mpi)
    #     delta3_mri = self.delta_head3(x3_mpi)
    #     delta4_mri = self.delta_head4(x4_mpi)

    #     feats['delta1_mri'] = delta1_mri
    #     feats['delta2_mri'] = delta2_mri
    #     feats['delta3_mri'] = delta3_mri

    #     feats['M1'] = M1
    #     feats['M2'] = M2
    #     feats['M3'] = M3

    #     x4_fusion = x4_mpi + delta4_mri
    #     x5 = self.down4(x4_fusion)
    #     x3_fusion = x3_mpi + delta3_mri * M3
    #     x2_fusion = x2_mpi + delta2_mri * M2
    #     x1_fusion = x1_mpi + delta1_mri * M1


    #     feats['x1_fusion_feats'] = x1_fusion
    #     feats['x2_fusion_feats'] = x2_fusion
    #     feats['x3_fusion_feats'] = x3_fusion

    #     feats['x1_fusion_track_feats'] = self.track_prediction(x1_fusion, self.track_prediction_head_x1_fusion)
    #     feats['x2_fusion_track_feats'] = self.track_prediction(x2_fusion, self.track_prediction_head_x2_fusion)
    #     feats['x3_fusion_track_feats'] = self.track_prediction(x3_fusion, self.track_prediction_head_x3_fusion)

    #     # feats['x4_fusion_feats'] = x4_fusion
    #     # feats['x5_fusion_feats'] = x5

    #     x = self.up1(x5, x4_fusion)
    #     # feats['x1_out_feats'] = x

    #     x = self.up2(x, x3_fusion)
    #     feats['x2_out_feats'] = x
    #     feats['x2_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x2_out)

    #     x = self.up3(x, x2_fusion)
    #     feats['x3_out_feats'] = x
    #     feats['x3_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x3_out)

    #     x = self.up4(x, x1_fusion)
    #     feats['x4_out_feats'] = x
    #     feats['x4_out_track_feats'] = self.track_prediction(x, self.track_prediction_head_x4_out)
        
        

    #     logits = self.outc(x)
    #     # seg_logits = self.seg_out(x)

    #     return logits, feats
    
# class UNet_distil(nn.Module):
#     def __init__(self, n_channels, n_classes, T, bilinear=False, input_signal=True, sig_dim=None):
#         super(UNet_distil, self).__init__()
#         self.input_signal = input_signal
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.base_channel = 8

#         if self.input_signal:
#             assert sig_dim is not None
#             self.sig2img = nn.Sequential(
#                 nn.Linear(sig_dim, 4096),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Linear(4096, 4096)
#             )

#         # self.estimator = nn.Sequential(
#         #     DoubleConv(3, 64),
#         #     DoubleConv(64, 64),
#         #     DoubleConv(64, 1),
#         # ) 

#         self.inc = (DoubleConv(3, self.base_channel))
#         self.down1 = (Down(self.base_channel, self.base_channel * 2))
#         self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
#         self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

#         factor = 2 if bilinear else 1
#         self.down4 = (Down(self.base_channel * 8, self.base_channel * 16 // factor))
#         self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
#         self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
#         self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
#         self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))
#         self.outc = (OutConv(self.base_channel, n_classes))

#     def forward(self, x_mpi, noisy_NR, init_pred_xstart):
#         if self.input_signal:
#             x_mpi = self.sig2img(x_mpi)
#             x_mpi = x_mpi.reshape(x_mpi.shape[0], 1, 64, 64)

#         x_mpi = torch.cat([x_mpi, noisy_NR, init_pred_xstart], 1)
#         # x_mpi = self.estimator(x_mpi)

#         x1 = self.inc(x_mpi)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)

#         x5 = self.down4(x4)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
    
#     def forward_feats(self, x_mpi, noisy_NR, init_pred_xstart):
#         if self.input_signal:
#             x_mpi = self.sig2img(x_mpi)
#             x_mpi = x_mpi.reshape(x_mpi.shape[0], 1, 64, 64)
        
#         x_mpi = torch.cat([x_mpi, noisy_NR, init_pred_xstart], 1)
#         # x_mpi = self.estimator(x_mpi)
#         feats = []

#         x1 = self.inc(x_mpi)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         feats.extend([x1, x2, x3, x4, x5])

#         x = self.up1(x5, x4)
#         feats.append(x)
#         x = self.up2(x, x3)
#         feats.append(x)
#         x = self.up3(x, x2)
#         feats.append(x)
#         x = self.up4(x, x1)
#         feats.append(x)
#         logits = self.outc(x)
#         feats.append(logits)

#         feats.append(x_mpi)

#         return logits, feats
    

# class UNet_distil2(nn.Module):
#     def __init__(self, n_channels, n_classes, T, bilinear=False, input_signal=True, sig_dim=None):
#         super(UNet_distil2, self).__init__()
#         self.input_signal = input_signal
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.base_channel = 8 # 8

#         if self.input_signal:
#             assert sig_dim is not None
#             self.sig2img = nn.Sequential(
#                 nn.Linear(sig_dim, 4096),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Linear(4096, 4096)
#             )

#         # self.estimator = nn.Sequential(
#         #     DoubleConv(3, 64),
#         #     DoubleConv(64, 64),
#         #     DoubleConv(64, 1),
#         # ) 

#         # ------------- mpi
#         self.inc = (DoubleConv(3, self.base_channel))
#         self.down1 = (Down(self.base_channel, self.base_channel * 2))
#         self.down2 = (Down(self.base_channel * 2, self.base_channel * 4))
#         self.down3 = (Down(self.base_channel * 4, self.base_channel * 8))

#         factor = 2 if bilinear else 1
#         self.down4 = (Down(self.base_channel * 16, self.base_channel * 16 // factor))

#         # ------------- fusion
#         self.inc_fusion = (DoubleConv(3, self.base_channel))
#         self.down1_fusion = (Down(self.base_channel, self.base_channel * 2))
#         self.down2_fusion = (Down(self.base_channel * 2, self.base_channel * 4))
#         self.down3_fusion = (Down(self.base_channel * 4, self.base_channel * 8))
#         self.down4_fusion = (Down(self.base_channel * 8, self.base_channel * 16 // factor))

#         self.up1 = (Up(self.base_channel * 16, self.base_channel * 8 // factor, bilinear))
#         self.up2 = (Up(self.base_channel * 8, self.base_channel * 4 // factor, bilinear))
#         self.up3 = (Up(self.base_channel * 4, self.base_channel * 2 // factor, bilinear))
#         self.up4 = (Up(self.base_channel * 2, self.base_channel, bilinear))

#         self.outc = (OutConv(self.base_channel, n_classes))

#         # self.compress_channel_op1_fusion = torch.nn.Conv2d(self.base_channel, self.base_channel // 2, kernel_size=1)
#         # self.compress_channel_op2_fusion = torch.nn.Conv2d(self.base_channel * 2, self.base_channel, kernel_size=1)
#         # self.compress_channel_op3_fusion = torch.nn.Conv2d(self.base_channel * 4, self.base_channel * 2, kernel_size=1)
#         # self.compress_channel_op4_fusion = torch.nn.Conv2d(self.base_channel * 8, self.base_channel * 4, kernel_size=1)
#         # self.compress_channel_op5_fusion = torch.nn.Conv2d(self.base_channel * 16, self.base_channel * 8, kernel_size=1)

#         # self.compress_channel_op1_mpi = torch.nn.Conv2d(self.base_channel, self.base_channel // 2, kernel_size=1)
#         # self.compress_channel_op2_mpi = torch.nn.Conv2d(self.base_channel * 2, self.base_channel, kernel_size=1)
#         # self.compress_channel_op3_mpi = torch.nn.Conv2d(self.base_channel * 4, self.base_channel * 2, kernel_size=1)
#         # self.compress_channel_op4_mpi = torch.nn.Conv2d(self.base_channel * 8, self.base_channel * 4, kernel_size=1)

#         # self.compress_channel_op6 = torch.nn.Conv2d(self.base_channel * 8, self.base_channel * 4, kernel_size=1)
#         # self.compress_channel_op7 = torch.nn.Conv2d(self.base_channel * 4, self.base_channel * 2, kernel_size=1)
#         # self.compress_channel_op8 = torch.nn.Conv2d(self.base_channel * 2, self.base_channel, kernel_size=1)
#         # self.compress_channel_op9 = torch.nn.Conv2d(self.base_channel, self.base_channel // 2, kernel_size=1)

#     def forward(self, x_mpi, noisy_NR, init_pred_xstart):
#         if self.input_signal:
#             x_mpi = self.sig2img(x_mpi)
#             x_mpi = x_mpi.reshape(x_mpi.shape[0], 1, 64, 64)

#         x_mpi = torch.cat([x_mpi, noisy_NR, init_pred_xstart], 1)

#         x1_mpi = self.inc(x_mpi)
#         x2_mpi = self.down1(x1_mpi)
#         x3_mpi = self.down2(x2_mpi)
#         x4_mpi = self.down3(x3_mpi)

#         x1_fusion = self.inc_fusion(x_mpi)
#         x2_fusion = self.down1_fusion(x1_fusion)
#         x3_fusion = self.down2_fusion(x2_fusion)
#         x4_fusion = self.down3_fusion(x3_fusion)

#         x5 = self.down4(torch.cat([x4_mpi, x4_fusion], 1))

#         x = self.up1(x5, x4_fusion)
#         x = self.up2(x, x3_fusion)
#         x = self.up3(x, x2_fusion)
#         x = self.up4(x, x1_fusion)
#         logits = self.outc(x)
#         return logits
  
#     def forward_feats(self, x_mpi, noisy_NR, init_pred_xstart):
#         feats = {}

#         if self.input_signal:
#             x_mpi = self.sig2img(x_mpi)
#             x_mpi = x_mpi.reshape(x_mpi.shape[0], 1, 64, 64)

#         x_mpi = torch.cat([x_mpi, noisy_NR, init_pred_xstart], 1)

#         x1_mpi = self.inc(x_mpi)
#         x2_mpi = self.down1(x1_mpi)
#         x3_mpi = self.down2(x2_mpi)
#         x4_mpi = self.down3(x3_mpi)

#         feats['x1_mpi_feats'] = x1_mpi
#         feats['x2_mpi_feats'] = x2_mpi
#         feats['x3_mpi_feats'] = x3_mpi
#         feats['x4_mpi_feats'] = x4_mpi

#         x1_fusion = self.inc_fusion(x_mpi)
#         x2_fusion = self.down1_fusion(x1_fusion)
#         x3_fusion = self.down2_fusion(x2_fusion)
#         x4_fusion = self.down3_fusion(x3_fusion)
#         x5 = self.down4(torch.cat([x4_mpi, x4_fusion], 1))

#         feats['x1_fusion_feats'] = x1_fusion
#         feats['x2_fusion_feats'] = x2_fusion
#         feats['x3_fusion_feats'] = x3_fusion
#         feats['x4_fusion_feats'] = x4_fusion
#         feats['x5_fusion_feats'] = x5

#         x = self.up1(x5, x4_fusion)
#         feats['x1_out_feats'] = x

#         x = self.up2(x, x3_fusion)
#         feats['x2_out_feats'] = x

#         x = self.up3(x, x2_fusion)
#         feats['x3_out_feats'] = x

#         x = self.up4(x, x1_fusion)
#         feats['x4_out_feats'] = x

#         logits = self.outc(x)

#         return logits, feats
    
if __name__ == '__main__':
    None
    # fusion_branch = UNet_multimodal(1, 1, T=1000)

    # x_mpi = torch.randn(16, 1, 64, 64)
    # x_mri = torch.randn(16, 1, 64, 64)

    # t = torch.tensor([10] * 16).long()
    # logits, MPI_feats, MRI_feats, fusion_feats = fusion_branch.get_feature_maps(x_mpi, x_mri, t)
    # print('-------------------------------------')
    # print(logits.shape)

    # print('-------------------------------------')
    # for mpi in MPI_feats:
    #     print(mpi.shape)
    
    # print('-------------------------------------')
    # for mri in MRI_feats:
    #     print(mri.shape)

    # print('-------------------------------------')
    # for f in fusion_feats:
    #     print(f.shape)

    model = UNet_multimodal_distill(1, 1, T=1, sig_dim=9000)
    sig = torch.randn(16, 1, 9000)

    xx = torch.randn(16, 1, 64, 64)
    logits, feats = model.forward_feat(xx, sig)
    print(logits.shape)
    for k in feats.keys():
        print(k, feats[k].shape)

    # print('-------------------------------------')
    # for k in MPI_feats.keys():
    #     print(k, MPI_feats[k].shape)
    
    # print(MPI_feats.keys())

    # indices = list(range(100))[::-1]
    # print(indices)

    # for i in indices:
    #     print(i)

# model = UNet_multimodal(1, 1, 1000)

# x_mpi = torch.randn(16, 1, 64, 64)
# x_mri = torch.randn(16, 1, 64, 64)
# t = torch.randint(1, 1000, (16, ))
# print(model(x_mpi, x_mri, t).shape)


