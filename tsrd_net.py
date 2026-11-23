"""TSRD-Net model implementation."""

import copy
import math
from typing import Tuple

import timm
import torch
from einops import rearrange
from kornia.filters import joint_bilateral_blur
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d


def get_activation(act: str, inpace: bool=True):
    act = act.lower()
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'silu':
        m = nn.SiLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError('')
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    return m


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DeformableConvBranch(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, bias=False)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = 1
        self.padding = padding

    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride, padding=self.padding)


class HybridLocalGlobalMixer(nn.Module):

    def __init__(self, dim, kernel_size=3, heads=8):
        super().__init__()
        self.local_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False), nn.BatchNorm2d(dim), nn.SiLU(inplace=True))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.local_conv(x) + x
        seq = rearrange(y, 'b c h w -> b (h w) c')
        attn_out, _ = self.attn(seq, seq, seq)
        attn_out = self.proj(attn_out)
        y2 = attn_out.transpose(1, 2).reshape(B, C, H, W)
        return y2 + y


class ImageGuidedCRF(nn.Module):

    def __init__(self, num_classes: int, num_iterations: int=5, spatial_ksize: int=7, spatial_sigma: float=3.0, bilateral_ksize: int=7, bilateral_sigma_spatial: float=3.0, bilateral_sigma_color: float=13.0, compat_spatial: float=3.0, compat_bilateral: float=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.compat_spatial = compat_spatial
        self.compat_bilateral = compat_bilateral
        ax = torch.arange(spatial_ksize) - (spatial_ksize - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * spatial_sigma ** 2))
        kernel = kernel / kernel.sum()
        self.register_buffer('spatial_kernel', kernel[None, None])
        self.bilateral_ksize = bilateral_ksize
        self.bilateral_sigma_space = bilateral_sigma_spatial
        self.bilateral_sigma_color = bilateral_sigma_color

    def forward(self, probs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = probs.shape
        device = probs.device
        U = -torch.log(probs + 1e-08)
        Q = probs
        for _ in range(self.num_iterations):
            q_flat = Q.view(B * C, 1, H, W)
            pad = self.spatial_kernel.shape[-1] // 2
            spat = F.conv2d(q_flat, self.spatial_kernel.to(device), padding=pad)
            spat = spat.view(B, C, H, W) * self.compat_spatial
            sigma_color = torch.full((B,), self.bilateral_sigma_color, device=device, dtype=Q.dtype)
            sigma_space = torch.full((B, 2), self.bilateral_sigma_space, device=device, dtype=Q.dtype)
            bilat = joint_bilateral_blur(Q, image, (self.bilateral_ksize, self.bilateral_ksize), sigma_color=sigma_color, sigma_space=sigma_space)
            bilat = bilat * self.compat_bilateral
            pairwise = spat + bilat
            Q = F.softmax(-U - pairwise, dim=1)
        return Q


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(-1, C, window_size, window_size)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] // (H * W / window_size ** 2))
    C = windows.shape[1]
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, C, H, W)


class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, height, width):
        super().__init__()
        assert d_model % 2 == 0, 'd_model must be even'
        pe = torch.zeros(d_model, height, width)
        y = torch.arange(height, dtype=torch.float).unsqueeze(1)
        x = torch.arange(width, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        y_enc = torch.sin(y * div_term.unsqueeze(0))
        y_enc = y_enc.transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[0::2, :, :] = y_enc
        x_enc = torch.cos(x * div_term.unsqueeze(1))
        x_enc = x_enc.unsqueeze(1).repeat(1, height, 1)
        pe[1::2, :, :] = x_enc
        self.register_buffer('pe', pe.view(d_model, height * width).permute(1, 0))

    def forward(self, B, device):
        return self.pe.unsqueeze(1).expand(-1, B, -1).to(device)


class UpTR(nn.Module):

    def __init__(self, in_dim: int, embed_dim: int, scale_factor: int, window_size: int, nhead: int, out_size: Tuple[int, int]):
        super().__init__()
        assert window_size % scale_factor == 0, 'window_size must be a multiple of scale_factor'
        H_out, W_out = out_size
        self.scale = scale_factor
        self.ws = window_size
        self.proj_in = nn.Conv2d(in_dim, embed_dim, 1)
        self.mem_self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=False)
        self.mem_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=False)
        self.cross_norm = nn.LayerNorm(embed_dim)
        pe = PositionalEncoding2D(embed_dim, H_out, W_out).pe
        self.register_buffer('pos_enc', pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        D = self.proj_in.out_channels
        H2, W2 = (H * self.scale, W * self.scale)
        mem = self.proj_in(x)
        mem_w = window_partition(mem, self.ws // self.scale)
        mem_w = mem_w.flatten(2).permute(2, 0, 1)
        sa_out, _ = self.mem_self_attn(mem_w, mem_w, mem_w)
        mem_w = self.mem_norm(sa_out + mem_w)
        qry = self.pos_enc.unsqueeze(1).expand(-1, B, -1)
        qry = qry.permute(1, 2, 0).view(B, D, H2, W2)
        qry_w = window_partition(qry, self.ws).flatten(2).permute(2, 0, 1)
        ca_out, _ = self.cross_attn(qry_w, mem_w, mem_w)
        out = self.cross_norm(ca_out + qry_w)
        out = out.permute(1, 2, 0).view(-1, D, self.ws, self.ws)
        x_up = window_reverse(out, self.ws, H2, W2)
        x_bilinear = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x_up = x_up + x_bilinear
        return x_up


class ConvBNSiLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2), norm_layer(out_channels), nn.SiLU(inplace=True))


class Conv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2))


class ConvBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2), norm_layer(out_channels))


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, if_downsample=False, **kwargs):
        super(ResidualBlock, self).__init__()
        self.if_downsample = if_downsample
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channel))

    def forward(self, x):
        identity = x
        if self.if_downsample:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class SpatialAttentionLayer(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.avgpool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.maxpool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.avgpool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxpool_w = nn.AdaptiveMaxPool2d((1, None))
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.SiLU(inplace=True))
        self.norm = nn.BatchNorm2d(d_model)
        self.norm1 = nn.GroupNorm(16, d_model)
        self.norm2 = nn.GroupNorm(16, d_model)
        self.ffn = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False), nn.SiLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(d_model, d_model, kernel_size=1, padding=0, bias=False))

    def forward(self, x, v):
        B, C, H, W = v.shape
        x_h = (self.avgpool_h(x) + self.maxpool_h(x)).squeeze(3).permute(0, 2, 1)
        x_w = (self.avgpool_w(x) + self.maxpool_w(x)).squeeze(2).permute(0, 2, 1)
        q = self.mlp(x_h)
        k = self.mlp(x_w).transpose(-1, -2)
        weight_score = torch.matmul(q, k)
        weight_probs = nn.ReLU(inplace=True)(weight_score).unsqueeze(1)
        src2 = weight_probs * v
        src = v + self.norm(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + src2
        src = self.norm2(src)
        return src.reshape(B, C, H, W)


class SpatialSkipAttention(nn.Module):

    def __init__(self, d_model, num_layers, norm=None):
        super().__init__()
        SAL = SpatialAttentionLayer(d_model)
        self.layers = nn.ModuleList([copy.deepcopy(SAL) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src1, src2=None) -> torch.Tensor:
        if src2 is None:
            src2 = src1
        output = src1
        for layer in self.layers:
            output = layer(output, src2)
        if self.norm is not None:
            output = self.norm(output)
        return output


class SkipFusion(nn.Module):

    def __init__(self, in_channels=128, decode_channels=128, img_size=16, dropout=0.1, window_size=8, nhead=4, out_size=None):
        super().__init__()
        out_size = out_size or (img_size, img_size)
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=1)
        self.upsampler = UpTR(in_dim=decode_channels, embed_dim=decode_channels, scale_factor=2, window_size=window_size, nhead=nhead, out_size=out_size)
        self.core = HybridLocalGlobalMixer(decode_channels, kernel_size=3, heads=8)
        self.post_conv = ResidualBlock(decode_channels, decode_channels)
        self.attn = SpatialSkipAttention(decode_channels, 1)

    def forward(self, x, res):
        x = self.upsampler(x)
        res = self.pre_conv(res)
        x = x + self.attn(res)
        x = self.core(x)
        x = self.post_conv(x)
        return x


class DeformableRefinement(nn.Module):

    def __init__(self, in_channels=64, decode_channels=64, dropout=0.1):
        super().__init__()
        self.channel = decode_channels
        split_c = decode_channels // 4
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=1)
        self.attn = SpatialSkipAttention(decode_channels, 1)
        self.conv_f = ConvBN(decode_channels * 2, decode_channels, kernel_size=1)
        self.conv1x1 = ConvBNSiLU(decode_channels // 4, decode_channels // 4, kernel_size=1)
        self.conv3x3 = ConvBNSiLU(decode_channels // 4, decode_channels // 4, kernel_size=3)
        self.branch5 = DeformableConvBranch(split_c, split_c, kernel_size=5, padding=2)
        self.branch7 = DeformableConvBranch(split_c, split_c, kernel_size=7, padding=3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-08
        self.post_conv = ResidualBlock(decode_channels, decode_channels, if_downsample=True)
        self.upsampler = UpTR(in_dim=decode_channels, embed_dim=decode_channels, scale_factor=2, window_size=16, nhead=4, out_size=(128, 128))

    def forward(self, x, res):
        x = self.upsampler(x)
        res = self.pre_conv(res)
        x = torch.cat([x, self.attn(res)], dim=1)
        x = self.conv_f(x)
        split = self.channel // 4
        x1 = self.conv1x1(x[:, :split, :, :])
        x2 = self.conv3x3(x[:, split:2 * split, :, :])
        x3 = self.branch5(x[:, 2 * split:3 * split])
        x4 = self.branch7(x[:, 3 * split:, :, :])
        x = torch.cat([x1, x2, x3, x4], dim=1)
        identity = x
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        x = x * y.expand_as(x)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * identity + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class AuxiliaryHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNSiLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class ReconstructionDecoder(nn.Module):

    def __init__(self, encoder_channels, decode_channels, num_classes, dropout=0.1, window_size=8):
        super().__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        encoder_layer = TransformerEncoderLayer(decode_channels, nhead=8, dim_feedforward=decode_channels * 2, dropout=dropout, activation='silu')
        self.attn = TransformerEncoder(copy.deepcopy(encoder_layer), 6)
        self.p3 = SkipFusion(in_channels=encoder_channels[-2], decode_channels=decode_channels, img_size=32, window_size=8, out_size=(32, 32))
        self.p2 = SkipFusion(in_channels=encoder_channels[-3], decode_channels=decode_channels, img_size=64, window_size=16, out_size=(64, 64))
        self.p1 = DeformableRefinement(encoder_channels[-4], decode_channels)
        self.h = AuxiliaryHead(decode_channels, num_classes)
        self.segmentation_head = nn.Sequential(ConvBNSiLU(decode_channels, decode_channels), nn.Dropout2d(p=dropout, inplace=True), Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()
        self.up_final = UpTR(in_dim=decode_channels, embed_dim=decode_channels, scale_factor=4, window_size=32, nhead=4, out_size=(512, 512))

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.pre_conv(res4)
        B, C, H, W = x.shape
        x = self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        if self.training:
            ah = self.h(x)
        x = self.p3(x, res3)
        x = self.p2(x, res2)
        x = self.p1(x, res1)
        x = self.up_final(x)
        x = self.segmentation_head(x)
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        if self.training:
            return (x, ah)
        else:
            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TSRDNet(nn.Module):

    def __init__(self, decode_channels=256, backbone_name='resnext101_32x4d.fb_swsl_ig1b_ft_in1k', pretrained=False, num_classes=6, crf_iters: int=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = ReconstructionDecoder(encoder_channels, decode_channels, num_classes=num_classes)
        self.crfrnn = ImageGuidedCRF(num_classes=num_classes, num_iterations=crf_iters, spatial_ksize=7, spatial_sigma=3.0, bilateral_ksize=7, bilateral_sigma_spatial=3.0, bilateral_sigma_color=13.0, compat_spatial=3.0, compat_bilateral=2.0)

    def forward(self, x):
        _, _, H, W = x.shape
        res1, res2, res3, res4 = self.backbone(x)
        outputs = self.decoder(res1, res2, res3, res4, H, W)
        if self.training:
            return outputs
        logits = outputs
        img_n = x.clamp(0, 1)
        return self._refine_probs(logits, img_n)

    def _refine_probs(self, logits, image):
        probs = F.softmax(logits, dim=1)
        if image.shape[-2:] != probs.shape[-2:]:
            image = F.interpolate(image, size=probs.shape[-2:], mode='bilinear', align_corners=False)
        return self.crfrnn(probs, image)
