from ltr import model_constructor
from functools import reduce

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       nested_tensor_from_tensor_list, accuracy)

from ltr.models.loss.matcher import build_matcher
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import collections.abc as container_abcs
from itertools import repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from typing import Optional
import copy
from ltr.models.neck.position_encoding import build_position_encoding2
from ltr.models.neck.featurefusion_network import build_CrossAttnention_network
from ltr.models.neck.convolutional_block_attention_module import build_CBAM_network
from ltr.models.neck.vit_encoder import build_featurefusion_network

from ltr.models.tracking.transt_seg import (TransTsegm, dice_loss, sigmoid_focal_loss)
from ltr.models.tracking.transt_iouhead import TransTiouh
from ltr.models.tracking.transt_iouh_seg import TransTiouhsegm
from ltr.models.tracking.SA_Tracker_iouhead import MixTrackingiouh
from ltr.models.neck.vit_encoder import Transformer as global_attention
# from lib.models.mixformer.position_encoding import build_position_encoding

# TODO: update the urls of the pre-trained models
MODEL_URLS = {
    "mixformer-B0": "",
    "mixformer-B1": "",
    "mixformer-B2": "",
    "mixformer-B3": "",
    "mixformer-B4": "",
    "mixformer-B5": "",
    "mixformer-B6": "",
}

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

__all__ = list(MODEL_URLS.keys())

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 mlp_fc2_bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_fc2_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition2(x, window_size):
    """ Split the feature map to windows.
    B, C, H, W --> B * H // win * W // win x win*win x C

    Args:
        x: (B, C, H, W)
        window_size (tuple[int]): window size

    Returns:
        windows: (num_windows*B, window_size * window_size, C)
    """
    B, C, H, W = x.shape
    x = x.reshape(
        [B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1]])
    windows = x.permute([0, 2, 4, 3, 5, 1]).reshape(
        [-1, window_size[0] * window_size[1], C])
    return windows


def window_reverse2(windows, window_size, H, W, C):
    """ Windows reverse to feature map.
    B * H // win * W // win x win*win x C --> B, C, H, W

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    x = windows.reshape(
        [-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C])
    x = x.permute([0, 5, 1, 3, 2, 4]).reshape([-1, C, H, W])
    return x

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MixingAttention(nn.Module):
    r""" Mixing Attention Module.
    Modified from Window based multi-head self attention (W-MSA) module with 
    relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self,
                dim,
                conv_ratio,
                window_size,
                dwconv_kernel_size,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.,
                proj_drop=0.):
        super().__init__()
        self.dim = dim
        attn_dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        pretrained_window_size = [0, 0, 0, 0]
        self.scale = qk_scale or head_dim ** -0.5
        conv_ratio = conv_ratio

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        self.relative_position_bias_table_ = nn.Parameter(
            torch.zeros((4 * window_size[0] - 1) * (4 * window_size[1] - 1), num_heads)) 
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_index_ = self._get_rel_pos()

        # prev proj layer
        self.proj_search = nn.Linear(dim, dim // 2)
        self.proj_search_norm = nn.LayerNorm(dim // 2)
        self.proj_template = nn.Linear(dim, dim // 2)
        self.proj_template_norm = nn.LayerNorm(dim // 2)

        # window-attention branch
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        # self.w_qs = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)
        # self.w_ks = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)
        # self.w_vs = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # final projection
        self.proj_template_reverse = nn.Linear(dim // 2, dim)
        self.proj_drop_template = nn.Dropout(proj_drop)

        self.proj_search_reverse = nn.Linear(dim // 2, dim)
        self.proj_drop_search = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.relative_position_bias_table_, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos(self):
        self.window_size_ = [ws * 2 for ws in self.window_size]
        coords_h = torch.arange(self.window_size_[0])
        coords_w = torch.arange(self.window_size_[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size_[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size_[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size_[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def xcorr_pixelwise(self, x, kernel): #z=kernel 
        """Pixel-wise correlation (implementation by matrix multiplication)
        The speed is faster because the computation is vectorized"""

        b, c, h, w = x.size() 
        kernel_mat = kernel.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)
        x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
        return torch.matmul(kernel_mat, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)
    
    def forward(self, x, template, s_H, s_W, t_H, t_W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        search = x

        # B * H // win * W // win x win*win x C
        search_attn = self.proj_search_norm(self.proj_search(search))
        search_window_num = search_attn.shape[0]

        template_attn = self.proj_template_norm(self.proj_template(template))
        template_window_num = template_attn.shape[0]

        # relative position encoding
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # attention branch
        x_atten = torch.cat([template_attn, search_attn], axis=0)
        B_, N, C = x_atten.shape
        qkv = self.qkv(x_atten).reshape([B_, N, 3, self.num_heads, C // self.num_heads]).permute([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + relative_position_bias.unsqueeze(0)

        if all(mask):
            # search mask
            nW = mask[0].shape[0]
            attn_ = attn[template_window_num:, :, :, :].reshape([search_window_num // nW, nW, self.num_heads, N, N]) + mask[0].unsqueeze(1).unsqueeze(0)
            attn_ = attn_.reshape([-1, self.num_heads, N, N])
            attn[template_window_num:, :, :, :] = attn_

            # template mask
            nW = mask[1].shape[0]
            attn_ = attn[:template_window_num, :, :, :].reshape([template_window_num // nW, nW, self.num_heads, N, N]) + mask[1].unsqueeze(1).unsqueeze(0)
            attn_ = attn_.reshape([-1, self.num_heads, N, N])
            attn[:template_window_num, :, :, :] = attn_

            attn = self.softmax(attn)

        else:
            if not any(mask) and all(mask):
                raise ValueError('Do not allow one None mask')
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        template_attn, search_attn = torch.split(x_atten, [template_window_num, search_window_num], dim=0)

        template_attn = self.proj_template_reverse(template_attn)
        template_attn = self.proj_drop_template(template_attn)

        search_attn = self.proj_search_reverse(search_attn)
        search_attn = self.proj_drop_search(search_attn)

        return template_attn, search_attn

    
    def extra_repr(self):
        return "dim={}, window_size={}, num_heads={}".format(
            self.dim, self.window_size, self.num_heads)

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # projection layers
        flops += N * self.dim * self.dim * 3 // 2
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class MixingBlock(nn.Module):
    r""" Mixing Block in MixFormer.
    Modified from Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        shift_size (int): Shift size for SW-MSA. We do not use shift in MixFormer. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 conv_ratio,
                 num_heads,
                 window_size=7,
                 dwconv_kernel_size=3,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 global_attn=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        conv_ratio = conv_ratio
        self.input_resolution = input_resolution
        # assert self.shift_size == 0, "No shift in MixFormer"
        if global_attn:
            self.template_global_attn = global_attention(dim=dim, depth=1, heads=num_heads, dim_head=dim // num_heads, mlp_dim=int(mlp_ratio)*dim)
            self.search_global_attn = global_attention(dim=dim, depth=1, heads=num_heads, dim_head=dim // num_heads, mlp_dim=int(mlp_ratio)*dim)
        
        self.CBAM_template = build_CBAM_network(dim=dim)
        self.CBAM_search = build_CBAM_network(dim=dim)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution.get('temp_sz')
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask_tempalte = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask_tempalte = attn_mask_tempalte.masked_fill(attn_mask_tempalte != 0, float(-100.0)).masked_fill(attn_mask_tempalte == 0, float(0.0))
        else:
            attn_mask_tempalte = None

        self.register_buffer("attn_mask_tempalte", attn_mask_tempalte)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution.get('search_sz')
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask_search = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask_search = attn_mask_search.masked_fill(attn_mask_search != 0, float(-100.0)).masked_fill(attn_mask_search == 0, float(0.0))
        else:
            attn_mask_search = None

        self.register_buffer("attn_mask_search", attn_mask_search)

        self.norm1 = norm_layer(dim)
        self.norm_search = norm_layer(dim)
        self.norm_tempalte = norm_layer(dim)
        self.attn = MixingAttention(
            dim,
            conv_ratio,
            window_size=to_2tuple(self.window_size),
            dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_search = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        
        self.mlp_template = Mlp(in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop)
        # self.H = None
        # self.W = None

        self.t_H = None
        self.t_W = None
        self.s_H = None
        self.s_W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        # template, search = torch.split(x, [self.t_H*self.t_W, self.s_H*self.s_W], dim=1)
        template, search = x[0], x[1]
        # mask_t, mask_t_online, mask_s = key_padding_mask[0][0], key_padding_mask[0][1], key_padding_mask[1]

        B_s, L, C = search.shape
        H, W = self.s_H, self.s_W
        assert L == H * W, "search feature has wrong size"

        B_t, L, C = template.shape
        H, W = self.t_H, self.t_W
        assert L == H * W, "search feature has wrong size"

        shortcut_search = search
        search = self.norm_search(search)
        search = search.reshape([B_s, self.s_H, self.s_W, C])

        shortcut_tempalte = template
        template = self.norm_tempalte(template)
        template = template.reshape([B_t, self.t_H, self.t_W, C])

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        search = F.pad(search, [0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t])
        template = F.pad(template, [0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t])

        t_H, t_W, s_H, s_W = self.t_H, self.t_W, self.s_H, self.s_W

        # cyclic shift
        if self.shift_size > 0:
            shifted_search = torch.roll(
                search, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask_search = self.attn_mask_search

            shifted_template = torch.roll(
                template, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask_template = self.attn_mask_tempalte
        else:
            shifted_search = search
            attn_mask_search = None

            shifted_template = template
            attn_mask_template = None
        
        # partition windows
        search_windows = window_partition(
            shifted_search, self.window_size)  # nW*B, window_size, window_size, C
        search_windows = search_windows.reshape(
            [-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C
        
        template_windows = window_partition(
            shifted_template, self.window_size)  # nW*B, window_size, window_size, C
        template_windows = template_windows.reshape(
            [-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        template_attn, search_attn = self.attn(
            search_windows, template_windows, s_H, s_W, t_H, t_W, mask=[attn_mask_search, attn_mask_template])  # nW*B, window_size*window_size, C
        
        # merge windows
        search_attn = search_attn.reshape(
            [-1, self.window_size, self.window_size, C])
        shifted_search_attn = window_reverse(search_attn, self.window_size, s_H, s_W)  # B H' W' C
        template_attn = template_attn.reshape(
            [-1, self.window_size, self.window_size, C])
        shifted_template_attn = window_reverse(template_attn, self.window_size, t_H, t_W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_search = torch.roll(
                shifted_search_attn,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
            
            attn_template = torch.roll(
                shifted_template_attn,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
            
        else:
            attn_search = shifted_search_attn
            attn_template = shifted_template_attn

        if pad_r > 0 or pad_b > 0:
            attn_search = attn_search[:, :s_H, :s_W, :]
            attn_template = attn_template[:, :t_H, :t_W, :]

        attn_search = attn_search.reshape([B_s, s_H * s_W, C])
        attn_template = attn_template.reshape([B_t, t_H * t_W, C])

        # FFN
        attn_search = shortcut_search + self.drop_path(attn_search)
        attn_search = attn_search + self.drop_path(self.mlp_search(self.norm1(attn_search)))

        attn_template = shortcut_tempalte + self.drop_path(attn_template)
        attn_template = attn_template + self.drop_path(self.mlp_template(self.norm2(attn_template)))

        return attn_search, attn_template
    

    def extra_repr(self):
        return "dim={}, input_resolution={}, num_heads={}, window_size={}, shift_size={}, mlp_ratio={}".format(
            self.dim, self.input_resolution, self.num_heads, self.window_size,
            self.shift_size, self.mlp_ratio)

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        
        # Mixing Attention
        flops += self.dim * H * W  # proj_cnn_norm
        flops += self.dim // 2 * H * W  # proj_attn_norm
        flops += self.dim * 1 * (self.conv_kernel_size ** 2) * H * W  # dwconv
        flops += self.dim * H * W  # batchnorm
        flops += self.dim * self.dim // 2 * H * W  # conv1x1
        # channel_interaction
        flops += self.dim * self.dim // 8 * 1 * 1
        flops += self.dim // 8 * 1 * 1
        flops += self.dim // 8 * self.dim // 2 * 1 * 1
        # spatial_interaction
        flops += self.dim // 2 * self.dim // 16 * H * W
        flops += self.dim // 16 * H * W
        flops += self.dim // 16 * 1 * H * W
        # branch norms
        flops += self.dim // 2 * H * W
        flops += self.dim // 2 * H * W
        # inside Mixing Attention
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class ConvMerging(nn.Module):
    r""" Conv Merging Layer.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Output channels after the merging layer.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, conv_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.reduction_temp = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2) if conv_ratio < 4 else nn.Conv2d(dim, out_dim, kernel_size=1, stride=1)
        self.norm_temp = nn.BatchNorm2d(dim)

        self.reduction_search = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2) if conv_ratio < 4 else nn.Conv2d(dim, out_dim, kernel_size=1, stride=1)
        self.norm_search = nn.BatchNorm2d(dim)

    def forward(self, template, search, t_H, t_W, s_H, s_W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B_t, L, C = template.shape
        assert L == t_H * t_W, "input feature has wrong size"
        assert t_H % 2 == 0 and t_W % 2 == 0, f"template size ({t_H}*{t_W}) are not even."

        B_s, L, C = search.shape
        assert L == s_H * s_W, "input feature has wrong size"
        assert s_H % 2 == 0 and s_W % 2 == 0, f"search size ({s_H}*{s_W}) are not even."

        template = template.reshape([B_t, t_H, t_W, C]).permute([0, 3, 1, 2])
        search = search.reshape([B_s, s_H, s_W, C]).permute([0, 3, 1, 2])

        template = self.norm_temp(template)
        template = self.reduction_temp(template).flatten(2).permute([0, 2, 1]) # B, C, H, W -> B, H*W, C

        search = self.norm_temp(search)
        search = self.reduction_search(search).flatten(2).permute([0, 2, 1]) # B, C, H, W -> B, H*W, C
        
        return template, search

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic layer for one stage in MixFormer.
    Modified from Swin Transformer BasicLayer.
    
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        out_dim (int): Output channels for the downsample layer. Default: 0.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 conv_ratio,
                 depth,
                 num_heads,
                 window_size=7,
                 dwconv_kernel_size=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 out_dim=0,
                 pos_type='sine',
                 nums_templates=2,
                 global_attn=False):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.conv_ratio = conv_ratio
        self.input_resolution = input_resolution
        self.global_attn = global_attn
        
        if global_attn:
            self.temp_pos_embedding = nn.Parameter(torch.randn(1, reduce(lambda x, y: x*y, (input_resolution.get('temp_sz'))), dim))
            trunc_normal_(self.temp_pos_embedding, std=.02)
            self.search_pos_embedding = nn.Parameter(torch.randn(1, reduce(lambda x, y: x*y, (input_resolution.get('search_sz'))), dim))
            trunc_normal_(self.search_pos_embedding, std=.02)

        # self.generate_pos = build_position_encoding2(dim, position_embedding_type=pos_type)

        # build blocks
        self.blocks = nn.ModuleList([
            MixingBlock(
                dim=dim,
                input_resolution=input_resolution,
                conv_ratio=conv_ratio,
                num_heads=num_heads,
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                # shift_size=0 if (i % 2 == 0) else window_size // 2,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, (np.ndarray, list)) else drop_path,
                norm_layer=norm_layer,
                global_attn=global_attn) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer, conv_ratio=conv_ratio)
        else:
            self.downsample = None

    def forward(self, x, t_H, t_W, s_H, s_W):
        """ Forward function.
        
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # mask_t, mask_t_online, mask_s = mask[0][0], mask[0][1], mask[1]
        # mask_t, mask_s = key_padding_mask[0], key_padding_mask[1]

        for blk_idx, blk in enumerate(self.blocks):
            # siamese local attention extract
            blk.t_H, blk.t_W, blk.s_H, blk.s_W = t_H, t_W, s_H, s_W
            attn_search, attn_template = blk(x, None) # [32, 4096, 24], [32, 1024, 24]

            # generate the position encoding and the mask
            s_B, t_B, C = attn_search.size()[0], attn_template.size()[0], attn_search.size()[-1]
            search_feat, template_feat = attn_search.reshape([s_B, s_H, s_W, C]).permute(0, 3, 1, 2), attn_template.reshape([t_B, t_H, t_W, C]).permute(0, 3, 1, 2)
            
            # CNN attention module
            template_feat_cnn, search_feat_cnn = blk.CBAM_template(template_feat), blk.CBAM_search(search_feat)
            template_feat_cnn += template_feat
            search_feat_cnn += search_feat

            # Mutitemps reshape
            template_feat_cnn = template_feat_cnn.reshape(s_B, -1, C, t_H, t_W)
            src_temp, src_search = template_feat_cnn.flatten(3).permute(1, 0, 3, 2).flatten(0,1), search_feat_cnn.flatten(2).permute(0, 2, 1)

            if blk_idx == 0 and self.global_attn:
                src_temp += self.temp_pos_embedding[:, :(t_H*t_W)]
                src_search += self.search_pos_embedding[:, :(s_H*s_W)]
            
            if self.global_attn:
                src_temp = blk.template_global_attn(src_temp)
                src_search = blk.search_global_attn(src_search)

            x = [src_temp, src_search]

        if self.downsample is not None:
            template_down, search_down = self.downsample(src_temp, src_search, t_H, t_W, s_H, s_W)
            if self.conv_ratio < 4:
                s_H, s_W = (s_H + 1) // 2, (s_W + 1) // 2
                t_H, t_W = (t_H + 1) // 2, (t_W + 1) // 2
            # x = torch.cat([template_down, search_down], dim=1)
            x = [template_down, search_down]
            return t_H, t_W, x, s_H, s_W
        else:
            return t_H, t_W, x, s_H, s_W
    
    def extra_repr(self):
        return "dim={}, input_resolution={}, depth={}".format(
            self.dim, self.input_resolution, self.depth)

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ConvEmbed(nn.Module):
    r""" Image to Conv Stem Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, input_size, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=patch_size[0] // 2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size[0] // 2, stride=patch_size[0] // 2)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, [0, self.patch_size[1] - W % self.patch_size[1], 0, 0])
        if H % self.patch_size[0] != 0:
            x = F.pad(x, [0, 0, 0, self.patch_size[0] - H % self.patch_size[0]])

        x = self.stem(x)
        x = self.proj(x)
        if self.norm is not None:
            _, _, Wh, Ww = x.shape
        x = x.flatten(2).permute([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        x = x.permute([0, 2, 1])
        x = x.reshape([-1, self.embed_dim, Wh, Ww])
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        # stem first 3x3 + BN
        flops = (Ho * 2) * (Wo * 2) * self.embed_dim // 2 * self.in_chans * 9
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2
        # stem second 3x3 + BN
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2 * self.embed_dim // 2 * 9
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2
        # stem third 3x3 + BN
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2 * self.embed_dim // 2 * 9
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2
        # proj
        flops += Ho * Wo * self.embed_dim * self.embed_dim // 2 * (
            self.patch_size[0] // 4 * self.patch_size[1] // 4)
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class MixFormer(nn.Module):
    """ A PaddlePaddle impl of MixFormer: 
        `MixFormer: Mixing Features across Windows and Dimensions (CVPR 2022, Oral)`

    Modified from Swin Transformer.
    
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 img_size={'temp_sz':128, 'search_sz':256},
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=8,
                 dwconv_kernel_size=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=True,
                 patch_norm=True,
                 use_checkpoint=False,
                 **kwargs):
        super(MixFormer, self).__init__()
        self.num_classes = num_classes = class_num
        self.num_layers = len(depths)
        if isinstance(embed_dim, int):
            embed_dim = [embed_dim * 2 ** i_layer for i_layer in range(self.num_layers)]
        assert isinstance(embed_dim, list) and len(embed_dim) == self.num_layers
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim[-1])
        self.mlp_ratio = mlp_ratio
        # nomal conv ratio
        self.conv_ratio = [2 ** i_layer for i_layer in range(self.num_layers)] 

        # split image into patches
        self.patch_embed_temp = ConvEmbed(
            input_size=img_size.get('temp_sz'),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches_temp = self.patch_embed_temp.num_patches // (patch_size*self.conv_ratio[-2])
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        self.patch_embed_search = ConvEmbed(
            input_size=img_size.get('search_sz'),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches_search = self.patch_embed_search.num_patches // (patch_size*self.conv_ratio[-2])
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed_temp = nn.Parameter(torch.zeros(1, num_patches_temp, embed_dim[-1]))
            trunc_normal_(self.absolute_pos_embed_temp, std=.02)

            self.absolute_pos_embed_search = nn.Parameter(torch.zeros(1, num_patches_search, embed_dim[-1]))
            trunc_normal_(self.absolute_pos_embed_search, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth decay rule
        pos_type = kwargs.get('position_encoding', 'sine')
        nums_templates = kwargs.get('nums_templates', 1)
        
        self.input_resolution = []
        self.cross_attn = build_featurefusion_network(sm_dim=self.embed_dim[-1], lg_dim=self.embed_dim[-1], \
                            cross_attn_depth=2, cross_attn_heads=num_heads[-1],  cross_attn_dim_head = self.embed_dim[-1] // num_heads[-1], dropout = 0.)
        self.avg = torch.nn.AdaptiveAvgPool1d(1)

        # build layers
        self.layers = nn.ModuleList()
        temp_sz, search_sz = img_size.get('temp_sz'), img_size.get('search_sz')
        for i_layer in range(self.num_layers):
            downsample_ratio = 2 ** i_layer * patch_size if i_layer < self.num_layers - 1 else 2 ** (i_layer-1)*patch_size
            temp_sz_downsample, search_sz_downsample = temp_sz // downsample_ratio, search_sz // downsample_ratio
            input_resolution = {'temp_sz':(temp_sz_downsample, temp_sz_downsample), \
                                'search_sz':(search_sz_downsample, search_sz_downsample)}
                                
            layer = BasicLayer(
                dim=int(self.embed_dim[i_layer]),
                input_resolution=input_resolution,
                conv_ratio = self.conv_ratio[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=ConvMerging
                if (i_layer < self.num_layers - 1) else None,
                out_dim=int(self.embed_dim[i_layer + 1]) 
                if (i_layer < self.num_layers - 1) else 0, 
                pos_type = pos_type,
                nums_templates=nums_templates,
                global_attn=True if i_layer > 1 else False)
            self.layers.append(layer)
            self.input_resolution.append(input_resolution)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, templates, search):
        templates_patch = self.patch_embed_temp(templates)
        search_patch = self.patch_embed_search(search)
        t_B, t_C, t_H, t_W = templates_patch.size()
        s_B, s_C, s_H, s_W = search_patch.size()

        templates_patch = templates_patch.flatten(2).permute(0, 2, 1)
        search_patch = search_patch.flatten(2).permute(0, 2, 1)

        x = [templates_patch, search_patch]

        for layer in self.layers:
            t_H, t_W, x, s_H, s_W = layer(x, t_H, t_W, s_H, s_W)

        # templates_feature, search_feature = torch.split(x, [t_H*t_W, s_H*s_W], dim=1)
        templates_feature, search_feature = x[0], x[1]
        sm_tokens, lg_tokens = templates_feature+self.absolute_pos_embed_temp, search_feature+self.absolute_pos_embed_search

        # sm_tokens = self.SwinTransformer_sm(img_s) # [1, 8*8, 512]
        b, hw, c = sm_tokens.shape[0], sm_tokens.shape[1], sm_tokens.shape[2]

        sm_cls = self.avg(sm_tokens.permute(0,2,1)).permute(0,2,1)
        sm_tokens = self.crop_z_feature(sm_tokens, b, hw, c)
        sm_tokens = torch.cat((sm_cls, sm_tokens), dim=1)

        # lg_tokens = self.SwinTransformer_sm(img_l)  # [1, 16*16, 512]
        lg_cls = self.avg(lg_tokens.permute(0,2,1)).permute(0,2,1)
        lg_tokens = torch.cat((lg_cls, lg_tokens), dim=1)
        sm_tokens, lg_tokens, lg_tokens_fusion = self.cross_attn(sm_tokens, lg_tokens)

        lg_track_final = torch.cat([lg_tokens, lg_tokens_fusion], dim=2)  # [1, 196, 1024]
        lg_track_final_reshape = lg_track_final.permute(1,0,2)# [196=14*14, 1, 512]
        reg_cls_feat = lg_track_final_reshape.unsqueeze(0).transpose(1, 2)

        return reg_cls_feat

    def forward(self, template, search):
        reg_cls_feat = self.forward_features(template, search)
        return reg_cls_feat
    
    def crop_z_feature(self, sm_tokens_lg, b, hw, c):
        h= int(hw**0.5)
        sm_tokens_lg_bhwc = sm_tokens_lg.reshape(b, h, -1, c)
        crop_z_feature = sm_tokens_lg_bhwc[:, 3:10, 3:10, :]
        crop_z_feature = crop_z_feature.reshape(b, -1, c)

        return crop_z_feature

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        # norm
        flops += self.num_features * self.patches_resolution[
            0] * self.patches_resolution[1] // (2**self.num_layers)
        # last proj
        flops += self.num_features * 1280 * self.patches_resolution[
            0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += 1280 * self.num_classes
        return flops

class MixTracking(nn.Module):
    """ This is the base class for Transformer Tracking, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, hidden_dim):
        """ Initializes the model.
        """
        super().__init__()
        num_classes = 1
        hidden_dim = hidden_dim*2
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.backbone = backbone

    def forward(self, search, templates):
       
        bs, n_t, c, h, w = templates.shape
        templates = templates.reshape(bs * n_t, c, h, w)

        reg_cls_feat = self.backbone(templates, search)

        outputs_class = self.class_embed(reg_cls_feat)
        outputs_coord = self.bbox_embed(reg_cls_feat).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out
    
    def track(self, search, templates: list):
        pass


class MixtrackingBackboneType:

    def __init__(self) -> None:
        pass

    def MixFormer_B0(self, pretrained=False, use_ssld=False, **kwargs):
        model = MixFormer(
            embed_dim=24,
            # depths=[1, 2, 6, 6],
            # num_heads=[3, 6, 12, 24],
            # depths=[2, 4, 16],
            depths=[2, 2, 6],
            num_heads=[3, 6, 12],
            drop_path_rate=0.,
            **kwargs)

        return model


    def MixFormer_B1(self, pretrained=False, use_ssld=False, **kwargs):
        model = MixFormer(
            embed_dim=96,
            # depths=[1, 2, 6, 6],
            # num_heads=[2, 4, 8, 16],
            depths=[2, 4, 16],
            num_heads=[3, 6, 12],
            drop_path_rate=0.,
            **kwargs)

        return model

    def MixFormer_B2(self, pretrained=False, use_ssld=False, **kwargs):
        model = MixFormer(
            embed_dim=32,
            depths=[2, 2, 8, 8],
            num_heads=[2, 4, 8, 16],
            drop_path_rate=0.05,
            **kwargs)

        return model

    def MixFormer_B3(self, pretrained=False, use_ssld=False, **kwargs):
        model = MixFormer(
            embed_dim=48,
            depths=[2, 2, 8, 6],
            num_heads=[3, 6, 12, 24],
            drop_path_rate=0.1,
            **kwargs)

        return model

    def MixFormer_B4(self, pretrained=False, use_ssld=False, **kwargs):
        model = MixFormer(
            embed_dim=64,
            depths=[2, 2, 8, 8],
            num_heads=[4, 8, 16, 32],
            drop_path_rate=0.2,
            **kwargs)

        return model

    def MixFormer_B5(self, pretrained=False, use_ssld=False, **kwargs):
        model = MixFormer(
            embed_dim=96,
            depths=[1, 2, 8, 6],
            num_heads=[6, 12, 24, 48],
            drop_path_rate=0.3,
            **kwargs)

        return model

    def MixFormer_B6(self, pretrained=False, use_ssld=False, **kwargs):
        model = MixFormer(
            embed_dim=96,
            depths=[2, 4, 16, 12],
            num_heads=[6, 12, 24, 48],
            drop_path_rate=0.5,
            **kwargs)

        return model

    def MixFormer_Base(self, settings, **kwargs):
        model = MixFormer(
            img_size={'temp_sz': settings.temp_sz, 'search_sz':settings.search_sz},
            embed_dim=settings.embed_dim,
            depths=settings.depths,
            num_heads=settings.num_heads,
            drop_path_rate=settings.drop_path_rate,
            position_encoding_type=settings.position_embedding,
            nums_templates= settings.nums_templates, # multitemps or singletemps
            **kwargs)

        return model

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        # self.iouhead_loss = nn.MSELoss()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"] # torch.Size([bs, 1, 128, 128])

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose() #torch.Size([32, 1, 256, 256])
        target_masks = target_masks.to(src_masks) #torch.Size([bs, 1, 256, 256])

        # upsample predictions to the target size
        src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        #torch.Size([bs, 1, 256, 256])

        src_masks = src_masks[:, 0].flatten(1) #torch.Size([18, 660969])

        target_masks = target_masks[:, 0].flatten(1) #torch.Size([18, 660969])

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, src_masks.shape[0]),
            "loss_dice": dice_loss(src_masks, target_masks, src_masks.shape[0]),
        }
        return losses

    def loss_iouh(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_iouh" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_iouh = outputs['pred_iouh'][idx]
        with torch.no_grad():
            src_boxes = outputs['pred_boxes'][idx]
            # target_boxes = torch.cat([torch.cat([target['boxes']] * 1024, 0) for target in targets], 0)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            giou, iou = box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
            iou = iou.unsqueeze(1)

        # src_iouh.shape
        # torch.Size([3799, 1])
        # iou.shape
        # torch.Size([3799, 1])

        losses = {
            "loss_iouh": self.iouhead_loss(src_iouh, iou),
        }
        return losses

    def iouhead_loss(self, src_iouh, iou):
        loss = torch.mean(((1-iou)**2)*((src_iouh - iou)**2))
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'iouh': self.loss_iouh
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@model_constructor
def SATracker(settings):
    MBT = MixtrackingBackboneType()
    BackboneType = settings.backbone
    MixBackbone = getattr(MBT, BackboneType)
    backbone = MixBackbone(settings)  # backbone without positional encoding and attention mask
    model = MixTracking(
        backbone,
        hidden_dim = settings.hidden_dim
    )
    if settings.iou_head:
        assert settings.masks == False
        model = MixTrackingiouh(model, hidden_dim = settings.hidden_dim, freeze_transt=settings.freeze_transt)
    device = torch.device(settings.device)
    model.to(device)

    return model

# @model_constructor
# def transt_resnet50(settings):
#     num_classes = 1
#     backbone_net = build_backbone(settings, backbone_pretrained=True)
#     featurefusion_network = build_featurefusion_network(settings)
#     model = TransT(
#         backbone_net,
#         featurefusion_network,
#         num_classes=num_classes
#     )
#     if settings.iou_head:
#         assert settings.masks == False
#         model = TransTiouh(model, freeze_transt=settings.freeze_transt)
#     elif settings.masks:
#         assert settings.iou_head == False
#         model = TransTiouh(model, freeze_transt=settings.freeze_transt)
#         model = TransTiouhsegm(model, freeze_transt=settings.freeze_transt)
#     device = torch.device(settings.device)
#     model.to(device)
#     return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    if settings.masks:
        weight_dict["loss_mask"] = 1
        weight_dict["loss_dice"] = 1
    if settings.iou_head:
        weight_dict["loss_iouh"] = 1
    losses = ['labels', 'boxes']
    if settings.masks:
        losses += ["masks"]
    if settings.iou_head:
        losses += ["iouh"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion


