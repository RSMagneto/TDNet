import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)

    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)

    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings

def extract_image_patches(images, ksizes, strides, rates=[1, 1]):
    assert len(images.size()) == 4
    images, paddings = same_padding(images, ksizes, strides, rates)
    unfold = torch.nn.Unfold(kernel_size=ksizes, padding=0, stride=strides)
    patches = unfold(images)
    return patches, paddings

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
        )

def conv1x1(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes): 
        super(BasicBlock, self).__init__()
        midplanes = int(inplanes//2)
        self.conv1 = conv3x3(inplanes, midplanes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv1x1(midplanes, planes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

class SPA(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=4, in_channels=6, softmax_scale=10):
        super(SPA, self).__init__()
        self.ksize = ksize
        self.in_channels = in_channels
        self.softmax_scale = softmax_scale
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.conv = BasicBlock(in_channels, in_channels)
        self.out = BasicBlock(in_channels, in_channels)

    def forward(self, x):
        x = self.conv(x)
        residual = x
        raw_int_bs = list(x.size())
        w, h = raw_int_bs[2], raw_int_bs[3]
        patches, paddings = extract_image_patches(x, ksizes=[self.ksize, self.ksize],
                                                    strides=[self.stride_1, self.stride_1])
        patches1 = patches.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patches2 = patches1.permute(0, 4, 1, 2, 3)
        patches_group_C = torch.split(patches2, 1, dim=0)
        patches3 = patches.permute(0, 2, 1)
        patches_group_hwC = torch.split(patches3, 1, dim=0)
        y_hwC = []
        y_C = []
        for x_C, x_hwC in zip(patches_group_C, patches_group_hwC):
            c_s = x_C.shape[2]
            k_s = x_C[0].shape[2]

            xi_hwC_q = x_hwC.view(x_hwC.shape[1], -1)
            xi_hwC_k = x_hwC.view(x_hwC.shape[1], -1).permute(1, 0)
            score_map_hwC = torch.matmul(xi_hwC_q, xi_hwC_k)
            score_map_hwC = score_map_hwC.view(1, score_map_hwC.shape[0], math.ceil(w / self.stride_2),
                                       math.ceil(h / self.stride_2))
            b_s, l_s, h_s, w_s = score_map_hwC.shape
            score_map_hwC = score_map_hwC.view(l_s, -1)
            score_map_hwC = F.softmax(score_map_hwC * self.softmax_scale, dim=1)
            yi_hwC = torch.mm(score_map_hwC, xi_hwC_q)
            yi_hwC = yi_hwC.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi_hwC = yi_hwC.view(1, l_s, -1).permute(0, 2, 1)
            zi_hwC = torch.nn.functional.fold(zi_hwC, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize),
                                          padding=paddings[0], stride=self.stride_1)
            inp_hwC = torch.ones_like(zi_hwC)
            inp_unf_hwC = torch.nn.functional.unfold(inp_hwC, (self.ksize, self.ksize), padding=paddings[0],
                                                 stride=self.stride_1)
            out_mask_hwC = torch.nn.functional.fold(inp_unf_hwC, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize),
                                                padding=paddings[0], stride=self.stride_1)
            zi_hwC = zi_hwC / out_mask_hwC
            y_hwC.append(zi_hwC)

            x_C = x_C.view(l_s, c_s, -1)
            xi_C_k = x_C
            xi_C_q =x_C.permute(0, 2, 1)
            score_map_C = torch.matmul(xi_C_q, xi_C_k)

            score_map_C = F.softmax(score_map_C * self.softmax_scale, dim=-1)
            yi_C = torch.bmm(score_map_C, xi_C_q)
            yi_C = yi_C.view(l_s, -1)
            zi_C = yi_C.view(1, l_s, -1).permute(0, 2, 1)
            zi_C = torch.nn.functional.fold(zi_C, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize),
                                              padding=paddings[0], stride=self.stride_1)
            inp_C = torch.ones_like(zi_C)
            inp_unf_C = torch.nn.functional.unfold(inp_C, (self.ksize, self.ksize), padding=paddings[0],
                                                     stride=self.stride_1)
            out_mask_C = torch.nn.functional.fold(inp_unf_C, (raw_int_bs[2], raw_int_bs[3]),
                                                    (self.ksize, self.ksize),
                                                    padding=paddings[0], stride=self.stride_1)
            zi_C = zi_C / out_mask_C
            y_C.append(zi_C)

        out = torch.cat(y_hwC, dim=0)
        spa_map = torch.cat(y_C, dim=0)

        out = residual + out + spa_map
        return out

class Branch_SPA(nn.Module):
    def __init__(self, ksize=8, stride1=4, stride2=4, embed_dim=32):
        super(Branch_SPA, self).__init__()
        self.spa1 = SPA(ksize=ksize, stride_1=stride1, stride_2=stride2, in_channels=embed_dim )
        self.spa2 = SPA(ksize=ksize, stride_1=stride1, stride_2=stride2, in_channels=embed_dim )
        self.conv = BasicBlock(embed_dim, embed_dim)

    def forward(self, x):
        out1 = self.spa1(x)
        out2 = self.spa2(out1)
        out = self.conv(out2)
        return out

class SPE(nn.Module):
    def __init__(self,in_channels, ratio, softmax_scale=10):
        super(SPE, self).__init__()
        self.in_channels = in_channels

        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels//ratio, in_channels)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels // ratio, in_channels)
        )
        self.conv = BasicBlock(in_channels, in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.softmax_scale = softmax_scale

    def forward(self, x):
        x = self.conv(x)
        residual = x
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        channels = self.pool(x).view(b, c, -1)
        channels = channels.permute(0, 2, 1)
        channnels_group = torch.split(channels, 1, dim=0)
        bn = x.view(b, c, h * w)
        bn_group = torch.split(bn, 1, dim=0)
        y = []

        for xi, bn in zip(channnels_group, bn_group):
            xi = xi.view(xi.shape[1], -1)
            bn = bn.view(bn.shape[1], -1)
            q = self.mlp1(xi).permute(1, 0)
            k = self.mlp1(xi)
            score = torch.matmul(q,k)
            score = F.softmax(score, dim=1)
            out = torch.mm(score, bn)
            out = out.view(1, c, h, w)

            y.append(out)
        y = torch.cat(y, dim=0)
        y = residual+y
        return y

class Branch_SPE(nn.Module):
    def __init__(self, in_channels, ratio):
        super(Branch_SPE, self).__init__()
        self.spe1 = SPE(in_channels, ratio)
        self.spe2 = SPE(in_channels, ratio)
        self.conv1 = BasicBlock(in_channels, in_channels)
        self.conv2 = BasicBlock(in_channels, in_channels)
        self.convtrans1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.convtrans2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.spe1(x)
        out2 = self.conv1(out1)
        out3 = self.convtrans1(out2)
        out4 = self.spe2(out3)
        out5 = self.convtrans1(out4)
        out = self.conv2(out5)
        return out

class Branch_Com(nn.Module):
    def __init__(self, in_channels):
        super(Branch_Com, self).__init__()
        self.conv1 = BasicBlock(in_channels, in_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = BasicBlock(in_channels, in_channels)
        self.conv3 = BasicBlock(in_channels, in_channels)
        self.conv4 = BasicBlock(in_channels, in_channels)
        self.convtrans = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.pool(out1)
        out3 = self.conv2(out2)
        out4 = self.convtrans(out3)
        out5 = self.conv3(out4)
        out = self.conv4(out5)
        return out

class model(nn.Module):
    def __init__(self, ksize=16, stride1=8, stride2=8, embed_dim=32, ratio=4):
        super(model, self).__init__()

        self.fusionBlock = Branch_Com(embed_dim)
        self.branch_spa = Branch_SPA(ksize=ksize, stride1=stride1, stride2=stride2, embed_dim=embed_dim)
        self.branch_spe = Branch_SPE(in_channels=embed_dim, ratio=ratio)

        self.fusionBlock1 = nn.Sequential(
            BasicBlock(embed_dim*2, embed_dim),
            BasicBlock(embed_dim, 1)
        )

        self.fusionBlock2 = nn.Sequential(
            BasicBlock(embed_dim * 2, embed_dim),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            BasicBlock(embed_dim, embed_dim),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            BasicBlock(embed_dim , embed_dim),
            BasicBlock(embed_dim, 4)
        )

        self.fusionBlock3 = nn.Sequential(
            BasicBlock(embed_dim * 3, embed_dim * 2),
            BasicBlock(embed_dim * 2, embed_dim),
            BasicBlock(embed_dim, 4)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=5, out_channels=embed_dim, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=4, out_channels=embed_dim, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, ms,pan):

        ms_up = torch.nn.functional.interpolate(ms, size=(pan.shape[2], pan.shape[2]), mode='bilinear')
        out1 = torch.cat([ms_up, pan], 1)
        out2 = self.conv1(out1)
        out = self.fusionBlock(out2)

        ms1 = self.conv2(ms)
        ms_out = self.branch_spe(ms1)

        pan1 = self.conv3(pan)
        pan_out=self.branch_spa(pan1)

        panout1 = torch.cat([out, pan_out], 1)
        panout=self.fusionBlock1(panout1)

        msout1 = torch.cat([out, ms_out], 1)
        msout = self.fusionBlock2(msout1)

        allout1 = torch.cat([out, ms_out, pan_out], 1)
        allout = self.fusionBlock3(allout1)


        return msout, panout, allout,ms_out,pan_out,out

