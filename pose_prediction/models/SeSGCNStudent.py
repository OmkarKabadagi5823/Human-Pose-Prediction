from platform import java_ver
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import math

class DWSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(DWSepConv, self).__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, k*in_channels, kernel_size=1, padding=0, groups=in_channels),
            nn.BatchNorm2d(k*in_channels),
            nn.Dropout(0.1, inplace=True)
        )

        self.relu6 = nn.ReLU6()

        self.pointwise = nn.Conv2d(k*in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu6(x)
        x = self.pointwise(x)

        return x

class STSepConv(nn.Module):
    def __init__(self, time_dim, vertices_dim):
        super(STSepConv, self).__init__()

        self.V = nn.Parameter(torch.FloatTensor(time_dim, vertices_dim, vertices_dim))
        stdv = 1.0 / math.sqrt(self.V.size(1))
        self.V.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(vertices_dim, time_dim, time_dim))
        stdv = 1.0 / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

    def forward(self, x, Vmask, Tmask):
        Vmasked = Vmask * self.V
        Tmasked = Tmask * self.T

        x = torch.einsum('nctv,vtq->ncqv', (x, Vmasked))
        x = torch.einsum('nctv,tvw->nctw', (x, Tmasked))

        return x.contiguous()

class STDWSepGCN(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size,
        stride,
        time_dim, vertices_dim,
        dropout,
        bias=True
        ):
        super(STDWSepGCN, self).__init__()

        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1

        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.stsepconv = STSepConv(time_dim, vertices_dim)
        self.dwsepconv = DWSepConv(in_channels, out_channels, k=1)

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x, Vmask, Tmask):
        x = self.stsepconv(x, Vmask, Tmask)
        x = self.dwsepconv(x)
        x = x + self.residual(x)
        x = self.prelu(x)

        return x

class TCN(nn.Module):
    def __init__(
        self, 
        in_channels, out_channels, 
        kernel_size, 
        dropout, 
        bias=True
        ):
        super(TCN, self).__init__()

        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1
        
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class SeSGCN(nn.Module):
    def __init__(
        self, 
        in_channels, 
        in_frames, out_frames, 
        stdw_sep_gcn_dropout, 
        vertices_dim,
        tcn_layers, tcn_kernel_size, tcn_dropout):
        super(SeSGCN, self).__init__()

        self.stdw_sep_gcns = nn.ModuleList()
        self.tcns = nn.ModuleList()

        self.prelus = nn.ModuleList()

        # Encoder
        self.stdw_sep_gcns.append(STDWSepGCN(in_channels, 66, [1, 1], 1, in_frames, vertices_dim, stdw_sep_gcn_dropout))
        self.stdw_sep_gcns.append(STDWSepGCN(66, 66, [1, 1], 1, in_frames, vertices_dim, stdw_sep_gcn_dropout))
        self.stdw_sep_gcns.append(STDWSepGCN(66, 66, [1, 1], 1, in_frames, vertices_dim, stdw_sep_gcn_dropout))
        self.stdw_sep_gcns.append(STDWSepGCN(66, 66, [1, 1], 1, in_frames, vertices_dim, stdw_sep_gcn_dropout))
        self.stdw_sep_gcns.append(STDWSepGCN(66, in_channels, [1, 1], 1, in_frames, vertices_dim, stdw_sep_gcn_dropout))

        # Decoder
        self.tcns.append(TCN(in_frames, out_frames, tcn_kernel_size, tcn_dropout))

        for _ in range(1, tcn_layers):
            self.tcns.append(TCN(out_frames, out_frames, tcn_kernel_size, tcn_dropout))

        for _ in range(tcn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x, Vmask, Tmask):
        for layer, stdw_dep_gcn in enumerate(self.stdw_sep_gcns):
            x = stdw_dep_gcn(x, Vmask[layer], Tmask[layer])

        x = x.permute(0, 2, 1, 3)

        for layer, (tcn, prelu) in enumerate(zip(self.tcns, self.prelus)):
            if layer == 0:
                x = prelu(tcn(x))
            else:
                x = prelu(tcn(x)) + x
