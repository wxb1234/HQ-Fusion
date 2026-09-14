"""by lyuwenyu
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'backbone1', 'encoder', 'encoder2','decoder', ]

    def __init__(self, backbone: nn.Module, backbone1, encoder, encoder2, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.backbone1 = backbone1
        self.decoder = decoder
        self.encoder = encoder
        self.encoder2 = encoder2
        self.multi_scale = multi_scale


    def forward(self, vis, ir, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            vis = F.interpolate(vis, size=[sz, sz])
            ir = F.interpolate(ir, size=[sz, sz])

        viss = self.backbone(vis)
        irs = self.backbone1(ir)

        irs = self.encoder(irs)
        viss = self.encoder2(viss)

        f = []
        for _, (i,v) in enumerate(zip(irs, viss)):
            f.append(i+v)

        x = self.decoder([f, irs, viss], targets)

        return x



    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
