import torch
import torch.nn as nn

from yolov7_utils.common import Conv, Concat
from nas.supernet.dynamic_layers.dynamic_op import DynamicConv2d, DynamicBatchNorm2d


class ELAN(nn.Module):
    def __init__(self, c1, c2, k, depth):
        super(ELAN, self).__init__()
        assert c1 % 2 == 0
        self.c2 = c2
        self.depth = depth


# ELANBlock for Backbone
class BBoneELAN(ELAN):
    def __init__(self, c1, c2, k, depth):
        super(BBoneELAN, self).__init__(c1, c2, k, depth)
        assert c1 % 2 == 0 and depth < 5
        
        layers = []
        
        # depth 1
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        # depth 2
        self.cv3 = Conv(c2, c2, k, 1)
        self.cv4 = Conv(c2, c2, k, 1)
        # depth 3
        self.cv5 = Conv(c2, c2, k, 1)
        self.cv6 = Conv(c2, c2, k, 1)
        # depth 4
        self.cv7 = Conv(c2, c2, k, 1)
        self.cv8 = Conv(c2, c2, k, 1)
        
        self.act_idx = [0, 1, 3, 5, 7][:depth+1] 
    
        layers.append(self.cv1)
        layers.append(self.cv2)
        layers.append(self.cv3)
        layers.append(self.cv4)
        layers.append(self.cv5)
        layers.append(self.cv6)
        layers.append(self.cv7)
        layers.append(self.cv8)
        self.layers = nn.Sequential(*layers)
    
    def get_active_net(self):
        raise NotImplementedError
    
    def forward(self, x, d=None):
        outputs = []
        # depth 1
        x1 = self.cv1(x)
        outputs.append(x1)
        x2 = self.cv2(x)    
        outputs.append(x2)
        # depth 2
        x3 = self.cv3(x2)
        outputs.append(x3)
        x4 = self.cv4(x3)
        outputs.append(x4)
        # depth 3
        x5 = self.cv5(x4)
        outputs.append(x5)
        x6 = self.cv6(x5)
        outputs.append(x6)
        # depth 4
        x7 = self.cv7(x6)
        outputs.append(x7)
        x8 = self.cv8(x7)
        outputs.append(x8)
        
        if d is not None:
            return torch.cat([outputs[i] for i in self.act_idx[:d+1]], dim=1)
        return torch.cat([outputs[i] for i in self.act_idx], dim=1)


# ELANBlock for Head
# there are differences about cardinality(path) and channel size
class HeadELAN(ELAN):
    def __init__(self, c1, c2, k, depth):
        super(HeadELAN, self).__init__(c1, c2, k, depth)
        assert c1 % 2 == 0 and c2 % 2 == 0 and depth < 6
        c_ = int(c2 / 2)
        self.c2 = c2
        self.depth = depth
        
        # depth 1
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        # depth 2
        self.cv3 = Conv(c2, c_, k, 1)
        # depth 3
        self.cv4 = Conv(c_, c_, k, 1)
        # depth 4
        self.cv5 = Conv(c_, c_, k, 1)
        # depth 5
        self.cv6 = Conv(c_, c_, k, 1)
        
        self.act_idx = [0, 1, 2, 3, 4, 5, 6][:depth+1] 
    
    def forward(self, x, d=None):
        outputs = []
        # depth 1
        x1 = self.cv1(x)
        outputs.append(x1)
        x2 = self.cv2(x)    
        outputs.append(x2)
        # depth 2
        x3 = self.cv3(x2)
        outputs.append(x3)
        # depth 3
        x4 = self.cv4(x3)
        outputs.append(x4)
        # depth 4
        x5 = self.cv5(x4)
        outputs.append(x5)
        # depth 5
        x6 = self.cv6(x5)
        outputs.append(x6)
        
        if d is not None:
            return torch.cat([outputs[i] for i in self.act_idx[:d+1]], dim=1)
        return torch.cat([outputs[i] for i in self.act_idx], dim=1)
    
    
# Dynamic Convolution for elastic channel size
class DyConv(nn.Module):
    # Dynamic Convolution for elastic channel size
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride
        super(DyConv, self).__init__()
        self.conv = DynamicConv2d(c1, c2, k, s) # auto same padding
        self.bn = DynamicBatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))