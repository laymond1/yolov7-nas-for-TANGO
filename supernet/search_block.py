import torch
import torch.nn as nn

from yolov7.models.common import Conv, Concat


# ELANBlock for Backbone
class BBoneELANBlock(nn.Module):
    def __init__(self, c1, k, depth):
        super(BBoneELANBlock, self).__init__()
        assert c1 % 2 == 0 and depth < 5
        c_ = int(c1 / 2)
        
        self.depth = depth
        
        # depth 1
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # depth 2
        self.cv3 = Conv(c_, c_, k, 1)
        self.cv4 = Conv(c_, c_, k, 1)
        # depth 3
        self.cv5 = Conv(c_, c_, k, 1)
        self.cv6 = Conv(c_, c_, k, 1)
        # depth 4
        self.cv7 = Conv(c_, c_, k, 1)
        self.cv8 = Conv(c_, c_, k, 1)
        
        self.act_idx = [0, 1, 3, 5, 7][:depth+1] 
    
    def forward(self, x):
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
        
        return torch.cat([outputs[i] for i in self.act_idx], dim=1)


# ELANBlock for Head
# there are differences with BBoneELANBlock about cardinality(path) and channel size
class HEADELANBlock(nn.Module):
    def __init__(self, c1, k, depth):
        super(HEADELANBlock, self).__init__()
        assert c1 % 2 == 0 and depth < 6
        c_ = int(c1 / 2)
        c_2 = int(c_ / 2)
        self.depth = depth
        
        # depth 1
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # depth 2
        self.cv3 = Conv(c_, c_2, k, 1)
        # depth 3
        self.cv4 = Conv(c_2, c_2, k, 1)
        # depth 4
        self.cv5 = Conv(c_2, c_2, k, 1)
        # depth 5
        self.cv6 = Conv(c_2, c_2, k, 1)
        
        self.act_idx = [0, 1, 2, 3, 4, 5, 6][:depth+1] 
    
    def forward(self, x):
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
        
        return torch.cat([outputs[i] for i in self.act_idx], dim=1)