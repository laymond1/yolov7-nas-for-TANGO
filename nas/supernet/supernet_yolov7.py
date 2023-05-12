## libraries in  yolov7 
import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
from yolov7_utils.common import *
from yolov7_utils.experimental import *
from yolov7_utils.autoanchor import check_anchor_order
from yolov7_utils.general import make_divisible, check_file, set_logging
from yolov7_utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from yolov7_utils.loss import SigmoidBin

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

## libraries from OFA code
import copy
import random

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicMBConvLayer,
)
from ofa.utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    MBConvLayer,
    ResidualBlock,
)
from ofa.imagenet_classification.networks import MobileNetV3
from ofa.utils import make_divisible, val2list, MyNetwork

## libraries for type hint
from typing import List, Tuple, Union, Optional, Callable, Any

## Super class for YOLOSuperNet
from nas.supernet.yolo import *

## search_block.py
from .search_block import ELAN, ELANBlock, BBoneELAN, HeadELAN

class YOLOSuperNet(YOLOModel):
    def __init__(
        self,
        cfg='./yaml/yolov7_dynamicsupernet.yml', 
        ch=3, 
        nc=None, 
        anchors=None,
        ks_list=3,
        depth_list=[3,3,3,3,5,5,5,5]
        ):
        
        self.runtime_depth = 0
        
        super(YOLOSuperNet, self).__init__(cfg, ch, nc, anchors)
        
        self.depth_list = self.yaml['depth_list']
        self.set_max_net()
        
    # Override
    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        elan_idx = 0
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m, IKeypoint):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            if isinstance(m, ELAN) and isinstance(self.runtime_depth, list): # 
                depth = self.runtime_depth[elan_idx]
                elan_idx += 1
                x = m(x, d=depth)
            else:
                x = m(x)  # run
            
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def set_max_net(self):
        max_list = lambda x: [max(n) for n in x]
        self.set_active_subnet(d=max_list(self.depth_list))
        

    def set_active_subnet(self, d=None, **kwargs):
        self.runtime_depth = d
        # need to adjust the number of filters in the next layer
        # adjust_width_next_layer()   # 굳이 필요 없음.     
        
    
    def sample_active_subnet(self):       
        # sample depth
        depth_setting = []
        for d_set in self.depth_list:
            d = random.choice(d_set)
            depth_setting.append(d)
        
        self.set_active_subnet(depth_setting) # ex) [3, 2, 3, 1, 4, 4, 1, 3]
        
        return {"d": depth_setting}
    
    def get_active_net_config(self): # self
        idx = 0
        d = deepcopy(self.yaml)
        
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            if 'ELAN'in m:
                args[-1] = self.runtime_depth[idx]
                idx += 1
        
        del d['depth_list']
                
        return d
    
    # TODO : need to be implemented
    def get_active_subnet(self, preserve_weight=True):
        # define subnet
        config = self.get_active_net_config()
        ch = config['ch']
        subnet = YOLOModel(cfg=config, ch=[ch])
        
        if preserve_weight:
            # extract ELANBlock
            elan_idx = 0
            model = deepcopy(self.model)
            for i, m in enumerate(model):
                if isinstance(m, ELAN):
                    depth = self.runtime_depth[elan_idx]
                    act_idx = m.act_idx[depth]
                    model[i] = ELANBlock(m.mode, deepcopy(m.layers[:act_idx+1]), depth)
                    model[i].i, model[i].f, model[i].type, model[i].np = m.i, m.f, m.type, m.np
                    elan_idx += 1
            subnet.model = model
        
        return subnet
    

if __name__ == "__main__":
    profile=False
    device = select_device('0')
    
    # Create model
    supernet = YOLOSuperNet(cfg='./yaml/yolov7_dynamicsupernet.yml').to(device)
    supernet.train()
    sample_depth_setting = supernet.sample_active_subnet()   
    subnet = supernet.get_active_subnet()
    print(sample_depth_setting)
    
    if profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = supernet(img, profile=profile)
        y = subnet(img, profile=profile)
    
    img = torch.rand(1, 3, 640, 640).to(device)
    y = supernet(img, profile=profile)
    y = subnet(img)
        
    sample_config = supernet.get_active_net_config()
    
    
    from yolo import parse_model
    ch = sample_config['ch']
    model, save = parse_model(sample_config, [ch])