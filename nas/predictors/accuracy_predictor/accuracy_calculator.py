'''
The currently written version uses real evluation MAP calculated for test dataset.
Accuracy predictor will be added to increase the search efficiency.
'''

import os
import yaml
from test import test  # import test.py to get mAP for each subnet
from tqdm import tqdm

from yolov7_utils.datasets import create_dataloader
from yolov7_utils.general import colorstr, check_img_size


class AccuracyCalculator():
    def __init__(
        self, 
        opt,
        supernet,

    ):
        self.opt = opt
        self.supernet = supernet

        # Set DDP variables
        opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

        # Hyperparameters
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        # load data yaml
        with open(opt.data, encoding="UTF-8") as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        self.is_coco = opt.data.endswith('coco.yaml') or opt.data.endswith('coco128.yaml')
        # Image sizes
        gs = max(int(supernet.stride.max()), 32)  # grid size (max stride)
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
        self.imgsz_test = imgsz_test
        # prepare test dataset
        test_path = data_dict['val']
        self.testloader = create_dataloader(test_path, imgsz_test, opt.batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

    def predict_accuracy(self, subnet_list):
        acc_list = []
        # subnet_list: list of subnets
        for subnet in subnet_list:
            self.supernet.set_active_subnet(subnet['d'])
            
            # Calculate mAP
            results, _, _ = test(self.opt.data,
                                batch_size=self.opt.batch_size * 2,
                                imgsz=self.imgsz_test,
                                conf_thres=0.001,
                                iou_thres=0.7,
                                model=self.supernet,
                                single_cls=self.opt.single_cls,
                                dataloader=self.testloader,
                                # save_dir=save_dir,
                                save_json=False,
                                plots=False,
                                is_coco=self.is_coco,
                                v5_metric=self.opt.v5_metric)
            
            # mp, mr, map50, map, avg_loss = results
            map= results[3]
            acc_list.append(map)

        return acc_list
    
    def predict_accuracy_once(self, subnet):
        # activate the subnet
        self.supernet.set_active_subnet(subnet['d'])
            
        # Calculate mAP
        results, _, _ = test(self.opt.data,
                            batch_size=self.opt.batch_size * 2,
                            imgsz=self.imgsz_test,
                            conf_thres=0.001,
                            iou_thres=0.7,
                            model=self.supernet,
                            single_cls=self.opt.single_cls,
                            dataloader=self.testloader,
                            # save_dir=save_dir,
                            save_json=False,
                            plots=False,
                            is_coco=self.is_coco,
                            v5_metric=self.opt.v5_metric)
            
        # mp, mr, map50, map, avg_loss = results
        map = results[3]
        return map