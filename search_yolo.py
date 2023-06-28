import argparse
import os
import time
import yaml

import torch

from yolov7_utils.experimental import attempt_load
from yolov7_utils.torch_utils import select_device

from nas.search_algorithm import EvolutionFinder
from nas.supernet.supernet_yolov7 import YOLOSuperNet
from nas.predictors.efficiency_predictor import LatencyPredictor
from nas.predictors.accuracy_predictor import AccuracyCalculator

def parse_args():
    parser = argparse.ArgumentParser("autonn_supernet_nas")
    # hyp params for evolution search
    parser.add_argument("--pop-size", default=10, type=int, help="population size")
    parser.add_argument("--num-generations", default=500, type=int, help="number of generations")
    parser.add_argument("--parent-ratio", default=0.25, type=float, help="parent ratio")
    parser.add_argument("--mutate-prob", default=0.1, type=float, help="mutate probability")
    parser.add_argument("--mutation-ratio", default=0.5, type=float, help="mutation ratio")
    # hyp params for search type
    parser.add_argument("--constraint-type", default="flops", type=str, help="constraint type")
    parser.add_argument("--flops", default=600, type=float, help="flops")
    parser.add_argument("--efficiency-predictor", default=None, type=str, help="efficiency predictor")
    # hyp parmas for accuracy predictor
    parser.add_argument("--accuracy-predictor", default=None, type=str, help="accuracy predictor")
    parser.add_argument("--weights", default="yolov7_supernetv2.pt", type=str, help="weights path")
    parser.add_argument('--cfg', type=str, default='./yaml/yolov7_supernet.yml', help='model.yaml path')
    parser.add_argument('--data', default='./yaml/data/coco.yaml', type=str, help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='./yaml/data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--save-dir', type=str, default='runs/finetune', help='directory to save results')
    parser.add_argument('--fintune_epochs', type=int, default=5, help='number of finetuning epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    return opt


def run_search(opt):
    constraint_type, efficiency_constraint, accuracy_predictor = \
        opt.constraint_type, opt.flops, opt.accuracy_predictor
    
    efficiency_predictor = LatencyPredictor(target="galaxy10")
    
    device = select_device(opt.device)
    
    # Create model
    supernet = attempt_load(opt.weights, map_location=device)
    
    # opt parameters for fintuning
    accuracy_predictor = AccuracyCalculator(opt, supernet)

    # build the evolution finder
    finder = EvolutionFinder(
        constraint_type=constraint_type, 
        efficiency_constraint=efficiency_constraint, 
        efficiency_predictor=efficiency_predictor, 
        accuracy_predictor=accuracy_predictor
    )

    # start searching
    result_list = []
    for flops in [opt.flops]:
        st = time.time()
        finder.set_efficiency_constraint(flops)
        best_valids, best_info = finder.run_evolution_search()
        print('-----------------------------------------------')
        print(best_info)
        print('-----------------------------------------------')
        ed = time.time()
        # print('Found best architecture at flops <= %.2f M in %.2f seconds! It achieves %.2f%s predicted accuracy with %.2f MFLOPs.' % (flops, ed-st, best_info[0] * 100, '%', best_info[-1]))
        result_list.append(best_info)
        
    # # save model into yaml
    # print('save model into yaml')
    # for i, result in enumerate(result_list):
    #     best_depth = result[1]['d']
    #     # Active subet
    #     supernet.set_active_subnet(best_depth)
    #     sample_config = supernet.get_active_net_config()
    #     # Create yaml file
    #     yaml_file = './yaml/yolov7_searched_%d.yml' % i
    #     if not os.path.exists(os.path.dirname(yaml_file)):
    #         os.makedirs(os.path.dirname(yaml_file))
    #     with open(yaml_file, 'w') as f:
    #         yaml.dump(sample_config, f, sort_keys=False)
    #     print(f'# Depth: {best_depth} | Saved best {i} model\'s config in {yaml_file}')

    
if __name__ == "__main__":
    opt = parse_args()
    run_search(opt)