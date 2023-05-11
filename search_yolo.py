import argparse
import time
import yaml

from nas.search_algorithm import EvolutionFinder
from nas.supernet.supernet_yolov7 import YOLOSuperNet


def parse_args():
    parser = argparse.ArgumentParser("autonn_supernet_nas")
    # hyp params for evolution search
    parser.add_argument("--pop-size", default=100, type=int, help="population size")
    parser.add_argument("--num-generations", default=500, type=int, help="number of generations")
    parser.add_argument("--parent-ratio", default=0.25, type=float, help="parent ratio")
    parser.add_argument("--mutate-prob", default=0.1, type=float, help="mutate probability")
    parser.add_argument("--mutation-ratio", default=0.5, type=float, help="mutation ratio")
    # hyp params for search type
    parser.add_argument("--constraint-type", default="flops", type=str, help="constraint type")
    parser.add_argument("--flops", default=600, type=float, help="flops")
    parser.add_argument("--efficiency-predictor", default=None, type=str, help="efficiency predictor")
    parser.add_argument("--accuracy-predictor", default=None, type=str, help="accuracy predictor")
    args = parser.parse_args()
    return args


def run_search(args):
    constraint_type, efficiency_constraint, efficiency_predictor, accuracy_predictor = \
        args.constraint_type, args.flops, args.efficiency_predictor, args.accuracy_predictor
    
    # build the evolution finder
    finder = EvolutionFinder(
        constraint_type=constraint_type, 
        efficiency_constraint=efficiency_constraint, 
        efficiency_predictor=efficiency_predictor, 
        accuracy_predictor=accuracy_predictor
    )

    # start searching
    result_list = []
    for flops in [600, 400, 350]:
        st = time.time()
        finder.set_efficiency_constraint(flops)
        best_valids, best_info = finder.run_evolution_search()
        ed = time.time()
        # print('Found best architecture at flops <= %.2f M in %.2f seconds! It achieves %.2f%s predicted accuracy with %.2f MFLOPs.' % (flops, ed-st, best_info[0] * 100, '%', best_info[-1]))
        result_list.append(best_info)
        
    # save model into yaml
    for i, result in enumerate(result_list):
        best_depth = result[1]['d']
        # Create model
        supernet = YOLOSuperNet(cfg='./yaml/yolov7_supernet.yml')
        supernet.set_active_subnet(best_depth)
        sample_config = supernet.get_active_net_config()
        # Create yaml file
        yaml_file = './yaml/yolov7_searched_%d.yml' % i
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_config, f, sort_keys=False)
        print(f'# Saved best {i} model config in {yaml_file}')

    
if __name__ == "__main__":
    args = parse_args()
    run_search(args)