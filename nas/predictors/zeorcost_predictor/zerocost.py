"""
This contains implementations of:
synflow, grad_norm, fisher, and grasp, and variants of jacov and snip
based on https://github.com/mohsaied/zero-cost-nas
"""
import torch
import logging
import math
import torch.nn.functional as F

import predictive

logger = logging.getLogger(__name__)

class ZeroCost():
    def __init__(self, method_type="jacov"):
        # available zero-cost method types: 'jacov', 'snip', 'synflow', 'grad_norm', 'fisher', 'grasp'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.method_type = method_type
        self.dataload = "random"
        self.num_imgs_or_batches = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def query(self, graph, dataloader=None, info=None):
        loss_fn = graph.get_loss_fn()

        n_classes = graph.num_classes
        """
        params:
            net_orig:
                neural network
            dataloader:
                a data loader (typically for training data)
            dataload_info:
                a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
            device:
                # GPU/CPU device used
            loss_fn:
                loss function to use within the zero-cost metrics
            measure_names:
                an array of measure names to compute, if left blank, all measures are computed by default
            measures_arr:
                [not used] if the measures are already computed but need to be summarized, pass them here
        """
        score = predictive.find_measures(
                net_orig=graph,                                                     
                dataloader=dataloader,
                dataload_info=(self.dataload, self.num_imgs_or_batches, n_classes),
                device=self.device,
                loss_fn=loss_fn,
                measure_names=[self.method_type],
            )

        if math.isnan(score) or math.isinf(score):
            score = -1e8

        if self.method_type == 'synflow':
            if score == 0.:
                return score

            score = math.log(score) if score > 0 else -math.log(-score)

        return score