import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def b_arch_to_feat(arch):
    # This function converts a backbone config to a feature vector (20-D).
    d_list = copy.deepcopy(arch['d'])

    # convert to onehot -> 1~5의 scale에선 학습 잘 안됨 (normalization과 비슷)
    # 5*4 = 20-D feature vector
    onehot = [0 for _ in range(20)]

    for i in range(4):
        tidx = int(d_list[i]) - 1
        onehot[i*5 + tidx] = 1

    return torch.Tensor(onehot)

def h_arch_to_feat(arch):
    # This function converts a backbone config to a feature vector (20-D).
    d_list = copy.deepcopy(arch['d'])

    # 7*4 = 28-D feature vector
    onehot = [0 for _ in range(28)]

    for i in range(4):
        tidx = int(d_list[i]) - 1
        onehot[i*7 + tidx] = 1

    return torch.Tensor(onehot)


class Net(nn.Module):
    """
    The base model for MAML (Meta-SGD) for meta-NAS-predictor.
    """

    def __init__(self, nfeat, hw_embed_on, hw_embed_dim, layer_size):
        super(Net, self).__init__()
        self.layer_size = layer_size
        self.hw_embed_on = hw_embed_on

        self.add_module('fc1', nn.Linear(nfeat, layer_size))
        self.add_module('fc2', nn.Linear(layer_size, layer_size))

        if hw_embed_on:
            self.add_module('fc_hw1', nn.Linear(hw_embed_dim, layer_size))
            self.add_module('fc_hw2', nn.Linear(layer_size, layer_size))
            hfeat = layer_size * 2 
        else:
            hfeat = layer_size

        self.add_module('fc3', nn.Linear(hfeat, hfeat))
        self.add_module('fc4', nn.Linear(hfeat, hfeat))

        self.add_module('fc5', nn.Linear(hfeat, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, hw_embed=None, params=None):
        # hw_embed = hw_embed.repeat(len(x), 1)
        if params == None:
            out = self.relu(self.fc1(x))
            out = self.relu(self.fc2(out))

            if self.hw_embed_on:
                hw = self.relu(self.fc_hw1(hw_embed))
                hw = self.relu(self.fc_hw2(hw))
                out = torch.cat([out, hw], dim=-1)

            out = self.relu(self.fc3(out))
            out = self.relu(self.fc4(out))
            out = self.fc5(out)

        else:
            out = F.relu(F.linear(x, params['meta_learner.fc1.weight'],
                                params['meta_learner.fc1.bias']))
            out = F.relu(F.linear(out, params['meta_learner.fc2.weight'],
                                params['meta_learner.fc2.bias']))
            
            if self.hw_embed_on:
                hw = F.relu(F.linear(hw_embed, params['meta_learner.fc_hw1.weight'],
                                    params['meta_learner.fc_hw1.bias']))
                hw = F.relu(F.linear(hw, params['meta_learner.fc_hw2.weight'],
                                    params['meta_learner.fc_hw2.bias']))
                out = torch.cat([out, hw], dim=-1)

            out = F.relu(F.linear(out, params['meta_learner.fc3.weight'],
                                params['meta_learner.fc3.bias']))
            out = F.relu(F.linear(out, params['meta_learner.fc4.weight'],
                                params['meta_learner.fc4.bias']))
            out = F.linear(out, params['meta_learner.fc5.weight'],
                                params['meta_learner.fc5.bias']) 

        return out