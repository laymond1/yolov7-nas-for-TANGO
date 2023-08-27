import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import Net
from utils import b_arch_to_feat, h_arch_to_feat
from sklearn.metrics import mean_squared_log_error

target = "galaxy_s22_gpu"
pred_target = "head"
seed = 10
epoches = 5000
batch_size = 128

file_path = os.path.join("./latency/{}_{}.json".format(target, pred_target))

with open(file_path, "r") as f:
    lut = json.load(f)
d_list = list(lut.keys())

if pred_target == "backbone":
    nfeat = 20
elif pred_target == "head":
    nfeat = 28

x = []
y = []
for d in d_list:
    arch = {'d':d}
    if pred_target == "backbone":
        x.append(b_arch_to_feat(arch))
    elif pred_target == "head":
        x.append(h_arch_to_feat(arch))
    y.append(torch.as_tensor(lut[d]))

x = torch.stack(x,dim=0)
y = torch.as_tensor(y)

dataset = TensorDataset(x, y)

# performance test
#train_set, val_set = torch.utils.data.random_split(dataset, [0.95, 0.05])
#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train(train_queue, model, criterion, optimizer):
    model.train() # mode change

    for step, (x_train, y_train) in enumerate(train_queue):
        
        n = x_train.size(0)
        x_train = Variable(x_train).cuda().float()
        y_train = Variable(y_train, requires_grad=False).cuda().float() 

        optimizer.zero_grad()
        logits = model(x_train).squeeze()
        loss = criterion(logits, y_train)

        loss.backward() # weight compute
        optimizer.step() # weight update

def infer(eval_queue, model):
    model.eval()

    y_real= []
    preds = []
    with torch.no_grad():
        for step, (x, y) in enumerate(eval_queue):
            n = x.size(0)
            x = Variable(x).cuda().float()
            y = Variable(y).cuda().float()
            
            logits = model(x).squeeze()
            preds.extend(logits.cpu().numpy())
            y_real.extend(y.cpu().numpy())
    

    MSLE = mean_squared_log_error(y_real, preds)
    RMSLE = np.sqrt(MSLE)

    return RMSLE

np.random.seed(seed)
torch.cuda.set_device(0)
torch.manual_seed(seed)
cudnn.benchmark = True 
torch.cuda.manual_seed(seed)

criterion = nn.MSELoss().cuda()

net = Net(
    nfeat=nfeat,
    hw_embed_on=False,
    hw_embed_dim=0,
    layer_size=64
).cuda()
net.apply(weight_init)

optimizer = torch.optim.Adam(
    net.parameters(),
    lr=1e-4,
    betas=(0.9,0.999),
    weight_decay=1e-4,
    eps=1e-08
)

for epoch in range(epoches):
    
    # training
    train(train_loader, net, criterion, optimizer)

    # evaluating
    if (epoch+1) % 100 == 0:
        RMSLE = infer(train_loader, net)
        print('training... ', epoch+1)
        print(RMSLE)


torch.save(net.state_dict(), os.path.join("trained_pred/{}_{}.pt".format(target,pred_target)))