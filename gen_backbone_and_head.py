from nas.supernet.supernet_yolov7 import YOLOSuperNet
from latency_predictor.arch_utils import *
from os.path import join
from tqdm.auto import tqdm

device = "cpu"
supernet = YOLOSuperNet(cfg="./yaml/yolov7_dynamicsupernet.yml").to(device)

model_out_dir = "./exported_models"

BACKBONE_MAX_DEPTH = 6  # should be 6

# cal. latency for backbone
# progress_bar = tqdm(total = (BACKBONE_MAX_DEPTH-1)**4)
# for i in range(1, BACKBONE_MAX_DEPTH):
#     for j in range(1, BACKBONE_MAX_DEPTH):
#         for k in range(1, BACKBONE_MAX_DEPTH):
#             for m in range(1, BACKBONE_MAX_DEPTH):
#                 supernet.set_active_subnet([i,j,k,m,1,1,1,1]) # Maximum: [5,5,5,5,7,7,7,7]
#                 subnet = supernet.get_active_subnet()

#                 b_arch, _, save = split_backbone_head(subnet)

#                 # .pt 저장
#                 backbone = skin_backbone(b_arch, save)
#                 torch.save(backbone, join(model_out_dir, f'backbone_{i}_{j}_{k}_{m}.pt'))
#                 progress_bar.update(1)

HEAD_MAX_DEPTH = 8  # should be 8

# cal. latency for head
progress_bar = tqdm(total=(HEAD_MAX_DEPTH - 1) ** 4)
for i in range(1, HEAD_MAX_DEPTH):
    for j in range(1, HEAD_MAX_DEPTH):
        for k in range(1, HEAD_MAX_DEPTH):
            for m in range(1, HEAD_MAX_DEPTH):
                supernet.set_active_subnet(
                    [1, 1, 1, 1, i, j, k, m]
                )  # Maximum: [5,5,5,5,7,7,7,7]
                subnet = supernet.get_active_subnet()

                _, h_arch, save = split_backbone_head(subnet)

                # .pt 저장
                head = skin_head(h_arch, save)
                torch.save(head, join(model_out_dir, f"head_{i}_{j}_{k}_{m}.pt"))
                progress_bar.update(1)
