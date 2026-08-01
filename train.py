# -*- encoding: utf-8 -*-
'''
@File         :train.py
@Date         :2026/06/25 20:12:25
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
torch.multiprocessing.set_sharing_strategy('file_system')



from tensorboardX import SummaryWriter


def main(model,dataset,configs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default = -1, type=int)
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = args.local_rank

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        device = configs.device #get("device", "cuda")
        rank = 0
        world_size = 1
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )

    num_gpus = torch.cuda.device_count()

    dataloader,sampler = build_dataloader(configs,dataset_class=dataset,collate_fn=None)




if __name__ == '__main__':
    main()