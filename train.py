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

import logging
import time


import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler



from dataset import build_dataloader


cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
torch.multiprocessing.set_sharing_strategy('file_system')



from tensorboardX import SummaryWriter

def create_scheduler_with_warmup(optimizer, warmup_steps, total_steps, base_lr, eta_min=0):
    # 定义 warmup 阶段的学习率因子
    def lambda_lr(step):
        if step < warmup_steps:
            return step / warmup_steps  # 线性增长到 1
        else:
            # 超过 warmup 后，因子保持 1，让 CosineAnnealingLR 接手
            return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    # 余弦退火调度器（假设从 base_lr 开始衰减）
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min
    )
    # 将两个调度器串联
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )
    return scheduler

def init_log(log_dir,script_path,rank,SummaryWriter):
    os.makedirs(log_dir,exist_ok=True)
    # 日志文件名（带时间戳）
    if rank == 0:
        log_file = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        writer = SummaryWriter(logdir=log_dir)
        config_path = os.path.join(os.path.dirname(script_path), "config.py")
        os.system(f"cp {config_path} {log_dir}")
    else:
        # 禁用非主进程的日志输出
        logging.basicConfig(level=logging.ERROR)
    return logging,writer



def main(model,dataset,configs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default = -1, type=int)
    args = parser.parse_args()

    

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

    #参数初始化
    lr = configs.lr
    weight_decay = configs.weight_decay
    warmup_steps = configs.warmup_steps
    epochs = configs.epochs
    total_lr = configs.lr * world_size
    log_dir = configs.log_dir
    script_path = os.path.abspath(__file__)



    dataloader,sampler = build_dataloader(configs,dataset_class=dataset,collate_fn=None)

    steps_per_epoch = len(dataloader)
    total_steps = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,betas=(0.9,0.95))
    scheduler = create_scheduler_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=lr,
        eta_min=0
    )
    logging,writer = init_log(log_dir,script_path,rank,SummaryWriter)

    if rank == 0:
        logging.info(
            f"[train] epochs: {epochs}; steps_per_epoch: {steps_per_epoch}, "
            f"total_steps (optimizer steps): {total_steps}"
        )
        logging.info(f"Total LR (global): {total_lr:.2e}, Per-GPU LR: {lr:.2e}, world_size: {world_size}")

    iteration = 0
    model.train()
    for epoch in range(epochs):
        for i, batch in  enumerate(dataloader):
            iteration += 1
            optimizer.zero_grad()
            _,loss = model(batch)
            # loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            logging.info(f"Epoch: {epoch}/{epochs}; iteration:{iteration}, Loss: {loss.item():.4e}")
            



if __name__ == '__main__':
    from stockmodel import StockModel
    from config import TrainConfig
    configs = TrainConfig()
    model = StockModel(configs)
    from dataset import DataSet
    # dataset = DataSet(configs)
    
    main(model,DataSet,configs)