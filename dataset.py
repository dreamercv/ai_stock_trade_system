# -*- encoding: utf-8 -*-
'''
@File         :dataset.py
@Date         :2026/07/23 15:04:51
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys
import bisect
import glob
import torch
import numpy as np
import pandas as pd

from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader,Dataset


class DataSet(Dataset):
    def __init__(self,config):
        super().__init__()
        self.code_list = config.code_list
        self.history_len = config.history_len
        self.future_len = config.future_len
        self.seq_len = self.history_len + self.future_len
        self.stock_infos = self.prepare_dataset(self.code_list)
        self.names = sorted(self.stock_infos.keys())
        # 每个股票的有效样本数量
        self.sample_counts = [
            len(self.stock_infos[name]) - self.seq_len + 1 
            for name in self.names
        ]
        self.cum_counts = np.cumsum(self.sample_counts).tolist()
        self.total = self.cum_counts[-1] if self.cum_counts else 0
        self.cols = ["o","h","l","c","v"]


    def __getitem__(self,index):
        clip_id,internal_idx,clip_name = self._locate(index)
        paths = sorted(self.stock_infos[clip_name][internal_idx:internal_idx+self.seq_len])

        seq_data = []
        norm_values = None

        for i, path in enumerate(paths):
            
            df = pd.read_csv(path)
            df["t"] = pd.to_datetime(df["t"])
            df = df.sort_values("t")
            trade_data = np.stack([df[col] for col in self.cols]).transpose(1,0)
            if i == self.history_len - 1:
                norm_values = trade_data[-1,:]
            seq_data.append(trade_data)
        seq_data = np.stack(seq_data) / norm_values


        return torch.Tensor(seq_data)


    def __len__(self):
        return self.total


    def _locate(self, index):
        """返回 (clip_id, internal_idx)"""
        # 找到第一个 cum_counts > index 的位置
        clip_id = bisect.bisect_right(self.cum_counts, index)
        if clip_id == 0:
            internal_idx = index
        else:
            internal_idx = index - self.cum_counts[clip_id - 1]
        clip_name = self.names[clip_id]
        return clip_id, internal_idx,clip_name



            


    def prepare_dataset(self,path_list):
        stock_infos = {}
        with open(path_list,"r") as f:
            lines = f.readlines()
            for line in lines:
                clip_path = line.strip()
                code_name = clip_path.strip(os.sep).split(os.sep)[-1]
                datas = sorted(glob.glob(f"{clip_path.rstrip(os.sep)}/*.csv"))
                stock_infos[code_name] = datas
        return stock_infos

def worker_rnd_init(x):
    np.random.seed(13 + x)


def build_dataloader(config, dataset_class=None,collate_fn=None):
    dataset = dataset_class(config)
    if config.is_train:
        # 仅当已调用 torch.distributed.init_process_group() 时才用 DistributedSampler，否则单卡/本地调试会报错
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        pin_memory = False
    else:
        sampler = SequentialSampler(dataset)
        shuffle = False
        pin_memory = True
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_rnd_init
    )

    return dataloader,sampler


if __name__ == '__main__':
    from config import Config
    dataset = DataSet(Config)
    # dataset.__getitem__(0)
    dataset.__getitem__(48+136 - 1)
    
    
    print()