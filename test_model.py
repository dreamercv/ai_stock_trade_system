# -*- encoding: utf-8 -*-
'''
@File         :test_model.py
@Date         :2026/08/18 21:19:10
@Author       :Binge.Van
@E-mail       :1367955240@qq.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

import torch
from torch import nn

from config import TrainConfig
from stockmodel import StockModel
from dataset import DataSet

def main():
    pass


if __name__ == '__main__':
    config = TrainConfig()
    model = StockModel(config)
    x = torch.ones((config.batch_size,config.history_len+config.future_len,config.data_dim,len(config.cols)))
    # model(x)
    
    y = torch.ones((config.batch_size,config.history_len,config.data_dim,len(config.cols)))
    model.decoder(y)
    dataset = DataSet(config)
    dataset.__getitem__(0)
