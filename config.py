# -*- encoding: utf-8 -*-
'''
@File         :config.py
@Date         :2026/07/24 17:57:34
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

from dataclasses import dataclass


@dataclass
class Config:
    # train config
    is_train = True
    devide="cuda"
    code_list  = "/workspace/afb5szh-01/models/stock/dataset/code_list.txt"
    batch_size = 2
    num_workers = 2
    history_len = 60 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    future_len = 5
    



if __name__ == '__main__':
    main()