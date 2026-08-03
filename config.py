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

from dataclasses import dataclass,field


deepseek_token = "sk-1e25cdb95607405ebffd0f8948d0ae87"



    
@dataclass
class VQVAEConfig:

    day_num:int = 5 # 输入的天数必须是5的倍数
    day_dim:int=48
    seq_len:int = 30 # 5天*48个序列，每个序列有6个特征(open close low high 成交量 量比  ...)被切分成30个token
    input_dim:int = 6


    encoder_nhead:int=8 
    encoder_num_layers:int=2
    decoder_nhead:int=8 
    decoder_num_layers:int=2

    dim_feedforward:int=512

    dropout:float=0.1

    latent_dim:int = 256
    
    codebook_size:int=512

    commitment_weight:float=0.25
    ema_decay:float=0.8

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


@dataclass
class TrainConfig:
    # train config
    is_train = True
    device="cuda"
    log_dir = "logs/exp1"
    code_list  = "/workspace/afb5szh-01/models/stock/dataset/code_list.txt"
    batch_size = 2
    num_workers = 2


    epochs = 100
    lr = 1e-4
    weight_decay = 1e-4
    warmup_steps = 1000
    
    
    history_len = 60 
    future_len = 5

    cols = ["o","h","l","c","v"]

    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    
    def __post_init__(self):
        # 统一派生/对齐字段，避免 StockModel 里手算一遍
        self.gpt.block_size = (
            self.history_len // self.vqvae.day_num * self.vqvae.seq_len
            + self.vqvae.seq_len
        )
        self.gpt.vocab_size = self.vqvae.codebook_size
        self.gpt.n_embd = self.vqvae.latent_dim
        self.gpt.dropout = self.vqvae.dropout
