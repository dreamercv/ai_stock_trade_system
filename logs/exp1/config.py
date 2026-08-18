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
    seq_len:int = 30 # 30个token，相当于三十个时间序列 ：要求 day_num*day_dim//seq_len能整除。5天*48个序列，每个序列有6个特征(open close low high 成交量 量比  ...)被切分成30个token 
    input_dim:int = 6

    assert (day_num*day_dim) % seq_len ==0, f"必须整除"


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
    block_size: int = 30 # 回归步数
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 8  # 须整除 n_embd；与 latent_dim=256 配套（256%8==0）
    n_embd: int = 256  # 与 VQVAE latent_dim 对齐
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


@dataclass
class TrainConfig:
    # train config
    is_train = True
    device="cuda"
    log_dir = "logs/exp1"
    code_list  = "/home/ai_system/dataset/dataset_list.txt"
    batch_size = 2
    num_workers = 2


    chunk_num = 5 # 5天一个块 # 五天分成30块，30个时间序列
    data_dim = 48 # 15分钟的数据一天有48个序列，每个序列有cols个特征

    epochs = 100
    lr = 1e-4
    weight_decay = 1e-4
    warmup_steps = 1000
    
    his_chunk = 12
    fur_chunk = 1
    
    

    cols = ["o","h","l","c","v"]

    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    
    def __post_init__(self):
        # 更新trainconfig
        self.history_len = self.chunk_num *  self.his_chunk
        self.future_len  = self.chunk_num *  self.fur_chunk


        # 更新vqvaeconfg
        self.vqvae.input_dim = len(self.cols)
        self.vqvae.day_num = self.chunk_num
        self.vqvae.day_dim = self.data_dim


        # 统一派生/对齐字段，避免 StockModel 里手算一遍
        self.gpt.block_size = (self.history_len + self.future_len) // self.chunk_num * self.vqvae.seq_len

        self.gpt.vocab_size = self.vqvae.codebook_size
        self.gpt.n_embd = self.vqvae.latent_dim
        self.gpt.dropout = self.vqvae.dropout
        if self.gpt.n_embd % self.gpt.n_head != 0:
            self.gpt.n_head = self.vqvae.encoder_nhead
