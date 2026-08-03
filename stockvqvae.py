# -*- encoding: utf-8 -*-
'''
@File         :stockvqvae.py
@Date         :2026/06/18 16:25:47
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  : https://github.com/lucidrains/vector-quantize-pytorch/tree/master

'''


import os,sys

from dataclasses import dataclass


import torch
from torch import nn
from einops import rearrange

from vector_quantize_pytorch import VectorQuantize

class Encoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=256, seq_len=240, out_len=30, nhead=8, num_layers=2,dim_feedforward=512,dropout=0.1):
        super().__init__()
        # 卷积下采样
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # 全局交互 Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, 6, 240) -> (B, 240, 6) ？注意：transpose后应是 (B,6,240)
        x = self.conv(x)               # (B, 256, 30)
        x = x.transpose(1, 2)          # (B, 30, 256)
        x = self.transformer(x)        # (B, 30, 256)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, output_dim=6, seq_len=240, token_len=30, nhead=8, num_layers=2,dim_feedforward=512, dropout=0.1):
        super().__init__()
        # 全局交互 Transformer（与编码器对称）
        decoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 转置卷积上采样
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, output_dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z_q):
        # z_q: (B, 30, 256)
        z_q = self.transformer(z_q)        # 全局交互 (B, 30, 256)
        z_q = z_q.transpose(1, 2)          # (B, 256, 30)
        x = self.deconv(z_q)               # (B, 6, 240)
        x = x.transpose(1, 2)              # (B, 240, 6)
        return x

class StockVQVAE(nn.Module):
    def __init__(self,config,batchsize):
        super().__init__()
        self.batchsize = batchsize

        self.input_dim = config.input_dim
        self.day_num = config.day_num
        self.day_dim = config.day_dim
        self.input_seq_len = config.day_num * config.day_dim # 240
        self.seq_len = config.seq_len # 30
        self.encoder_nhead=config.encoder_nhead
        self.encoder_num_layers=config.encoder_num_layers
        self.decoder_nhead=config.decoder_nhead
        self.decoder_num_layers=config.decoder_num_layers

        self.dim_feedforward=config.dim_feedforward

        self.dropout=config.dropout

        self.encoder = Encoder(
            input_dim=config.input_dim, latent_dim=config.latent_dim, seq_len=self.input_seq_len, out_len=config.seq_len, 
            nhead=config.encoder_nhead, num_layers=config.encoder_num_layers,dim_feedforward=config.dim_feedforward,dropout=config.dropout
        )
        
        self.vq = VectorQuantize(
            dim = config.latent_dim, 
            codebook_size=config.codebook_size,
            decay=config.ema_decay,
            commitment_weight=config.commitment_weight
        )

        self.decoder = Decoder(
            latent_dim=config.latent_dim, output_dim=config.input_dim, seq_len=self.input_seq_len, token_len=self.seq_len,
            nhead=config.decoder_nhead, num_layers=config.decoder_num_layers,dim_feedforward=config.dim_feedforward,dropout=config.dropout
        )

    def preprocess(self,x):
        bs,total_days,day_dim,dim = x.shape
        assert total_days % self.day_num == 0, \
            f"total_days({total_days}) must be a multiple of day_num({self.day_num})"
        token_chunk = int(total_days // self.day_num)
        x = rearrange(x, 'b (chunk seq) dd d -> (b chunk) (seq dd) d',chunk=token_chunk,seq=self.day_num)
        return x

    def postprocess(self,x):
        x = rearrange(x, '(b chunk) (seq dd) f -> b (chunk seq) dd f',
                            b=self.batchsize, seq=self.day_num, dd=self.day_dim)
        return x

    def forward(self,x):
        x = self.preprocess(x)
        ze = self.encoder(x)
        zq,indices,commit_loss = self.vq(ze)
        x_recon = self.decoder(zq) 
        x_recon = self.postprocess(x_recon)

        return x_recon, indices, commit_loss




if __name__ == '__main__':
    # bs 240 F --> bs 30 d_latent # 5*48=240
    bs = 2
    day = 10
    day_dim = 48
    feat_dim = 6 #(open close low high 成交量 量比  ...)
    input = torch.ones((bs,day,day_dim,feat_dim))
    model = StockVQVAE(VQVAEConfig)
    output = model(input)
    print(output[0].shape)