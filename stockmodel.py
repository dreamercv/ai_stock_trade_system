import torch
from torch import nn

from einops import rearrange

from stockvqvae import StockVQVAE
from stockgpt import StockGPT

from config import TrainConfig


class StockModel(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.chunk_num = config.chunk_num
        self.batchsize = config.batch_size

        self.day_num = config.vqvae.day_num
        self.token_len = config.vqvae.seq_len
        self.his_token_num = config.his_chunk
        self.fur_token_num = config.fur_chunk
        self.data_dim = config.data_dim

        # self.token_num = token_num
        # self.his_token_num = his_day_num // fur_day_num
        # self.fur_day_num = fur_day_num
        # self.day_dim = day_dim

        
        

        self.vqvae = StockVQVAE(config.vqvae)#input_dim = 6,latent_dim = 256,day_num = 5,day_dim=48,seq_len = 30)
        self.gpt = StockGPT(config.gpt)

    def idx_vqvae2gpt(self,indices):
        indices = indices.reshape(self.batchsize,-1,self.token_len)
        token_num = indices.shape[1]
        his_indices = indices[:,:self.his_token_num,:].reshape(self.batchsize,-1)
        if token_num > self.his_token_num:
            fur_indices = indices[:,self.his_token_num:,:].reshape(self.batchsize,-1)
            
        else:
            fur_indices = None
        return his_indices,fur_indices

    def idx_gpt2vqvae(self,indices):
        indices = rearrange(indices, 'b (s t) -> (b s) t',t=self.token_num)
        return indices

    def preprocess(self,x):
        bs,total_days,day_dim,dim = x.shape
        assert total_days % self.chunk_num == 0, \
            f"total_days({total_days}) must be a multiple of day_num({self.chunk_num})"
        token_chunk = int(total_days // self.chunk_num)
        x = rearrange(x, 'b (chunk seq) dd d -> (b chunk) (seq dd) d',chunk=token_chunk,seq=self.chunk_num)
        return x

    def postprocess(self,x):
        x = rearrange(x, '(b chunk) (seq dd) f -> b (chunk seq) dd f',
                            b=self.batchsize, seq=self.day_num)
        return x

    def forward(self,x):
        
        x = self.preprocess(x)
        ze = self.vqvae.encoder(x)
        zq,indices,commit_loss = self.vqvae.vq(ze)
        x_recon = self.vqvae.decoder(zq)
        x_recon = self.postprocess(x_recon)


        his_indices,fur_indices = self.idx_vqvae2gpt(indices)
        logits, loss = self.gpt(his_indices,fur_indices)
        return logits, loss 

    def decoder(self,x):
        ze = self.vqvae.encoder(self.preprocess(x))
        _,indices,_ = self.vqvae.vq(ze)
        his_indices,_ = self.idx_vqvae2gpt(indices)



        full_seq = self.gpt.autoregressive(his_indices, max_new_tokens=self.token_len)  
        fur_tokens = full_seq[:, -self.token_len:]   
        idx = fur_tokens.reshape(-1, self.token_len) 
        zq = self.vqvae.vq.get_output_from_indices(idx)  
        x_recon = self.vqvae.decoder(zq)              
        x_recon = rearrange(x_recon, 'b (chunk dd) d-> b chunk dd d',dd=self.data_dim)
        return x_recon
        

if __name__ == '__main__':
    bs = 2
    
    day = 65
    day_dim = 48
    feat_dim = 6 #(open close low high 成交量 量比  ...)
    x = torch.ones((bs,day,day_dim,feat_dim))
    from config import TrainConfig
    configs = TrainConfig()
    model = StockModel(configs)

    model(x)
    # model.decoder(x[:,:60])

    """
    2020.6-2021.6         2020.6-2020.12    1-0.5
    2021.6-2023.10        2020.12-2021.12   2-1
    2023.10-2024.10       2022.1-2025.5     1-3.5
    2024.10-2026.4        2025.5-2025.12    1.5-0.5
    2026.4-2026.6         2025.12-2026.6    

    """