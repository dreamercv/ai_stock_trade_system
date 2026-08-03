import torch
from torch import nn

from einops import rearrange

from stockvqvae import StockVQVAE
from stockgpt import StockGPT

from config import TrainConfig

configs = TrainConfig()
class StockModel(nn.Module):
    def __init__(self,
                batchsize = 2,
                vq_encoder_nhead = 8,#60 // 5 * 30 + 30,
                vq_encoder_num_layers = 2,
                vq_decoder_nhead = 8,
                vq_decoder_num_layers = 2,
                dim_feedforward = 512,
                
                input_dim=6,
                latent_dim=256,
                fur_day_num = 5,
                day_dim = 48,
                token_num = 30,
                commitment_weight=0.25,
                ema_decay=0.8,

                vocab_size = 512,
                his_day_num = 60,
                gpt_nhead = 8,
                gpt_num_layersr=12,
                bias = True,

                dropout = 0.1,


        ):
        super().__init__()

        block_size = his_day_num // fur_day_num * token_num + token_num

        self.token_num = token_num
        self.batchsize = batchsize
        self.his_token_num = his_day_num // fur_day_num
        self.fur_day_num = fur_day_num
        self.day_dim = day_dim

        

        vqvae_config = VQVAEConfig(
            encoder_nhead = vq_encoder_nhead,#60 // 5 * 30 + 30,
            encoder_num_layers = vq_encoder_num_layers,
            decoder_nhead = vq_decoder_nhead,
            decoder_num_layers = vq_decoder_num_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,

            input_dim = input_dim,
            latent_dim = latent_dim,
            day_num = fur_day_num,
            day_dim = day_dim,
            seq_len = token_num,
            codebook_size = vocab_size,
            commitment_weight = commitment_weight,
            ema_decay = ema_decay

        )

        

        gpt_config = GPTConfig(
            block_size = block_size,#60 // 5 * 30 + 30,
            vocab_size = vocab_size,
            n_layer = gpt_num_layersr,
            n_head = gpt_nhead,
            n_embd = latent_dim,
            dropout = dropout,
            bias = True,

        )

        self.vqvae = StockVQVAE(vqvae_config,batchsize=batchsize)#input_dim = 6,latent_dim = 256,day_num = 5,day_dim=48,seq_len = 30)
        self.gpt = StockGPT(gpt_config)

    def idx_vqvae2gpt(self,indices):
        indices = indices.reshape(self.batchsize,-1,self.token_num)
        token_num = indices.shape[1]
        his_indices = indices[:,:self.his_token_num*self.token_num].reshape(self.batchsize,-1)
        if token_num > self.his_token_num:
            fur_indices = indices[:,self.his_token_num*self.token_num:].reshape(self.batchsize,-1)
            
        else:
            fur_indices = None
        return his_indices,fur_indices

    def idx_gpt2vqvae(self,indices):
        indices = rearrange(indices, 'b (s t) -> (b s) t',t=self.token_num)
        return indices

    def forward(self,x):
        

        ze = self.vqvae.encoder(self.vqvae.preprocess(x))
        zq,indices,commit_loss = self.vqvae.vq(ze)
        x_recon = self.vqvae.postprocess(self.vqvae.decoder(zq))


        his_indices,fur_indices = self.idx_vqvae2gpt(indices)
        logits, loss = self.gpt(his_indices,fur_indices)
        return logits, loss 

    def decoder(self,x):
        ze = self.vqvae.encoder(self.vqvae.preprocess(x))
        _,indices,_ = self.vqvae.vq(ze)
        his_indices,_ = self.idx_vqvae2gpt(indices)



        full_seq = self.gpt.autoregressive(his_indices, max_new_tokens=self.token_num)  
        fur_tokens = full_seq[:, -self.token_num:]   
        idx = fur_tokens.reshape(-1, self.token_num) 
        zq = self.vqvae.vq.get_output_from_indices(idx)  
        x_recon = self.vqvae.decoder(zq)              
        x_recon = x_recon.view(self.batchsize, self.fur_day_num, self.day_dim, -1)


        return x_recon
        

if __name__ == '__main__':
    bs = 2
    
    day = 65
    day_dim = 48
    feat_dim = 6 #(open close low high 成交量 量比  ...)
    x = torch.ones((bs,day,day_dim,feat_dim))
    model = StockModel(batchsize=bs)

    # model(x)
    model.decoder(x[:,:60])

    """
    2020.6-2021.6         2020.6-2020.12    1-0.5
    2021.6-2023.10        2020.12-2021.12   2-1
    2023.10-2024.10       2022.1-2025.5     1-3.5
    2024.10-2026.4        2025.5-2025.12    1.5-0.5
    2026.4-2026.6         2025.12-2026.6    

    """