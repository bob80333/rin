import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FFN(nn.Module):
    def __init__(self, width, expansion=4, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(width, width * expansion)
        self.fc2 = nn.Linear(width * expansion, width)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# read embeds into latents
class ReadBlock(nn.Module):
    def __init__(self, embedding_width, latent_width):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(embed_dim=latent_width, kdim=embedding_width, vdim=embedding_width, batch_first=True, num_heads=latent_width//64)
        
        self.ffn = FFN(latent_width)
        
        self.mha_ln = nn.LayerNorm(latent_width)
        self.ffn_ln = nn.LayerNorm(latent_width)
        
    def forward(self, embeddings, latents):
        norm_latents = self.mha_ln(latents)
        latents = latents + self.mha(norm_latents, embeddings, embeddings)[0]
        
        norm_latents = self.ffn_ln(latents)
        latents = latents + self.ffn(norm_latents)
        
        return latents
    
# update latents / do computation
class ComputationBlock(nn.Module):
    def __init__(self, latent_width):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(embed_dim=latent_width, batch_first=True, num_heads=latent_width//64)
        
        self.ffn = FFN(latent_width)
        
        self.mha_ln = nn.LayerNorm(latent_width)
        self.ffn_ln = nn.LayerNorm(latent_width)
        
    def forward(self, latents):
        norm_latents = self.mha_ln(latents)
        latents = latents + self.mha(norm_latents, norm_latents, norm_latents)[0]
        
        norm_latents = self.ffn_ln(latents)
        latents = latents + self.ffn(norm_latents)
        
        return latents
    
# write latents into embeds
class WriteBlock(nn.Module):
    def __init__(self, embedding_width, latent_width):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(embed_dim=embedding_width, kdim=latent_width, vdim=latent_width, batch_first=True, num_heads=embedding_width//64)
        
        self.ffn = FFN(embedding_width)
        
        self.mha_ln = nn.LayerNorm(embedding_width)
        self.ffn_ln = nn.LayerNorm(embedding_width)
        
    def forward(self, embeddings, latents):
        embed_norm = self.mha_ln(embeddings)
        embeddings = embeddings + self.mha(embed_norm, latents, latents)[0]
        
        embed_norm = self.ffn_ln(embeddings)
        embeddings = embeddings + self.ffn(embed_norm)
        
        return embeddings
        
        
# from k-diffusion by crowsonkb

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
        
        
# based on authors code in pix2seq repo

def get_angles(pos, i, dim):
  angle_rates = 1 / torch.pow(10000., (2 * (i//2)).float() / dim)
  return pos.float() * angle_rates.float()

def positional_encoding(coords, dim):
  """coords in (bsz, size), return (bsz, size, dim)."""
  angle_rads = get_angles(coords.unsqueeze(-1),
                          torch.arange(dim)[None, None, :].to(coords.device),
                          dim)
  # apply sin to even indices in the array; 2i
  angle_rads1 = torch.sin(angle_rads[:, :, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads2 = torch.cos(angle_rads[:, :, 1::2])

  pos_encoding = torch.cat([angle_rads1, angle_rads2], -1)

  return pos_encoding.float()

class ScalarEmbedding(nn.Module):
    def __init__(self, in_features, out_features, normalize=True, scaling=1e4):
        super().__init__()
        self.scaling = scaling
        self.normalize = normalize
        self.out_features = out_features
        
        self.fc1 = nn.Linear(out_features, out_features)
        self.silu1 = nn.SiLU()
        self.fc2 = nn.Linear(out_features, out_features)
        
    def forward(self, x):
        x = positional_encoding(x*self.scaling, self.out_features)
        if self.normalize:
            x_mean = x.mean(dim=-1, keepdim=True)
            x_std = x.std(dim=-1, keepdim=True)
            x = (x - x_mean) / x_std
        x = self.fc1(x)
        x = self.silu1(x)
        x = self.fc2(x)
        return x
        
        