from model.layers import WriteBlock, ReadBlock, ComputationBlock, FFN, ScalarEmbedding

import torch
import torch.nn as nn

class RINBlock(nn.Module):
    def __init__(self, embedding_width, latent_width, num_layers):
        super().__init__()
        self.read = ReadBlock(embedding_width, latent_width)
        self.write = WriteBlock(embedding_width, latent_width)
        
        self.computation = [ComputationBlock(latent_width) for _ in range(num_layers)]
        self.computation = nn.Sequential(*self.computation)
        
    def forward(self, embeddings, latents):
        latents = self.read(embeddings, latents)
        latents = self.computation(latents)
        embeddings = self.write(embeddings, latents)
        
        return embeddings, latents
            


class RIN(nn.Module):
    def __init__(self, img_size, patch_size, num_latents, latent_dim, embed_dim, num_blocks, num_layers_per_block):
        super().__init__()
        self.img_size = img_size
        self.size = img_size // patch_size
        self.patch_size = patch_size
        
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        
        self.embed_dim = embed_dim
        
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        
        self.input_conv = nn.Conv2d(3, self.embed_dim, self.patch_size, self.patch_size)
        self.input_ln = nn.LayerNorm(self.embed_dim) # not sure about this, but it's in the paper (https://github.com/crowsonkb mentioned in EAI discord that a similar LN before patching in ViT diffuion models caused some artifacts)
        
        self.output_linear = nn.Linear(self.embed_dim, 3 * self.patch_size * self.patch_size)
        # self.output_ln = nn.LayerNorm(embed_dim) # from same discord convo, mentioned that this is good at least for normal ViT (not sure why)
        
        self.pos_embedding = nn.Parameter(torch.empty(1, self.size * self.size, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, a=-0.02, b=0.02)
        
        self.latents = nn.Parameter(torch.empty(1, self.num_latents, self.latent_dim))
        nn.init.trunc_normal_(self.latents, a=-0.02, b=0.02)
        
        self.prev_latents_ffn = FFN(self.latent_dim, dropout=0.0)
        self.prev_latents_ln = nn.LayerNorm(self.latent_dim)
        
        self.timestep_embed = ScalarEmbedding(1, self.latent_dim)
        
        self.blocks = nn.ModuleList()
        
        for _ in range(self.num_blocks):
            self.blocks.append(RINBlock(self.embed_dim, self.latent_dim, self.num_layers_per_block))
            
    def forward(self, input, timestep, prev_latents = None):
        
        # encode input into embeddings
        embeddings = self.input_conv(input)
        # squeeze into 2d
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # add pos embed & ln
        embeddings = self.input_ln(embeddings) + self.pos_embedding
        
        latents = self.latents.repeat(embeddings.shape[0], 1, 1)

        if prev_latents is not None:
            prev_latents = self.prev_latents_ffn(prev_latents)
            latents = latents + self.prev_latents_ln(prev_latents)
            
        # add timestep   
        ts_embed = self.timestep_embed(timestep[:, None])
        
        latents = torch.concat([latents, ts_embed], dim=1)
            
        for block in self.blocks:
            embeddings, latents = block(embeddings, latents)
            
        # now we have our embeddings, so we can decode them into an image
        patches = self.output_linear(embeddings)
        # now we have to unflatten them
        image = patches.transpose(1, 2).reshape(-1, 3, self.img_size, self.img_size)
        
        return image, latents[:, :-1]
            
        
        
        
            
        
        
            
            
        