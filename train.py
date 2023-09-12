import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from tqdm import tqdm, trange
import numpy as np

from model.model import RIN

def infinite_generator(dataloader):
    while True:
        for batch in dataloader:
            yield batch
            
def gamma(t, ns=0.0002, ds=0.00025):
    return torch.cos(((t + ns) / (1 + ds)) * np.pi / 2)**2

def ddpm_step(x_t, eps_pred, t_now, t_next):
    # Estimate x at t_next with DDPM updating rule
    t_now = torch.tensor(t_now, device='cuda')
    t_next = torch.tensor(t_next, device='cuda')
    gamma_now = gamma(t_now)
    alpha_now = gamma(t_now) / gamma(t_next)
    sigma_now = torch.sqrt(1 - alpha_now)
    z = torch.randn_like(x_t)
    #x_pred = (x_t - sigma_now * eps_pred) / alpha_now
    #x_pred = torch.clip(x_pred, -1, 1)
    #eps = (1 / (torch.sqrt(1 - gamma_now))) * (x_t - torch.sqrt(gamma_now) * x_pred)
    eps_pred = torch.clip(eps_pred, -1., 1.)
    x_next = (1 / torch.sqrt(alpha_now)) * (x_t - ((1 - alpha_now)/(torch.sqrt(1 - gamma_now))) * eps_pred) + sigma_now * z
    return x_next
    
# based off of ddpm step code in lucidrains RIN repo, since can't tell if bug in sampling or model sucks
def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)
    
def ddpm_step_lucidrains(x_t, eps_pred, t_now, t_next):
    t_now = torch.tensor(t_now, device='cuda')
    t_next = torch.tensor(t_next, device='cuda')
    gamma_now = gamma(t_now)
    gamma_next = gamma(t_next)
    alpha_now, sigma_now = gamma_to_alpha_sigma(gamma_now)
    alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next)
    
    # convert eps into x_0
    x_start = safe_div(x_t - sigma_now * eps_pred, alpha_now)
    
    # clip
    x_start.clamp_(-1., 1.)
    
    # get predicted noise
    pred_noise = safe_div(x_t - alpha_now * x_start, sigma_now)
    
    # calculate next x_t
    x_next = x_start * alpha_next + pred_noise * sigma_next
    return x_next

def generate(steps, noise, latents, model):
    x_t = noise
    for step in trange(steps):
        # Get time for current and next states.
        t = 1 - step / steps
        timestep = torch.ones(x_t.shape[0], device='cuda') * t
        t_m1 = max(1 - (step + 1) / steps, 0)
        # Predict eps.
        eps_pred, latents = model(x_t, timestep, latents)
        # Estimate x at t_m1.
        x_t = ddpm_step_lucidrains(x_t, eps_pred, t, t_m1)
    return x_t
    
if __name__ == "__main__":

    # similar to CIFAR-10 config from authors
    model = RIN(img_size=32, patch_size=4, num_latents=127, latent_dim=384, embed_dim=128, num_blocks=3, num_layers_per_block=2).to('cuda')

    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CIFAR10(root="data", download=True, transform=tf, train=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    generator = infinite_generator(dataloader)

    optim = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, betas=(0.9, 0.99))
    loss_fn = torch.nn.MSELoss()

    n_steps = 100_001

    pbar = trange(n_steps)

    for i in pbar:
        # get only images, ignore labels
        batch = next(generator)[0].to('cuda')
        batch = batch * 2 - 1
        
        timestep = torch.rand(batch.shape[0]).to('cuda')
        noise = torch.randn_like(batch).to('cuda')
        
        noised_batch = torch.sqrt(gamma(timestep[:, None, None, None])) * batch + torch.sqrt(1 - gamma(timestep[:, None, None, None])) * noise
        
        if torch.rand(1) < 0.9:
            # self conditioning
            with torch.no_grad():
                _, latents = model(noised_batch, timestep)
            
        else:
            latents = None
            
        optim.zero_grad()
        #print(latents.shape)
        pred, _ = model(noised_batch, timestep, latents)
        
        # eps style (predicting noise) as in paper, but supposedly v-pred is usually better (try later?)
        
        loss = loss_fn(pred, noise)
        loss.backward()
        
        optim.step()
        
        pbar.set_description(f"loss: {loss.item():.4f}")
        
        if i % 200 == 0:
            noise = torch.randn_like(batch[:16]).to('cuda')
            latents = torch.zeros(16, 127, 384).to('cuda')
            with torch.no_grad():
                images = generate(400, noise, latents, model)
            images = images.cpu() * 0.5 + 0.5
            torchvision.utils.save_image(images, f"images/{i}.png", nrow=4)
        
