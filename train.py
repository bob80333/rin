from copy import deepcopy

import numpy as np
import torch
import torchvision
import wandb
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from tqdm import tqdm, trange

from model.model import RIN

# from k-diffusion discord channel in eleutherai

import torch.distributed as dist

def stratified_uniform(shape, grad_accum_steps=1, grad_accum_step=0, group=None, world_size=None, rank=None, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution. The strata are not duplicated
    across processes or gradient accumulation steps."""
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group) if world_size is None else world_size
        rank = dist.get_rank(group) if rank is None else rank
    else:
        world_size = 1 if world_size is None else world_size
        rank = 0 if rank is None else rank
    world_size = world_size * grad_accum_steps
    rank = rank * grad_accum_steps + grad_accum_step
    n = shape[-1] * world_size
    start = rank * n // world_size
    end = (rank + 1) * n // world_size
    offsets = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[start:end]
    u = torch.rand(shape, dtype=dtype, device=device)
    return torch.clamp(offsets + u / n, 0, 1)

# from k-diffusion

device = 'cuda'


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


def infinite_generator(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def gamma(t, ns=0.0002, ds=0.00025):
    return torch.cos(((t + ns) / (1 + ds)) * np.pi / 2)**2


def ddpm_step(x_t, eps_pred, t_now, t_next):
    # Estimate x at t_next with DDPM updating rule
    t_now = torch.tensor(t_now, device=device)
    t_next = torch.tensor(t_next, device=device)
    gamma_now = gamma(t_now)
    alpha_now = gamma(t_now) / gamma(t_next)
    sigma_now = torch.sqrt(1 - alpha_now)
    z = torch.randn_like(x_t)
    #x_pred = (x_t - sigma_now * eps_pred) / alpha_now
    #x_pred = torch.clip(x_pred, -1, 1)
    #eps = (1 / (torch.sqrt(1 - gamma_now))) * (x_t - torch.sqrt(gamma_now) * x_pred)
    eps_pred = torch.clip(eps_pred, -1., 1.)
    x_next = (1 / torch.sqrt(alpha_now)) * (x_t - ((1 - alpha_now) /
                                                   (torch.sqrt(1 - gamma_now))) * eps_pred) + sigma_now * z
    return x_next

# based off of ddpm step code in lucidrains RIN repo, since can't tell if bug in sampling or model sucks


def gamma_to_alpha_sigma(gamma, scale=1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)


def safe_div(numer, denom, eps=1e-10):
    return numer / denom.clamp(min=eps)


def ddpm_step_lucidrains(x_t, eps_pred, t_now, t_next):
    t_now = torch.tensor(t_now, device=device)
    t_next = torch.tensor(t_next, device=device)
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


def generate(steps, noise, latents, model, conditioning):
    x_t = noise
    for step in trange(steps):
        # Get time for current and next states.
        t = 1 - step / steps
        timestep = torch.ones(x_t.shape[0], device=device) * t
        t_m1 = max(1 - (step + 1) / steps, 0)
        # Predict eps.
        eps_pred, latents = model(x_t, timestep, conditioning, latents)
        # Estimate x at t_m1.
        x_t = ddpm_step_lucidrains(x_t, eps_pred, t, t_m1)
    return x_t


if __name__ == "__main__":
    torch.manual_seed(42)

    wandb.init(project="rin")

    # similar to CIFAR-10 config from authors
    model = RIN(img_size=32, patch_size=2, num_latents=126, latent_dim=512,
                embed_dim=128, num_blocks=3, num_layers_per_block=2).to(device)

    model_ema = deepcopy(model)
    model_ema.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CIFAR10(root="data", download=True, transform=tf, train=True)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=True, num_workers=4)
    generator = infinite_generator(dataloader)

    optim = AdamW(model.parameters(), lr=3e-4,
                  weight_decay=1e-2, betas=(0.9, 0.99))
    loss_fn = torch.nn.MSELoss()

    n_steps = 1_001

    pbar = trange(n_steps)

    for i in pbar:
        # get only images, ignore labels
        batch, labels = next(generator)
        batch = batch.to('cuda')
        labels = labels.to('cuda')
        batch = batch * 2 - 1

        timestep = stratified_uniform(batch.shape[:1]).to(device)
        noise = torch.randn_like(batch).to(device)

        noised_batch = torch.sqrt(gamma(timestep[:, None, None, None])) * batch + torch.sqrt(
            1 - gamma(timestep[:, None, None, None])) * noise

        if torch.rand(1) < 0.9:
            # self conditioning
            with torch.no_grad():
                _, latents = model(noised_batch, timestep, labels)

        else:
            latents = None

        optim.zero_grad()
        # print(latents.shape)
        pred, _ = model(noised_batch, timestep, labels, latents)

        # eps style (predicting noise) as in paper, but supposedly v-pred is usually better (try later?)

        loss = loss_fn(pred, noise)
        loss.backward()

        optim.step()

        ema_update(model, model_ema, 0.9995)

        pbar.set_description(f"loss: {loss.item():.4f}")

        if i % 500 == 0:
            model.eval()
            noise = torch.randn_like(batch[:10]).to(device)
            labels = torch.arange(10).to(device)
            latents = torch.zeros(10, 126, 512).to(device)
            with torch.no_grad():
                images = generate(400, noise, latents, model_ema, labels)
            images = images.cpu() * 0.5 + 0.5
            torchvision.utils.save_image(images, f"images/{i}.png", nrow=4)
            model.train()

        wandb.log({"loss": loss.item()}, step=i)
