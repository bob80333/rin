import numpy as np
import torch
import torchvision

device = 'cuda'


def gamma(t, ns=0.0002, ds=0.00025):
    return torch.cos(((t + ns) / (1 + ds)) * np.pi / 2)**2


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


model = torch.load("model.pt")

model.eval()

noise = torch.randn_like(10, 16, 3, 32, 32).to(device)
latents = torch.zeros(10, 16, 126, 512).to(device)

for i in range(10):
    
    labels = torch.ones(16).to(device) * i
    
    with torch.inference_mode():
        images = generate(400, noise, latents, model, labels)
        
    images = images.cpu() * 0.5 + 0.5
    
    torchvision.utils.save_image(images, f"images_output/{i}.png", nrow=4)
    
model.train()