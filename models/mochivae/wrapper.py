from einops import rearrange
from diffusers import ModelMixin, ConfigMixin
from diffusers import AutoencoderKLMochi as AutoencoderKL
import torch


class MochiVAEWrapper(ModelMixin, ConfigMixin):
    def __init__(self, model_path, **kwargs):
        super(MochiVAEWrapper, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(model_path, torch_dtype=torch.float32)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 12, 1, 1, 1)
        self.latents_std = torch.tensor(self.vae.config.latents_std).view(1, 12, 1, 1, 1)

    @torch.no_grad()
    def encode(self, x):
        x = self.vae.encode(x).latent_dist.sample()
        x = (x - self.latents_mean) / self.latents_std
        return x

    @torch.no_grad()
    def decode(self, x):
        x = x * self.latents_std + self.latents_mean
        x = self.vae.decode(x, return_dict=False)[0]
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x

    def dtype(self):
        return self.vae.dtype