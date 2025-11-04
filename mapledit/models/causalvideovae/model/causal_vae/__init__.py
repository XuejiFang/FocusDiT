from .modeling_causalvae import CausalVAEModel
from ....mochivae.vae import MochiVAE
from ....mochivae.models import add_fourier_features, decode_latents_tiled_spatial

from einops import rearrange
from torch import nn
from diffusers import ModelMixin, ConfigMixin
from diffusers import AutoencoderKLMochi as AutoencoderKL
import torch

class CausalVAEModelWrapper(ModelMixin, ConfigMixin):
    def __init__(self, model_path, subfolder=None, cache_dir=None, use_ema=False, **kwargs):
        super(CausalVAEModelWrapper, self).__init__()
        # if os.path.exists(ckpt):
        # self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
        print(f"kwargs.ae: {kwargs['ae']}")
        if "Mochi" in kwargs['ae']:
            self.vae_cls = 'mochi'
            self.vae = AutoencoderKL.from_pretrained(model_path, torch_dtype=torch.float32)
            self.latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 12, 1, 1, 1)
            )
            self.latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 12, 1, 1, 1)
            )
        else:
            self.vae_cls = 'opensoraplan'
            self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs)
        if use_ema:
            self.vae.init_from_ema(model_path)
            self.vae = self.vae.ema

    @torch.no_grad()
    def encode(self, x):  # b c t h w
        if self.vae_cls == "opensoraplan":
            # x = self.vae.encode(x).sample()
            x = self.vae.encode(x).sample().mul_(0.18215)
        elif self.vae_cls == "mochi":
            # B C T H W
            x = self.vae.encode(x).latent_dist.sample()
            x = (x - self.latents_mean) / self.latents_std
        else:
            raise NotImplementedError

        return x
    
    @torch.no_grad()
    def decode(self, x):
        if self.vae_cls == "opensoraplan":
            # x = self.vae.decode(x)
            x = self.vae.decode(x / 0.18215)
            x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        elif self.vae_cls == "mochi":
            x = x * self.latents_std + self.latents_mean
            x = self.vae.decode(x, return_dict=False)[0]
            x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        else:
            raise NotImplementedError
        return x

    def dtype(self):
        return self.vae.dtype
    #
    # def device(self):
    #     return self.vae.device