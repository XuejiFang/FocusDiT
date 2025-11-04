from .models import Encoder, Decoder
from safetensors.torch import load_file
import torch
import torch.nn as nn
import json
from diffusers import ModelMixin, ConfigMixin

class MochiVAE(ModelMixin, ConfigMixin):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

        config = dict(
                    prune_bottlenecks=[False, False, False, False, False],
                    has_attentions=[False, True, True, True, True],
                    affine=True,
                    bias=True,
                    input_is_conv_1x1=True,
                    padding_mode="replicate"
                )

        self.encoder = Encoder(
                    in_channels=15,
                    base_channels=64,
                    channel_multipliers=[1, 2, 4, 6],
                    num_res_blocks=[3, 3, 4, 6, 3],
                    latent_dim=12,
                    temporal_reductions=[1, 2, 3],
                    spatial_reductions=[2, 2, 2],
                    **config,
                )
        encoder_ = load_file(f"{self.model_path}/encoder.safetensors")
        self.encoder.load_state_dict(encoder_, strict=False)

        self.decoder = Decoder(
                    out_channels=3,
                    base_channels=128,
                    channel_multipliers=[1, 2, 4, 6],
                    temporal_expansions=[1, 2, 3],
                    spatial_expansions=[2, 2, 2],
                    num_res_blocks=[3, 3, 4, 6, 3],
                    latent_dim=12,
                    has_attention=[False, False, False, False, False],
                    output_norm=False,
                    nonlinearity="silu",
                    output_nonlinearity="silu",
                    causal=True,
                )

        decoder_ = load_file(f"{self.model_path}/decoder.safetensors")
        self.decoder.load_state_dict(decoder_, strict=False)

        with open(f"{self.model_path}/vae_stats.json", 'r') as f:
            info = json.load(f)
            self.means = torch.Tensor(info['mean'])
            self.stds = torch.Tensor(info['std'])

        del encoder_, decoder_
        
