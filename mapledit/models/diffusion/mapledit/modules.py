from torch import nn
import torch
import numpy as np

from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
import re
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward, GatedSelfAttentionDense
from diffusers.models.attention_processor import Attention as Attention_
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from .rope import PositionGetter3D, RoPE3D

torch_npu = None
npu_config = None
set_run_dtype = None

logger = logging.get_logger(__name__)

class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        # TODO: check if this is correct
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x
    
def get_3d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16, 
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)
    grid_t = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale[0]
    grid_h = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale[1]
    grid_w = np.arange(grid_size[2], dtype=np.float32) / (grid_size[2] / base_size[2]) / interpolation_scale[2]
    grid = np.meshgrid(grid_w, grid_h, grid_t)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[2], grid_size[1], grid_size[0]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    # import ipdb;ipdb.set_trace()
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # import ipdb;ipdb.set_trace()
    # use 1/3 of dimensions to encode grid_t/h/w
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (T*H*W, D/3)

    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)  # (T*H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16, 
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale[0]
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use 1/3 of dimensions to encode grid_t/h/w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16, 
):
    """
    grid_size: int of the grid return: pos_embed: [grid_size, embed_dim] or
    [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)

    grid = np.arange(grid_size, dtype=np.float32) / (grid_size / base_size) / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D/2)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        num_frames=1, 
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True, 
    ):
        super().__init__()
        # assert num_frames == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161

        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t
        temp_pos_embed = get_1d_sincos_pos_embed(embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t)
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)
        # self.temp_embed_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, latent, num_frames):
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None
        # b c 1 h w
        # assert latent.shape[-3] == 1 and num_frames == 1
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed


            if self.num_frames != num_frames:
                # import ipdb;ipdb.set_trace()
                # raise NotImplementedError

                temp_pos_embed = get_1d_sincos_pos_embed(
                    embed_dim=self.temp_pos_embed.shape[-1],
                    grid_size=num_frames,
                    base_size=self.base_size_t,
                    interpolation_scale=self.interpolation_scale_t,
                )

                temp_pos_embed = torch.from_numpy(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)
        
        latent = rearrange(latent, '(b t) n c -> b t n c', b=b)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (video_latent + temp_pos_embed).to(video_latent.dtype) if video_latent is not None and video_latent.numel() > 0 else None
            image_latent = (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype) if image_latent is not None and image_latent.numel() > 0 else None

        video_latent = rearrange(video_latent, 'b t n c -> b (t n) c') if video_latent is not None and video_latent.numel() > 0 else None
        image_latent = rearrange(image_latent, 'b t n c -> (b t) n c') if image_latent is not None and image_latent.numel() > 0 else None

        if num_frames == 1 and image_latent is None:
            image_latent = video_latent
            video_latent = None
        # print('video_latent is None, image_latent is None', video_latent is None, image_latent is None)
        return video_latent, image_latent
    
class Attention(Attention_):
    def __init__(self, downsampler, attention_mode, use_rope, interpolation_scale_thw, **kwags):
        processor = AttnProcessor2_0(attention_mode=attention_mode, use_rope=use_rope, interpolation_scale_thw=interpolation_scale_thw)
        super().__init__(processor=processor, **kwags)
        self.downsampler = None
        if downsampler: # downsampler  k155_s122
            downsampler_ker_size = list(re.search(r'k(\d{2,3})', downsampler).group(1)) # 122
            down_factor = list(re.search(r's(\d{2,3})', downsampler).group(1))
            downsampler_ker_size = [int(i) for i in downsampler_ker_size]
            downsampler_padding = [(i - 1) // 2 for i in downsampler_ker_size]
            down_factor = [int(i) for i in down_factor]
            
            if len(downsampler_ker_size) == 2:
                self.downsampler = DownSampler2d(kwags['query_dim'], kwags['query_dim'], kernel_size=downsampler_ker_size, stride=1,
                                            padding=downsampler_padding, groups=kwags['query_dim'], down_factor=down_factor,
                                            down_shortcut=True)
            elif len(downsampler_ker_size) == 3:
                self.downsampler = DownSampler3d(kwags['query_dim'], kwags['query_dim'], kernel_size=downsampler_ker_size, stride=1,
                                            padding=downsampler_padding, groups=kwags['query_dim'], down_factor=down_factor,
                                            down_shortcut=True)
                
        # self.q_norm = nn.LayerNorm(kwags['dim_head'], elementwise_affine=True, eps=1e-6)
        # self.k_norm = nn.LayerNorm(kwags['dim_head'], elementwise_affine=True, eps=1e-6) 

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads

        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

class DownSampler3d(nn.Module):
    def __init__(self, *args, **kwargs):
        ''' Required kwargs: down_factor, downsampler'''
        super().__init__()
        self.down_factor = kwargs.pop('down_factor')
        self.down_shortcut = kwargs.pop('down_shortcut')
        self.layer = nn.Conv3d(*args, **kwargs)

    def forward(self, x, attention_mask, t, h, w):
        b = x.shape[0]
        x = rearrange(x, 'b (t h w) d -> b d t h w', t=t, h=h, w=w)
        if npu_config is None:
            x = self.layer(x) + (x if self.down_shortcut else 0)
        else:
            x_dtype = x.dtype
            x = npu_config.run_conv3d(self.layer, x, x_dtype) + (x if self.down_shortcut else 0)
        
        self.t = t//self.down_factor[0]
        self.h = h//self.down_factor[1]
        self.w = w//self.down_factor[2]
        x = rearrange(x, 'b d (t dt) (h dh) (w dw) -> (b dt dh dw) (t h w) d', 
                      t=t//self.down_factor[0], h=h//self.down_factor[1], w=w//self.down_factor[2], 
                         dt=self.down_factor[0], dh=self.down_factor[1], dw=self.down_factor[2])
    

        attention_mask = rearrange(attention_mask, 'b 1 (t h w) -> b 1 t h w', t=t, h=h, w=w)
        attention_mask = rearrange(attention_mask, 'b 1 (t dt) (h dh) (w dw) -> (b dt dh dw) 1 (t h w)',
                      t=t//self.down_factor[0], h=h//self.down_factor[1], w=w//self.down_factor[2], 
                         dt=self.down_factor[0], dh=self.down_factor[1], dw=self.down_factor[2])
        return x, attention_mask
        
    def reverse(self, x, t, h, w):
        x = rearrange(x, '(b dt dh dw) (t h w) d -> b (t dt h dh w dw) d', 
                      t=t, h=h, w=w, 
                         dt=self.down_factor[0], dh=self.down_factor[1], dw=self.down_factor[2])
        return x


class DownSampler2d(nn.Module):
    def __init__(self, *args, **kwargs):
        ''' Required kwargs: down_factor, downsampler'''
        super().__init__()
        self.down_factor = kwargs.pop('down_factor')
        self.down_shortcut = kwargs.pop('down_shortcut')
        self.layer = nn.Conv2d(*args, **kwargs)

    def forward(self, x, attention_mask, t, h, w):
        b = x.shape[0]
        x = rearrange(x, 'b (t h w) d -> (b t) d h w', t=t, h=h, w=w)
        x = self.layer(x) + (x if self.down_shortcut else 0)

        self.t = 1
        self.h = h//self.down_factor[0]
        self.w = w//self.down_factor[1]

        x = rearrange(x, 'b d (h dh) (w dw) -> (b dh dw) (h w) d', 
                      h=h//self.down_factor[0], w=w//self.down_factor[1], 
                      dh=self.down_factor[0], dw=self.down_factor[1])
    
        attention_mask = rearrange(attention_mask, 'b 1 (t h w) -> (b t) 1 h w', h=h, w=w)
        attention_mask = rearrange(attention_mask, 'b 1 (h dh) (w dw) -> (b dh dw) 1 (h w)',
                      h=h//self.down_factor[0], w=w//self.down_factor[1], 
                         dh=self.down_factor[0], dw=self.down_factor[1])
        return x, attention_mask
    
    def reverse(self, x, t, h, w):
        x = rearrange(x, '(b t dh dw) (h w) d -> b (t h dh w dw) d', 
                      t=t, h=h, w=w, 
                      dh=self.down_factor[0], dw=self.down_factor[1])
        return x
    
class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attention_mode='xformers', use_rope=False, interpolation_scale_thw=(1, 1, 1)):
        self.use_rope = use_rope
        self.interpolation_scale_thw = interpolation_scale_thw
        if self.use_rope:
            self._init_rope(interpolation_scale_thw)
        self.attention_mode = attention_mode
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")


    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        frame: int = 8, 
        height: int = 16, 
        width: int = 16, 
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)


        if attn.downsampler is not None:
            hidden_states, attention_mask = attn.downsampler(hidden_states, attention_mask, t=frame, h=height, w=width)
            frame, height, width = attn.downsampler.t, attn.downsampler.h, attn.downsampler.w

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            raise NotImplementedError
            # H = attn.heads
            # attention_mask = repeat(attention_mask, "B N N2 -> B H N N2", H=H)
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # qk norm
        # query = attn.q_norm(query)
        # key = attn.k_norm(key)

        if self.use_rope:
            # require the shape of (batch_size x nheads x ntokens x dim)
            pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width, device=query.device)
            query = self.rope(query, pos_thw)
            key = self.rope(key, pos_thw)
            
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
        # 0, 0 ->(bool) False, False ->(any) False ->(not) True
        # TODO
        if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
            attention_mask = None

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # import ipdb;ipdb.set_trace()
        # print(attention_mask)
        if self.attention_mode == 'flash':
            assert attention_mask is None, 'flash-attn do not support attention_mask'
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
        elif self.attention_mode == 'xformers':
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        elif self.attention_mode == 'math':
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        elif self.attention_mode == 'my':
            if attention_mask is None:
                attention_mask = 1
            eps = 1e-6
            scale_factor = query.size(-1) ** 0.25
            # attn_weight: B H N N
            query = query / scale_factor
            key = key / scale_factor
            x = query @ key.transpose(-2, -1) 
            exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values) * attention_mask
            attn_weight = exp_x / (exp_x.sum(dim=-1, keepdim=True) + eps)
            attn_weight_ = attn_weight / (attn_weight.sum(dim=-1, keepdim=True) + eps)
            hidden_states = attn_weight_ @ value
        else:
            raise NotImplementedError(f'Found attention_mode: {self.attention_mode}')
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if attn.downsampler is not None:
            hidden_states = attn.downsampler.reverse(hidden_states, t=frame, h=height, w=width)
        return hidden_states

class MLP(nn.Module):
    def __init__(self, input_channel):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_channel+1, 512)
        self.fc2 = nn.Linear(512+1, 128)
        self.fc3 = nn.Linear(128+1, 32)
        self.fc4 = nn.Linear(32+1, 16)
        self.fc5 = nn.Linear(16+1, 1)
        self.ac1 = nn.ReLU()
        self.ac2 = nn.ReLU()
        self.ac3 = nn.ReLU()
        self.ac4 = nn.ReLU()
        self.ac5 = nn.ReLU()

    def forward(self, x, mask_pre=None):
        add_mask = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype) if mask_pre is None else mask_pre[:,:,:1]

        x = self.ac1(self.fc1(torch.cat((x, add_mask), dim=2)))
        x = self.ac2(self.fc2(torch.cat((x, add_mask), dim=2)))
        x = self.ac3(self.fc3(torch.cat((x, add_mask), dim=2)))
        x = self.ac4(self.fc4(torch.cat((x, add_mask), dim=2)))
        # N B C -> N B 1
        x = self.fc5(torch.cat((x, add_mask), dim=2))

        return x

class MaskGen(nn.Module):
    def __init__(self, 
                 dim, 
                 num_embeds_ada_norm=None,
                 use_timestep=False,
                 use_softmask=True,
                 is_q_mask=False,
                 input_channel=1536):
        
        super().__init__()

        self.dim = dim
        self.use_timestep = use_timestep
        self.use_softmask = use_softmask
        self.use_hardmask = not use_softmask
        self.is_q_mask = is_q_mask
        self.is_k_mask = not is_q_mask

        if self.use_timestep:
            self.norm = AdaLayerNorm(dim, num_embeds_ada_norm+1)

        self.MLP = MLP(input_channel)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x, t=None, mask_pre=None):
        # x: [B, N, C], t: [B]
        B, N, C = x.shape

        if self.use_timestep:
            x = rearrange(x, "B N C -> N B C")
            x = self.norm(x, t)
            x = rearrange(x, "N B C -> B N C")


        if self.use_softmask:
            mask = self.MLP(x, mask_pre)
            mask = torch.sigmoid(mask)
            mask = self.dropout(mask)

            # if mask_pre is not None:
            #     mask = torch.min(mask, mask_pre)

            vis_mask = mask
            if self.is_q_mask:
                mask = repeat(mask, "B N 1 -> B N C", C=C)
            elif self.is_k_mask:
                raise NotImplementedError

            # B N C, B N 1
            return mask, vis_mask
        else:
            mask = self.MLP(x)
            # mask = torch.sign(mask)
            raise NotImplementedError
        

@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        attention_mode: str = "xformers", 
        downsampler: str = None, 
        use_rope: bool = False, 
        interpolation_scale_thw: Tuple[int] = (1, 1, 1), 
        use_q_mask: bool = False,
        use_k_mask: bool = False,
        use_timestep: bool = False,
        use_softmask: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.downsampler = downsampler

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            attention_mode=attention_mode, 
            downsampler=downsampler, 
            use_rope=use_rope, 
            interpolation_scale_thw=interpolation_scale_thw, 
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                attention_mode=attention_mode,
                downsampler=False, 
                use_rope=False, 
                interpolation_scale_thw=interpolation_scale_thw, 
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self.use_q_mask = use_q_mask
        self.use_k_mask = use_k_mask
        self.use_timestep = use_timestep
        self.use_softmask = use_softmask
        # self.mgen = None

        if self.use_q_mask:
            self.mgen_q = MaskGen(dim=dim,
                                num_embeds_ada_norm=num_embeds_ada_norm,
                                use_timestep=self.use_timestep,
                                use_softmask=self.use_softmask,
                                is_q_mask=True,
                                input_channel=cross_attention_dim,
                                )
        else:
            self.mgen_q = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        timestep_ori: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
        mask_q_pre: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        assert self.norm_type == "ada_norm_single" and self.downsampler is None
        assert self.pos_embed is None
        assert self.attn2 is not None

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Begin
        batch_size, token_len, token_channel = hidden_states.shape

        # 1. Generate K-Mask. 0 for retrain, 1 for masked, and initialized to 0
        
        # 2. LayerNorm
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        # 3. Self-Attention with K-Mask
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=None, frame=frame, height=height, width=width, 
            **cross_attention_kwargs,
        )
        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Cross-Attention
        # For PixArt norm2 isn't applied
        norm_hidden_states = hidden_states

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

        # 5. Generate Q-Mask.
        if self.use_q_mask:
            mask_q, vis_mask_q = self.mgen_q(hidden_states, timestep_ori, mask_q_pre)
        else:
            mask_q = None
            vis_mask_q = None

        # 6. LayerNorm
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # 7. Feed-forward &  Use Q-Mask
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output
        
        if mask_q is not None:
            hidden_states = mask_q * ff_output + hidden_states
        else:
            hidden_states = ff_output + hidden_states


        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states, mask_q, vis_mask_q
