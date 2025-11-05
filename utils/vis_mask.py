from einops import rearrange, repeat
import torch
from torchvision.utils import make_grid

def vis_mask_t(mask_t, expand=False, latent_height=16):
    """
    mask_t: [mask_block_0, ..., mask_block_31]
    """
    mask_all = []
    # mask_pre_block = None
    for mask in mask_t:
        # [negative, positive]
        # 0 for masked, 1 for retain
        mask = repeat(mask, 'b n c -> n b (c cc)', cc=3)
        mask = rearrange(mask, '(h w) b c -> b c h w', h=latent_height)
        mask = mask[-1] # TODO b/2

        if mask is not None:
            mask_all.append(mask)
            
    # 28, 3*15*20
    nrows = len(mask_all)
    mask_all = make_grid(mask_all, nrow=nrows, padding=2) if len(mask_all) != 0 else None

    return mask_all

def vis_mask(mask_all, latent_height):
    """
    mask_all: a list of both q and k masks, 
                [mask_1, mask_2, ..., mask_100]
                mask_i = [mask]
    """
    q_mask_all_vis = []
    k_mask_all_vis = []
    for mask_all_t in mask_all:
        q_mask_t, k_mask_t = mask_all_t
        if len(q_mask_t) !=0 and q_mask_t[0] is not None:
            q_mask_t = vis_mask_t(q_mask_t, expand=False, latent_height=latent_height)
            q_mask_all_vis.append(q_mask_t)

        if len(k_mask_t) !=0 and k_mask_t[0] is not None:
            k_mask_t = vis_mask_t(k_mask_t, expand=True, latent_height=latent_height)
            k_mask_all_vis.append(k_mask_t)
  

    if len(q_mask_all_vis) != 0:
        q_mask_all_vis = torch.stack(q_mask_all_vis)
        q_mask_all_vis = make_grid(q_mask_all_vis, nrow=1)

    else:
        q_mask_all_vis = None
    
    if len(k_mask_all_vis) != 0:
        k_mask_all_vis = torch.stack(k_mask_all_vis)
        k_mask_all_vis = make_grid(k_mask_all_vis, nrow=1)
    else:
        k_mask_all_vis = None

    return q_mask_all_vis, k_mask_all_vis
    
