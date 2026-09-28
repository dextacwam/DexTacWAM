# This file is from Genie-Envisioner (AgibotTech) at commit d54425c4, whose
# README licenses everything outside models/ltx_models, models/cosmos_models,
# models/pipeline and web_infer_utils/openpi_client under CC BY-NC-SA 4.0.
# It is redistributed here under those terms: see LICENSES/CC-BY-NC-SA-4.0.txt.
# NonCommercial use only; adaptations must carry the same licence.

import torch
import torch.nn.functional as F
from einops import rearrange


def resize_traj_and_ray(traj_n_ray, mem_size, future_size, height, width):
    '''
    traj_n_ray: bv c t h w
    '''
    orig_t = traj_n_ray.shape[3]
    try:
        assert orig_t > (mem_size + future_size)
    except:
        breakpoint()
        
    n_view = traj_n_ray.shape[2]

    mem = traj_n_ray[:, :, :mem_size]
    mem = rearrange(mem, 'bv c t h w -> (bv t) c h w')
    mem = F.interpolate(mem, (height, width), mode='bilinear')
    mem = rearrange(mem, '(bv t) c h w -> bv c t h w', t=mem_size)

    future = traj_n_ray[:, :, mem_size:]  # bv c t h w
    future = F.interpolate(future, (future_size, height, width), mode='trilinear')

    out = torch.cat([mem, future], dim=2)
    return out
