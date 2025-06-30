import torch
import random

def make_masks(lens, max_len, device):
    idx = torch.arange(max_len, device=device)[None, :]
    lens = torch.tensor(lens, device=device)[:, None]
    mask = (idx < lens).unsqueeze(1)  # Shape: [B, 1, L]
    return mask & mask.transpose(-1, -2)  # Shape: [B, 1, L, L]

def pad_batch(prots, max_len, device):
    bsz, _, dim = prots.size()
    out = torch.zeros(bsz, max_len, dim, device=device)
    for i, p in enumerate(prots):
        L = p.size(0)
        if L <= max_len:
            out[i, :L] = p
        else:
            start = random.randint(0, L - max_len)
            out[i] = p[start:start + max_len]
    return out

def combine_masks(mask1, mask2, device):
    bsz, _, L = mask1.size()
    out = torch.zeros(bsz, 1, 2 * L, 2 * L, device=device)
    out[:, :, :L, :L] = mask1
    out[:, :, L:, L:] = mask2
    return out.bool()


