import torch


def make_masks(lens: list[int], max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len).unsqueeze(0)  # Shape: [1, L]
    lens = torch.tensor(lens).unsqueeze(1)  # Shape: [B, 1]
    mask = idx < lens  # Shape: [B, L]
    mask = mask.unsqueeze(1)  # Shape: [B, 1, L]
    mask = mask.unsqueeze(-1)  # Shape: [B, 1, L, 1]
    mask = mask & mask.transpose(-1, -2)  # Shape: [B, 1, L, L]
    return mask.bool()


def pad_batch(prots: list[torch.Tensor], max_len: int) -> torch.Tensor:
    batch_size = len(prots)
    dim = prots[0].size(1)
    out = torch.zeros(batch_size, max_len, dim, device=prots[0].device)

    for index, protein in enumerate(prots):
        length = protein.size(0)
        if length < max_len:
            out[index, :length] = protein  # no modification, just paste it in
        elif length > max_len:
            # If longer, select a random contiguous chunk of max_len
            start = torch.randint(0, length - max_len + 1, (1,)).item()
            out[index] = protein[start : start + max_len]
        else:  # length == max_len
            out[index] = protein

    return out


def combine_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    bsz, _, L, _ = mask1.shape
    out = torch.zeros(bsz, 1, 2 * L, 2 * L, device=mask1.device)
    out[:, :, :L, :L] = mask1
    out[:, :, L:, L:] = mask2
    return out.bool()
