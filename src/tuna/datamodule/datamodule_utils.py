import torch

# TODO: add docstrings in numpy format


def make_masks(lens: list[int], max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len)[None, :]  # Shape: [1, L]
    lens = torch.tensor(lens)[:, None]  # Shape: [B, 1]
    mask = idx < lens  # Shape: [B, L]
    mask = mask.unsqueeze(1)  # Shape: [B, 1, L]
    mask = mask.unsqueeze(-1)  # Shape: [B, 1, L, 1]
    mask = mask & mask.transpose(-1, -2)  # Shape: [B, 1, L, L]
    return mask


def pad_batch(prots: list[torch.Tensor], max_len: int) -> torch.Tensor:
    """Pad a batch of protein tensors to max_len using vectorized operations.

    Args:
        prots: List of protein tensors, each of shape [len_i, dim] where len_i varies
        max_len: Length to pad to

    Returns:
        Padded and stacked tensor of shape [batch_size, max_len, dim]
    """
    batch_size = len(prots)
    dim = prots[0].shape[1]

    # Create output tensor
    out = torch.zeros(batch_size, max_len, dim)

    # Get lengths of all sequences
    lengths = torch.tensor([p.shape[0] for p in prots])

    # Create mask for sequences longer than max_len
    longer_mask = lengths > max_len
    shorter_mask = ~longer_mask

    # Handle shorter sequences (vectorized)
    if shorter_mask.any():
        shorter_prots = [p for i, p in enumerate(prots) if shorter_mask[i]]
        shorter_idx = torch.where(shorter_mask)[0]
        padded = torch.nn.utils.rnn.pad_sequence(shorter_prots, batch_first=True)
        out[shorter_idx, : padded.shape[1]] = padded

    # Handle longer sequences (vectorized where possible)
    if longer_mask.any():
        longer_prots = [p for i, p in enumerate(prots) if longer_mask[i]]
        longer_idx = torch.where(longer_mask)[0]
        # Generate random starts for all longer sequences at once
        starts = torch.randint(
            0,
            torch.tensor([p.shape[0] - max_len for p in longer_prots]),
            (len(longer_prots),),
        )
        # Extract slices
        sliced = torch.stack(
            [p[start : start + max_len] for p, start in zip(longer_prots, starts)]
        )
        out[longer_idx] = sliced

    return out


def combine_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    bsz, _, L, _ = mask1.shape
    out = torch.zeros(bsz, 1, 2 * L, 2 * L)
    out[:, :, :L, :L] = mask1
    out[:, :, L:, L:] = mask2
    return out.bool()
