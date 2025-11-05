import torch


def make_masks(
    lengths_A: list[int],
    lengths_B: list[int],
    pad_len_A: int,
    pad_len_B: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create attention masks for protein pairs.

    Args:
        lengths_A: List of actual lengths for protein A sequences
        lengths_B: List of actual lengths for protein B sequences
        pad_len_A: Padded length for protein A sequences
        pad_len_B: Padded length for protein B sequences
        device: Device to create masks on. If None, will use 'cpu'.
               Typically pass proteinA.device or proteinB.device to match your tensors.

    Returns:
        Tuple of (mask_A, mask_B, combined_mask_AB, combined_mask_BA)
    """
    if device is None:
        device = torch.device("cpu")

    mask_A = _make_single_mask(lengths_A, pad_len_A, device)
    mask_B = _make_single_mask(lengths_B, pad_len_B, device)
    combined_mask_AB = _make_pair_mask(mask_A, mask_B, device)
    combined_mask_BA = _make_pair_mask(mask_B, mask_A, device)

    return (mask_A, mask_B, combined_mask_AB, combined_mask_BA)


def _make_single_mask(
    lengths: list[int], max_sequence_length: int, device: torch.device
) -> torch.Tensor:
    """Create a symmetric sequence mask for a single protein."""
    lengths_tensor = torch.tensor(lengths, device=device)
    idx = torch.arange(max_sequence_length, device=device).unsqueeze(0)
    mask = idx < lengths_tensor.unsqueeze(1)
    mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
    return mask & mask.transpose(-1, -2)  # symmetric mask


def _make_pair_mask(
    mask_A: torch.Tensor, mask_B: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Combine two sequence masks into a pairwise mask."""
    lenA, lenB = mask_A.size(2), mask_B.size(2)
    combined_mask = torch.zeros(
        mask_A.size(0), 1, lenA + lenB, lenA + lenB, device=device
    )
    combined_mask[:, :, :lenA, :lenA] = mask_A
    combined_mask[:, :, lenA:, lenA:] = mask_B
    return combined_mask
