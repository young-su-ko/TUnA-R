import torch


class MaskMaker:
    """Creates per-sequence and combined pairwise attention masks on GPU."""

    def __init__(self, device=None, max_len=512):
        self.device = device
        self.max_len = max_len

    def make_masks(
        self, lengths_A: list[int], lengths_B: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate masks for two protein sequences and their combined representations.
        Returns:
            mask_A: [B, 1, L, L]
            mask_B: [B, 1, L, L]
            combined_mask_AB: [B, 1, 2L, 2L] (A first, B second)
            combined_mask_BA: [B, 1, 2L, 2L] (B first, A second)
        """
        mask_A = self._make_sequence_mask(lengths_A, self.max_len)
        mask_B = self._make_sequence_mask(lengths_B, self.max_len)
        combined_mask_AB = self._make_pair_mask(mask_A, mask_B)
        combined_mask_BA = self._make_pair_mask(mask_B, mask_A)

        return (mask_A, mask_B, combined_mask_AB, combined_mask_BA)

    def _make_sequence_mask(self, lengths, max_len):
        lengths_tensor = torch.tensor(lengths, device=self.device)
        idx = torch.arange(max_len, device=self.device).unsqueeze(0)
        mask = idx < lengths_tensor.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        return mask & mask.transpose(-1, -2)  # symmetric mask

    def _make_pair_mask(self, mask1, mask2):
        bsz, _, L, _ = mask1.shape
        out = torch.zeros(bsz, 1, 2 * L, 2 * L, device=self.device)
        out[:, :, :L, :L] = mask1
        out[:, :, L:, L:] = mask2
        return out.bool()
