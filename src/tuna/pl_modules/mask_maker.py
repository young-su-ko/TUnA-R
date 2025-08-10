import torch


class MaskMaker:
    """Creates per-sequence and combined pairwise attention masks on GPU."""

    def __init__(self, device: str = None):
        self.device = device

    def make_masks(
        self, lengths_A: list[int], lengths_B: list[int], pad_len_A: int, pad_len_B: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_A = self._make_sequence_mask(lengths_A, pad_len_A)
        mask_B = self._make_sequence_mask(lengths_B, pad_len_B)
        combined_mask_AB = self._make_pair_mask(mask_A, mask_B)
        combined_mask_BA = self._make_pair_mask(mask_B, mask_A)

        return (mask_A, mask_B, combined_mask_AB, combined_mask_BA)

    def _make_sequence_mask(
        self, lengths: list[int], max_sequence_length: int
    ) -> torch.Tensor:
        lengths_tensor = torch.tensor(lengths, device=self.device)
        idx = torch.arange(max_sequence_length, device=self.device).unsqueeze(0)
        mask = idx < lengths_tensor.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        return mask & mask.transpose(-1, -2)  # symmetric mask

    def _make_pair_mask(self, maskA, maskB):
        lenA, lenB = maskA.size(2), maskB.size(2)
        combined_mask = torch.zeros(
            maskA.size(0), 1, lenA + lenB, lenA + lenB, device=self.device
        )
        combined_mask[:, :, :lenA, :lenA] = maskA
        combined_mask[:, :, lenA:, lenA:] = maskB
        return combined_mask
