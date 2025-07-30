import torch


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
