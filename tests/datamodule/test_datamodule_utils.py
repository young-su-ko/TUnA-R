import torch

from tuna.datamodule.datamodule_utils import pad_batch


def test_pad_batch():
    """Smoke test for pad_batch function."""
    prots = [torch.ones(2, 3), torch.ones(3, 3)]
    result = pad_batch(prots, max_len=4)
    assert result.shape == (2, 4, 3)
    assert torch.equal(result[0, :2], torch.ones(2, 3))
    assert torch.equal(result[0, 2:], torch.zeros(2, 3))
    assert torch.equal(result[1, :3], torch.ones(3, 3))
