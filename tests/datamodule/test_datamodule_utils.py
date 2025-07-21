import pytest
import torch

from tuna.datamodule.datamodule_utils import (
    combine_masks,
    make_masks,
    pad_batch,
    resolve_embedding_type,
)


class TestMakeMask:
    @pytest.mark.parametrize(
        "lens, max_len, expected",
        [
            (
                [2, 2],
                2,
                torch.tensor(
                    [[[[True, True], [True, True]]], [[[True, True], [True, True]]]]
                ),
            ),
            (
                [1, 2],
                2,
                torch.tensor(
                    [[[[True, False], [False, False]]], [[[True, True], [True, True]]]]
                ),
            ),
            (
                [0, 1],
                2,
                torch.tensor(
                    [
                        [[[False, False], [False, False]]],
                        [[[True, False], [False, False]]],
                    ]
                ),
            ),
        ],
    )
    def test_make_masks_cases(self, lens, max_len, expected):
        result = make_masks(lens, max_len)
        assert result.dtype == torch.bool
        assert torch.equal(result, expected)

    def test_combine_masks(self):
        # Create sample masks
        mask1 = torch.tensor([[[[True, True], [True, True]]]])  # Shape: [1, 1, 2, 2]
        mask2 = torch.tensor([[[[True, False], [False, False]]]])  # Shape: [1, 1, 2, 2]

        result = combine_masks(mask1, mask2)

        # Check shape
        assert result.shape == (1, 1, 4, 4)

        # Check values
        expected = torch.tensor(
            [
                [
                    [
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, True, False],
                        [False, False, False, False],
                    ]
                ]
            ]
        )
        assert torch.equal(result, expected)


class TestPadBatch:
    def test_pad_batch(self):
        # Create sample input tensors
        dim = 3
        prot1 = torch.ones(2, dim)  # length 2
        prot2 = torch.ones(3, dim)  # length 3
        prots = [prot1, prot2]
        # Test case 1: max_len larger than all sequences
        max_len = 4
        result = pad_batch(prots, max_len)
        assert result.shape == (2, max_len, dim)
        assert torch.equal(result[0, :2], torch.ones(2, dim))
        assert torch.equal(result[0, 2:], torch.zeros(2, dim))
        assert torch.equal(result[1, :3], torch.ones(3, dim))
        assert torch.equal(result[1, 3:], torch.zeros(1, dim))

        # Test case 2: max_len smaller than some sequences
        max_len = 2
        result = pad_batch(prots, max_len)
        assert result.shape == (2, max_len, dim)
        assert torch.equal(result[0], torch.ones(max_len, dim))
        # For longer sequence, it should be a continuous slice of length max_len
        assert result[1].sum() == max_len * dim


class TestResolveEmbeddingType:
    @pytest.mark.parametrize(
        "model, expected",
        [
            ("tuna", "residue"),
            ("tfc", "residue"),
            ("esm_gp", "protein"),
            ("esm_mlp", "protein"),
        ],
    )
    def test_resolve_embedding_type(self, model, expected):
        assert resolve_embedding_type(model) == expected

    def test_resolve_embedding_type_invalid(self):
        with pytest.raises(ValueError):
            resolve_embedding_type("invalid_model")
