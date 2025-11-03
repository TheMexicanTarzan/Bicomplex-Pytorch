"""
Tests for bicomplex linear layers.
"""
import pytest
import torch
from bicomplex_pytorch import BiComplexLinear, to_idempotent, from_idempotent


class TestBiComplexLinear:
    """Test suite for BiComplexLinear layer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size, in_feat, out_feat = 32, 10, 20

        # Create bicomplex data (batch, features, 4)
        x = torch.randn(batch_size, in_feat, 4)

        layer = BiComplexLinear(in_feat, out_feat)
        output = layer(x)

        assert output.shape == (batch_size, out_feat, 4), \
            f"Expected shape {(batch_size, out_feat, 4)}, got {output.shape}"

    def test_backward_pass(self):
        """Test that backward pass computes gradients correctly."""
        x = torch.randn(16, 5, 4, requires_grad=True)
        layer = BiComplexLinear(5, 10)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input gradients not computed"
        assert layer.branch1.weight.grad is not None, \
            "Branch1 weight gradients not computed"
        assert layer.branch2.weight.grad is not None, \
            "Branch2 weight gradients not computed"

    @pytest.mark.parametrize("shared_weights", [True, False])
    def test_shared_vs_independent_weights(self, shared_weights):
        """Test both shared and independent weight configurations."""
        layer = BiComplexLinear(5, 10, shared_weights=shared_weights)
        x = torch.randn(8, 5, 4)
        output = layer(x)

        assert output.shape == (8, 10, 4)

        if shared_weights:
            assert hasattr(layer, 'complex_layer'), \
                "Shared weights config should have complex_layer"
            assert not hasattr(layer, 'branch2'), \
                "Shared weights config should not have branch2"
        else:
            assert hasattr(layer, 'branch1') and hasattr(layer, 'branch2'), \
                "Independent weights config should have both branches"

    def test_bias_option(self):
        """Test layer with and without bias."""
        x = torch.randn(4, 5, 4)

        # With bias (default)
        layer_with_bias = BiComplexLinear(5, 10, bias=True)
        out1 = layer_with_bias(x)

        # Without bias
        layer_no_bias = BiComplexLinear(5, 10, bias=False)
        out2 = layer_no_bias(x)

        assert out1.shape == out2.shape == (4, 10, 4)

    def test_idempotent_round_trip(self):
        """Test that idempotent conversion is reversible."""
        x = torch.randn(16, 8, 4)

        # Convert to idempotent and back
        z1, z2 = to_idempotent(x)
        x_reconstructed = from_idempotent(z1, z2)

        torch.testing.assert_close(x, x_reconstructed, rtol=1e-5, atol=1e-7)

    def test_gradient_flow(self):
        """Test that gradients flow through the entire network."""
        model = torch.nn.Sequential(
            BiComplexLinear(5, 10),
            BiComplexLinear(10, 3)
        )

        x = torch.randn(8, 5, 4, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that all parameters have gradients
        for param in model.parameters():
            assert param.grad is not None, "Parameter missing gradient"
            assert not torch.isnan(param.grad).any(), "NaN in gradients"

    def test_reproducibility(self):
        """Test that forward pass is deterministic with same seed."""
        torch.manual_seed(42)
        x = torch.randn(4, 5, 4)
        layer = BiComplexLinear(5, 10)

        torch.manual_seed(42)
        out1 = layer(x)

        torch.manual_seed(42)
        out2 = layer(x)

        torch.testing.assert_close(out1, out2)


@pytest.fixture
def sample_bicomplex_data():
    """Fixture providing sample bicomplex data for tests."""
    return torch.randn(32, 10, 4)


@pytest.fixture
def sample_layer():
    """Fixture providing a sample BiComplexLinear layer."""
    return BiComplexLinear(10, 20)