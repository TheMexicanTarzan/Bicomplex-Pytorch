"""
Tests for bicomplex linear layers.
"""
import pytest
import torch
import torch.nn.functional as F
from bicomplex_pytorch import BiComplexLinear, to_idempotent, from_idempotent
from bicomplex_pytorch.nn.modules.linear import _apply_complex


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
        assert layer.weight1_r.grad is not None, \
            "weight1_r gradients not computed"
        assert layer.weight2_r.grad is not None, \
            "weight2_r gradients not computed"

    @pytest.mark.parametrize("shared_weights", [True, False])
    def test_shared_vs_independent_weights(self, shared_weights):
        """Test both shared and independent weight configurations."""
        layer = BiComplexLinear(5, 10, shared_weights=shared_weights)
        x = torch.randn(8, 5, 4)
        output = layer(x)

        assert output.shape == (8, 10, 4)

        if shared_weights:
            assert hasattr(layer, 'weight1_r'), \
                "Shared weights config should have weight1_r"
            assert not hasattr(layer, 'weight2_r'), \
                "Shared weights config should not have weight2_r"
        else:
            assert hasattr(layer, 'weight1_r') and hasattr(layer, 'weight2_r'), \
                "Independent weights config should have both weight1_r and weight2_r"

    def test_shared_weights_produces_coupled_output(self):
        """Test that shared weights apply the same transform to both branches."""
        layer = BiComplexLinear(5, 10, shared_weights=True)
        x = torch.randn(4, 5, 4)

        # Get idempotent components of input
        z1, z2 = to_idempotent(x)

        # Manually apply the shared weight to both using apply_complex
        expected_out1 = _apply_complex(z1, layer.weight1_r, layer.weight1_i,
                                       layer.bias1_r, layer.bias1_i)
        expected_out2 = _apply_complex(z2, layer.weight1_r, layer.weight1_i,
                                       layer.bias1_r, layer.bias1_i)

        # Compare with layer output in idempotent form
        layer_idem = BiComplexLinear(5, 10, shared_weights=True, output_format='idempotent')
        # Copy weights
        with torch.no_grad():
            layer_idem.weight1_r.copy_(layer.weight1_r)
            layer_idem.weight1_i.copy_(layer.weight1_i)
            layer_idem.bias1_r.copy_(layer.bias1_r)
            layer_idem.bias1_i.copy_(layer.bias1_i)
        actual_out1, actual_out2 = layer_idem(x)

        torch.testing.assert_close(expected_out1, actual_out1)
        torch.testing.assert_close(expected_out2, actual_out2)

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

        # Verify no bias parameters exist
        assert layer_no_bias.bias1_r is None
        assert layer_no_bias.bias2_r is None

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

    def test_idempotent_input_output_formats(self):
        """Test idempotent input/output format options."""
        x = torch.randn(4, 5, 4)
        z1, z2 = to_idempotent(x)

        # Idempotent input, standard output
        layer1 = BiComplexLinear(5, 10, input_format='idempotent')
        out1 = layer1((z1, z2))
        assert isinstance(out1, torch.Tensor) and out1.shape[-1] == 4

        # Standard input, idempotent output
        layer2 = BiComplexLinear(5, 10, output_format='idempotent')
        out2 = layer2(x)
        assert isinstance(out2, tuple) and len(out2) == 2


@pytest.fixture
def sample_bicomplex_data():
    """Fixture providing sample bicomplex data for tests."""
    return torch.randn(32, 10, 4)


@pytest.fixture
def sample_layer():
    """Fixture providing a sample BiComplexLinear layer."""
    return BiComplexLinear(10, 20)
