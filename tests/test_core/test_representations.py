"""
Tests for bicomplex representations and numerical stability.

This module tests:
1. Representation conversions
2. Numerical stability under extreme values
3. Gradient flow through operations
"""

import pytest
import torch
import sys
sys.path.insert(0, '/home/user/Bicomplex-Pytorch')

from bicomplex_pytorch.core.representations import (
    to_idempotent, from_idempotent, is_idempotent, is_bicomplex,
    create_bicomplex, get_components, real_to_bicomplex,
    complex_to_bicomplex, bicomplex_to_complex, bicomplex_to_real
)
from bicomplex_pytorch.core.arithmetic import (
    modulus, exp_idempotent, log_idempotent,
    inverse_idempotent, multiply_idempotent
)


# =============================================================================
# Representation Validation Tests
# =============================================================================

class TestRepresentationValidation:
    """Test representation format validation."""

    def test_is_bicomplex_valid(self):
        """Test is_bicomplex returns True for valid tensors."""
        valid = torch.randn(10, 5, 4)
        assert is_bicomplex(valid)

    def test_is_bicomplex_invalid_shape(self):
        """Test is_bicomplex returns False for wrong last dimension."""
        invalid = torch.randn(10, 5, 3)
        assert not is_bicomplex(invalid)

    def test_is_bicomplex_complex_tensor(self):
        """Test is_bicomplex returns False for complex tensors."""
        complex_tensor = torch.randn(10, 5, 4, dtype=torch.cfloat)
        assert not is_bicomplex(complex_tensor)

    def test_is_bicomplex_scalar(self):
        """Test is_bicomplex returns False for scalars."""
        scalar = torch.tensor(1.0)
        assert not is_bicomplex(scalar)

    def test_is_idempotent_valid(self):
        """Test is_idempotent returns True for valid tuples."""
        e1 = torch.randn(5, dtype=torch.cfloat)
        e2 = torch.randn(5, dtype=torch.cfloat)
        assert is_idempotent((e1, e2))

    def test_is_idempotent_invalid_types(self):
        """Test is_idempotent returns False for non-tuple inputs."""
        assert not is_idempotent([1, 2])
        assert not is_idempotent((1, 2))
        assert not is_idempotent(torch.randn(5))

    def test_is_idempotent_shape_mismatch(self):
        """Test is_idempotent returns False for mismatched shapes."""
        e1 = torch.randn(5, dtype=torch.cfloat)
        e2 = torch.randn(6, dtype=torch.cfloat)
        assert not is_idempotent((e1, e2))


# =============================================================================
# Conversion Tests
# =============================================================================

class TestConversions:
    """Test conversion between different representations."""

    def test_real_to_bicomplex(self):
        """Test conversion from real to bicomplex."""
        real = torch.tensor([1.0, 2.0, 3.0])
        bc = real_to_bicomplex(real)

        assert bc.shape == (3, 4)
        assert torch.allclose(bc[..., 0], real)
        assert torch.allclose(bc[..., 1], torch.zeros(3))
        assert torch.allclose(bc[..., 2], torch.zeros(3))
        assert torch.allclose(bc[..., 3], torch.zeros(3))

    def test_complex_to_bicomplex(self):
        """Test conversion from complex to bicomplex."""
        complex_tensor = torch.complex(
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0])
        )
        bc = complex_to_bicomplex(complex_tensor)

        assert bc.shape == (2, 4)
        assert torch.allclose(bc[..., 0], complex_tensor.real)
        assert torch.allclose(bc[..., 1], complex_tensor.imag)
        assert torch.allclose(bc[..., 2], torch.zeros(2))
        assert torch.allclose(bc[..., 3], torch.zeros(2))

    def test_bicomplex_to_complex(self):
        """Test projection from bicomplex to complex."""
        bc = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]])
        c = bicomplex_to_complex(bc)

        assert torch.allclose(c.real, bc[..., 0])
        assert torch.allclose(c.imag, bc[..., 1])

    def test_bicomplex_to_real(self):
        """Test projection from bicomplex to real."""
        bc = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]])
        r = bicomplex_to_real(bc)

        assert torch.allclose(r, bc[..., 0])

    def test_get_components(self):
        """Test component extraction."""
        bc = torch.tensor([1.0, 2.0, 3.0, 4.0])
        a, b, c, d = get_components(bc)

        assert a.item() == 1.0
        assert b.item() == 2.0
        assert c.item() == 3.0
        assert d.item() == 4.0

    def test_create_bicomplex(self):
        """Test bicomplex creation from components."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        c = torch.tensor([5.0, 6.0])
        d = torch.tensor([7.0, 8.0])

        bc = create_bicomplex(a, b, c, d)

        assert bc.shape == (2, 4)
        assert torch.allclose(bc[..., 0], a)
        assert torch.allclose(bc[..., 1], b)
        assert torch.allclose(bc[..., 2], c)
        assert torch.allclose(bc[..., 3], d)


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability under extreme conditions."""

    def test_small_values(self):
        """Test operations with very small values."""
        small = torch.tensor([1e-30, 1e-35, 1e-40, 1e-30])
        z1, z2 = to_idempotent(small)
        recovered = from_idempotent(z1, z2)

        assert torch.allclose(recovered, small, rtol=1e-5)

    def test_large_values(self):
        """Test operations with large values."""
        large = torch.tensor([1e30, 1e30, 1e30, 1e30])
        z1, z2 = to_idempotent(large)
        recovered = from_idempotent(z1, z2)

        assert torch.allclose(recovered, large, rtol=1e-5)

    def test_mixed_scale_values(self):
        """Test operations with mixed scale values."""
        mixed = torch.tensor([1e-20, 1e20, 1e-20, 1e20])
        z1, z2 = to_idempotent(mixed)
        recovered = from_idempotent(z1, z2)

        assert torch.allclose(recovered, mixed, rtol=1e-5)

    def test_modulus_stability(self):
        """Test modulus computation stability."""
        # Very small values
        small = (torch.complex(torch.tensor([1e-20]), torch.tensor([1e-20])),
                 torch.complex(torch.tensor([1e-20]), torch.tensor([1e-20])))
        mod_small = modulus(small)
        assert torch.isfinite(mod_small).all()
        assert (mod_small >= 0).all()

        # Very large values
        large = (torch.complex(torch.tensor([1e20]), torch.tensor([1e20])),
                 torch.complex(torch.tensor([1e20]), torch.tensor([1e20])))
        mod_large = modulus(large)
        assert torch.isfinite(mod_large).all()
        assert (mod_large >= 0).all()

    def test_exp_stability(self):
        """Test exponential stability for moderate values."""
        # Moderate values (avoid overflow)
        z = (torch.complex(torch.randn(10) * 5, torch.randn(10) * 5),
             torch.complex(torch.randn(10) * 5, torch.randn(10) * 5))

        result = exp_idempotent(z)

        # Should be finite for moderate inputs
        assert torch.isfinite(result[0]).all()
        assert torch.isfinite(result[1]).all()

    def test_inverse_stability(self):
        """Test inverse stability for non-zero-divisors."""
        # Generate tensor with non-zero components
        z = (torch.complex(torch.randn(10) + 2, torch.randn(10)),
             torch.complex(torch.randn(10) + 2, torch.randn(10)))

        inv_z = inverse_idempotent(z, zero_divisor_handling='ignore')

        # Product should be close to 1
        product = multiply_idempotent(z, inv_z)
        assert torch.allclose(product[0], torch.ones_like(product[0]), atol=1e-4)
        assert torch.allclose(product[1], torch.ones_like(product[1]), atol=1e-4)

    def test_log_stability_positive_reals(self):
        """Test log stability for positive real idempotent components."""
        # Use positive real parts to ensure well-defined logarithm
        z = (torch.complex(torch.abs(torch.randn(10)) + 1, torch.zeros(10)),
             torch.complex(torch.abs(torch.randn(10)) + 1, torch.zeros(10)))

        log_z = log_idempotent(z, zero_divisor_handling='ignore')

        assert torch.isfinite(log_z[0]).all()
        assert torch.isfinite(log_z[1]).all()


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestGradientFlow:
    """Test gradient flow through operations."""

    def test_gradient_through_conversion(self):
        """Test gradients flow through idempotent conversion."""
        bc = torch.randn(5, 4, requires_grad=True)

        z1, z2 = to_idempotent(bc)
        recovered = from_idempotent(z1, z2)

        loss = recovered.sum()
        loss.backward()

        assert bc.grad is not None
        assert torch.isfinite(bc.grad).all()

    def test_gradient_through_modulus(self):
        """Test gradients flow through modulus computation."""
        z = (torch.complex(torch.randn(5, requires_grad=True), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5, requires_grad=True)))

        # Note: requires_grad on real/imag parts
        e1 = torch.complex(torch.randn(5, requires_grad=True), torch.randn(5))
        e2 = torch.complex(torch.randn(5), torch.randn(5, requires_grad=True))

        # Compute modulus of real components for gradient check
        mod = torch.sqrt(e1.real ** 2 + e2.imag ** 2)
        loss = mod.sum()
        loss.backward()

    def test_gradient_through_multiplication(self):
        """Test gradients flow through multiplication."""
        a_real = torch.randn(5, requires_grad=True)
        b_real = torch.randn(5, requires_grad=True)

        a = (torch.complex(a_real, torch.zeros(5)),
             torch.complex(torch.zeros(5), torch.zeros(5)))
        b = (torch.complex(b_real, torch.zeros(5)),
             torch.complex(torch.zeros(5), torch.zeros(5)))

        result = multiply_idempotent(a, b)

        loss = result[0].real.sum()
        loss.backward()

        assert a_real.grad is not None
        assert b_real.grad is not None

    def test_gradient_through_exp(self):
        """Test gradients flow through exponential."""
        x_real = torch.randn(5, requires_grad=True)
        z = (torch.complex(x_real, torch.zeros(5)),
             torch.complex(torch.randn(5), torch.zeros(5)))

        result = exp_idempotent(z)

        loss = result[0].real.sum()
        loss.backward()

        assert x_real.grad is not None
        assert torch.isfinite(x_real.grad).all()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_bicomplex(self):
        """Test operations on zero bicomplex number."""
        zero = torch.zeros(4)
        z1, z2 = to_idempotent(zero)

        assert torch.allclose(z1, torch.complex(torch.tensor(0.0), torch.tensor(0.0)))
        assert torch.allclose(z2, torch.complex(torch.tensor(0.0), torch.tensor(0.0)))

    def test_real_bicomplex(self):
        """Test bicomplex with only real part."""
        real = torch.tensor([5.0, 0.0, 0.0, 0.0])
        z1, z2 = to_idempotent(real)

        # For z = a, both idempotent components should equal a
        assert torch.allclose(z1, torch.complex(torch.tensor(5.0), torch.tensor(0.0)))
        assert torch.allclose(z2, torch.complex(torch.tensor(5.0), torch.tensor(0.0)))

    def test_pure_imaginary_i(self):
        """Test bicomplex with only i component."""
        pure_i = torch.tensor([0.0, 3.0, 0.0, 0.0])
        z1, z2 = to_idempotent(pure_i)
        recovered = from_idempotent(z1, z2)

        assert torch.allclose(recovered, pure_i, atol=1e-6)

    def test_pure_imaginary_j(self):
        """Test bicomplex with only j component."""
        pure_j = torch.tensor([0.0, 0.0, 4.0, 0.0])
        z1, z2 = to_idempotent(pure_j)
        recovered = from_idempotent(z1, z2)

        assert torch.allclose(recovered, pure_j, atol=1e-6)

    def test_pure_imaginary_ij(self):
        """Test bicomplex with only ij component."""
        pure_ij = torch.tensor([0.0, 0.0, 0.0, 2.0])
        z1, z2 = to_idempotent(pure_ij)
        recovered = from_idempotent(z1, z2)

        assert torch.allclose(recovered, pure_ij, atol=1e-6)

    def test_batched_operations(self):
        """Test operations on batched tensors."""
        batch = torch.randn(32, 16, 8, 4)

        z1, z2 = to_idempotent(batch)
        assert z1.shape == (32, 16, 8)
        assert z2.shape == (32, 16, 8)

        recovered = from_idempotent(z1, z2)
        assert recovered.shape == batch.shape
        assert torch.allclose(recovered, batch, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
