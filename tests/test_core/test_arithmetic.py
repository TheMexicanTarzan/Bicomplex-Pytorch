"""
Tests for bicomplex arithmetic operations.

This module tests algebraic properties and correctness of bicomplex operations
including zero divisor handling, conjugates, modulus, and basic arithmetic.
"""

import pytest
import torch
import sys
sys.path.insert(0, '/home/user/Bicomplex-Pytorch')

from bicomplex_pytorch.core.representations import (
    to_idempotent, from_idempotent, is_idempotent, is_bicomplex,
    create_bicomplex, get_components
)
from bicomplex_pytorch.core.arithmetic import (
    add_idempotent, multiply_idempotent, subtract_idempotent,
    inverse_idempotent, divide_idempotent, conjugate_idempotent,
    modulus, modulus_squared, d_modulus,
    exp_idempotent, log_idempotent, power_idempotent,
    is_zero_divisor, has_zero_divisor, regularize_zero_divisors,
    ZeroDivisorError
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_bicomplex():
    """Create a sample bicomplex tensor in standard form."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def batch_bicomplex():
    """Create a batch of bicomplex tensors."""
    return torch.randn(10, 5, 4)


@pytest.fixture
def zero_divisor_tensor():
    """Create a tensor containing zero divisors."""
    e1 = torch.complex(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]))
    e2 = torch.complex(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]))
    return (e1, e2)


# =============================================================================
# Idempotent Roundtrip Tests
# =============================================================================

class TestIdempotentRoundtrip:
    """Test roundtrip conversion between standard and idempotent forms."""

    def test_roundtrip_single(self, sample_bicomplex):
        """Test roundtrip for single bicomplex number."""
        z1, z2 = to_idempotent(sample_bicomplex)
        recovered = from_idempotent(z1, z2)
        assert torch.allclose(recovered, sample_bicomplex, atol=1e-6)

    def test_roundtrip_batch(self, batch_bicomplex):
        """Test roundtrip for batch of bicomplex numbers."""
        z1, z2 = to_idempotent(batch_bicomplex)
        recovered = from_idempotent(z1, z2)
        assert torch.allclose(recovered, batch_bicomplex, atol=1e-6)

    def test_roundtrip_random(self):
        """Test roundtrip for random bicomplex tensors."""
        for _ in range(10):
            original = torch.randn(32, 16, 4)
            z1, z2 = to_idempotent(original)
            recovered = from_idempotent(z1, z2)
            assert torch.allclose(recovered, original, atol=1e-5)


# =============================================================================
# Addition Tests
# =============================================================================

class TestAddition:
    """Test bicomplex addition properties."""

    def test_commutativity(self):
        """Test a + b = b + a."""
        a = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        b = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        ab = add_idempotent(a, b)
        ba = add_idempotent(b, a)

        assert torch.allclose(ab[0], ba[0], atol=1e-6)
        assert torch.allclose(ab[1], ba[1], atol=1e-6)

    def test_associativity(self):
        """Test (a + b) + c = a + (b + c)."""
        a = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        b = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        c = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        ab_c = add_idempotent(add_idempotent(a, b), c)
        a_bc = add_idempotent(a, add_idempotent(b, c))

        assert torch.allclose(ab_c[0], a_bc[0], atol=1e-5)
        assert torch.allclose(ab_c[1], a_bc[1], atol=1e-5)

    def test_identity(self):
        """Test a + 0 = a."""
        a = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        zero = (torch.zeros_like(a[0]), torch.zeros_like(a[1]))

        result = add_idempotent(a, zero)

        assert torch.allclose(result[0], a[0], atol=1e-6)
        assert torch.allclose(result[1], a[1], atol=1e-6)


# =============================================================================
# Multiplication Tests
# =============================================================================

class TestMultiplication:
    """Test bicomplex multiplication properties."""

    def test_commutativity(self):
        """Test a * b = b * a."""
        a = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        b = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        ab = multiply_idempotent(a, b)
        ba = multiply_idempotent(b, a)

        assert torch.allclose(ab[0], ba[0], atol=1e-5)
        assert torch.allclose(ab[1], ba[1], atol=1e-5)

    def test_associativity(self):
        """Test (a * b) * c = a * (b * c)."""
        a = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        b = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        c = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        ab_c = multiply_idempotent(multiply_idempotent(a, b), c)
        a_bc = multiply_idempotent(a, multiply_idempotent(b, c))

        assert torch.allclose(ab_c[0], a_bc[0], atol=1e-4)
        assert torch.allclose(ab_c[1], a_bc[1], atol=1e-4)

    def test_distributivity(self):
        """Test a * (b + c) = a*b + a*c."""
        a = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        b = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        c = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        lhs = multiply_idempotent(a, add_idempotent(b, c))
        rhs = add_idempotent(multiply_idempotent(a, b), multiply_idempotent(a, c))

        assert torch.allclose(lhs[0], rhs[0], atol=1e-4)
        assert torch.allclose(lhs[1], rhs[1], atol=1e-4)

    def test_identity(self):
        """Test a * 1 = a."""
        a = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))
        one = (torch.ones_like(a[0]), torch.ones_like(a[1]))

        result = multiply_idempotent(a, one)

        assert torch.allclose(result[0], a[0], atol=1e-6)
        assert torch.allclose(result[1], a[1], atol=1e-6)


# =============================================================================
# Zero Divisor Tests
# =============================================================================

class TestZeroDivisors:
    """Test zero divisor detection and handling."""

    def test_detect_zero_divisor(self, zero_divisor_tensor):
        """Test detection of zero divisors."""
        result = is_zero_divisor(zero_divisor_tensor)
        assert result[0].item() == True
        assert result[1].item() == False

    def test_has_zero_divisor(self, zero_divisor_tensor):
        """Test has_zero_divisor function."""
        assert has_zero_divisor(zero_divisor_tensor) == True

        normal = (torch.complex(torch.ones(3), torch.zeros(3)),
                  torch.complex(torch.ones(3), torch.zeros(3)))
        assert has_zero_divisor(normal) == False

    def test_inverse_raises_on_zero_divisor(self, zero_divisor_tensor):
        """Test that inverse raises ZeroDivisorError by default."""
        with pytest.raises(ZeroDivisorError):
            inverse_idempotent(zero_divisor_tensor)

    def test_inverse_with_regularization(self, zero_divisor_tensor):
        """Test that inverse works with regularization."""
        result = inverse_idempotent(
            zero_divisor_tensor,
            zero_divisor_handling='regularize'
        )
        assert torch.isfinite(result[0]).all()
        assert torch.isfinite(result[1]).all()

    def test_regularize_zero_divisors(self, zero_divisor_tensor):
        """Test zero divisor regularization."""
        regularized = regularize_zero_divisors(zero_divisor_tensor)
        assert not has_zero_divisor(regularized)


# =============================================================================
# Inverse and Division Tests
# =============================================================================

class TestInverseAndDivision:
    """Test multiplicative inverse and division."""

    def test_inverse_product_is_one(self):
        """Test z * z^(-1) = 1."""
        z = (torch.complex(torch.randn(5) + 1, torch.randn(5)),
             torch.complex(torch.randn(5) + 1, torch.randn(5)))

        z_inv = inverse_idempotent(z, zero_divisor_handling='ignore')
        product = multiply_idempotent(z, z_inv)

        assert torch.allclose(product[0], torch.ones_like(product[0]), atol=1e-5)
        assert torch.allclose(product[1], torch.ones_like(product[1]), atol=1e-5)


# =============================================================================
# Conjugate Tests
# =============================================================================

class TestConjugate:
    """Test bicomplex conjugation properties."""

    def test_conjugate_involution(self):
        """Test (z*)* = z."""
        z = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        z_conj = conjugate_idempotent(z)
        z_conj_conj = conjugate_idempotent(z_conj)

        assert torch.allclose(z_conj_conj[0], z[0], atol=1e-6)
        assert torch.allclose(z_conj_conj[1], z[1], atol=1e-6)

    def test_conjugate_swaps_components(self):
        """Test that idempotent conjugate swaps e1 and e2."""
        z = (torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
             torch.complex(torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])))

        z_conj = conjugate_idempotent(z)

        assert torch.allclose(z_conj[0], z[1])
        assert torch.allclose(z_conj[1], z[0])


# =============================================================================
# Modulus Tests
# =============================================================================

class TestModulus:
    """Test modulus computations."""

    def test_modulus_non_negative(self):
        """Test that modulus is always non-negative."""
        for _ in range(10):
            z = (torch.complex(torch.randn(20), torch.randn(20)),
                 torch.complex(torch.randn(20), torch.randn(20)))
            mod = modulus(z)
            assert (mod >= 0).all()

    def test_modulus_squared_relation(self):
        """Test |z|^2 equals modulus_squared."""
        z = (torch.complex(torch.randn(10), torch.randn(10)),
             torch.complex(torch.randn(10), torch.randn(10)))

        mod = modulus(z)
        mod_sq = modulus_squared(z)

        assert torch.allclose(mod ** 2, mod_sq, atol=1e-5)

    def test_d_modulus_multiplicative(self):
        """Test |z*w|_D = |z|_D * |w|_D."""
        z = (torch.complex(torch.randn(5) + 0.5, torch.randn(5)),
             torch.complex(torch.randn(5) + 0.5, torch.randn(5)))
        w = (torch.complex(torch.randn(5) + 0.5, torch.randn(5)),
             torch.complex(torch.randn(5) + 0.5, torch.randn(5)))

        zw = multiply_idempotent(z, w)

        lhs = d_modulus(zw, eps=0)
        rhs = d_modulus(z, eps=0) * d_modulus(w, eps=0)

        assert torch.allclose(lhs, rhs, atol=1e-4)


# =============================================================================
# Exponential and Logarithm Tests
# =============================================================================

class TestExpLog:
    """Test exponential and logarithm functions."""

    def test_exp_log_inverse(self):
        """Test exp(log(z)) = z for non-zero-divisors."""
        z = (torch.complex(torch.abs(torch.randn(5)) + 0.5, torch.randn(5)),
             torch.complex(torch.abs(torch.randn(5)) + 0.5, torch.randn(5)))

        log_z = log_idempotent(z, zero_divisor_handling='ignore')
        exp_log_z = exp_idempotent(log_z)

        assert torch.allclose(exp_log_z[0], z[0], atol=1e-4)
        assert torch.allclose(exp_log_z[1], z[1], atol=1e-4)

    def test_exp_additivity(self):
        """Test exp(a + b) = exp(a) * exp(b)."""
        a = (torch.complex(torch.randn(5) * 0.5, torch.randn(5) * 0.5),
             torch.complex(torch.randn(5) * 0.5, torch.randn(5) * 0.5))
        b = (torch.complex(torch.randn(5) * 0.5, torch.randn(5) * 0.5),
             torch.complex(torch.randn(5) * 0.5, torch.randn(5) * 0.5))

        lhs = exp_idempotent(add_idempotent(a, b))
        rhs = multiply_idempotent(exp_idempotent(a), exp_idempotent(b))

        assert torch.allclose(lhs[0], rhs[0], atol=1e-4)
        assert torch.allclose(lhs[1], rhs[1], atol=1e-4)


# =============================================================================
# Power Tests
# =============================================================================

class TestPower:
    """Test power function properties."""

    def test_power_zero(self):
        """Test z^0 = 1."""
        z = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        result = power_idempotent(z, 0)

        assert torch.allclose(result[0], torch.ones_like(result[0]), atol=1e-6)
        assert torch.allclose(result[1], torch.ones_like(result[1]), atol=1e-6)

    def test_power_one(self):
        """Test z^1 = z."""
        z = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        result = power_idempotent(z, 1)

        assert torch.allclose(result[0], z[0], atol=1e-6)
        assert torch.allclose(result[1], z[1], atol=1e-6)

    def test_power_two(self):
        """Test z^2 = z * z."""
        z = (torch.complex(torch.randn(5), torch.randn(5)),
             torch.complex(torch.randn(5), torch.randn(5)))

        power_result = power_idempotent(z, 2)
        mult_result = multiply_idempotent(z, z)

        assert torch.allclose(power_result[0], mult_result[0], atol=1e-5)
        assert torch.allclose(power_result[1], mult_result[1], atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
