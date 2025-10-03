"""
Tests for vectors.py module
"""

import math
import pytest
from vectors import dot_product, vector_length, angle_between, _validate_vector


class TestVectors:
    """Tests for vector operations"""
    
    def test_dot_product_basic(self):
        """Test dot product for basic cases"""
        assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0
        assert dot_product([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert dot_product([-1.0, 2.0], [3.0, -4.0]) == -11.0
    
    def test_dot_product_3d(self):
        """Test dot product for 3D vectors"""
        assert dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
    
    def test_dot_product_different_length(self):
        """Test error when vectors have different lengths"""
        with pytest.raises(ValueError):
            dot_product([1.0, 2.0], [1.0, 2.0, 3.0])
    
    def test_vector_length_basic(self):
        """Test vector length calculation"""
        assert vector_length([3.0, 4.0]) == 5.0
        assert vector_length([0.0, 0.0]) == 0.0
        assert vector_length([1.0, 0.0]) == 1.0
    
    def test_angle_between_orthogonal(self):
        """Test angle between perpendicular vectors"""
        angle = angle_between([1.0, 0.0], [0.0, 1.0])
        assert math.isclose(angle, math.pi / 2, rel_tol=1e-10)
    
    def test_angle_between_parallel(self):
        """Test angle between parallel vectors"""
        angle = angle_between([1.0, 0.0], [2.0, 0.0])
        assert math.isclose(angle, 0.0, rel_tol=1e-10)
    
