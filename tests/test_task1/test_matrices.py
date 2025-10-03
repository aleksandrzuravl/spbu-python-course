"""
Tests for matrices.py module
"""

import pytest
from matrices import matrix_add, matrix_multiply, matrix_transpose, _validate_matrix


class TestMatrices:
    """Tests for matrix operations"""
    
    def test_matrix_add_basic(self):
        """Test matrix addition"""
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        expected = [[6.0, 8.0], [10.0, 12.0]]
        assert matrix_add(A, B) == expected
    
    def test_matrix_add_different_sizes(self):
        """Test error when adding matrices of different sizes"""
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with pytest.raises(ValueError):
            matrix_add(A, B)
    
    def test_matrix_multiply_basic(self):
        """Test matrix multiplication"""
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[2.0, 0.0], [1.0, 2.0]]
        expected = [[4.0, 4.0], [10.0, 8.0]]
        assert matrix_multiply(A, B) == expected
    
    def test_matrix_multiply_incompatible(self):
        """Test error when multiplying incompatible matrices"""
        A = [[1.0, 2.0, 3.0]]  # 1x3
        B = [[1.0], [2.0]]     # 2x1
        with pytest.raises(ValueError):
            matrix_multiply(A, B)
    
    def test_matrix_transpose_square(self):
        """Test transpose of square matrix"""
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        expected = [[1.0, 3.0], [2.0, 4.0]]
        assert matrix_transpose(matrix) == expected
    
    def test_matrix_transpose_rectangular(self):
        """Test transpose of rectangular matrix"""
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        assert matrix_transpose(matrix) == expected
