"""
Module for matrix operations.
Implements basic matrix operations: addition, multiplication, transpose.
"""

from typing import List


def matrix_add(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
    """
    Adds two matrices.
    
    Args:
        matrix1: first matrix (list of lists of numbers)
        matrix2: second matrix (list of lists of numbers)
    
    Returns:
        New matrix - result of addition
        
    Raises:
        ValueError: if matrices have different dimensions
        TypeError: if inputs are not valid matrices
    """
    _validate_matrix(matrix1)
    _validate_matrix(matrix2)
    
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])
    
    if rows1 != rows2 or cols1 != cols2:
        raise ValueError("Matrices must have the same dimensions")
    
    result: List[List[float]] = []
    for i in range(rows1):
        row: List[float] = []
        for j in range(cols1):
            row.append(matrix1[i][j] + matrix2[i][j])
        result.append(row)
    
    return result


def matrix_multiply(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
    """
    Multiplies two matrices.
    
    Args:
        matrix1: first matrix
        matrix2: second matrix
    
    Returns:
        New matrix - result of multiplication
        
    Raises:
        ValueError: if number of columns in first matrix doesn't match 
                   number of rows in second matrix
    """
    _validate_matrix(matrix1)
    _validate_matrix(matrix2)
    
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])
    
    if cols1 != rows2:
        raise ValueError(
            f"Number of columns in first matrix ({cols1}) must equal "
            f"number of rows in second matrix ({rows2})"
        )
    
    result: List[List[float]] = []
    for i in range(rows1):
        row: List[float] = []
        for j in range(cols2):
            element = 0.0
            for k in range(cols1):
                element += matrix1[i][k] * matrix2[k][j]
            row.append(element)
        result.append(row)
    
    return result


def matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    """
    Transposes a matrix (swaps rows and columns).
    
    Args:
        matrix: input matrix
    
    Returns:
        New matrix - transposed version
    """
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    result: List[List[float]] = []
    for j in range(cols):
        new_row: List[float] = []
        for i in range(rows):
            new_row.append(matrix[i][j])
        result.append(new_row)
    
    return result
