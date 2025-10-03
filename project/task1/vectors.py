"""
Module for vector operations.
Implements basic vector operations: dot product, length, angle between vectors.
"""

import math
from typing import List


def dot_product(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculates the dot product of two vectors.
    
    Args:
        vector1: first vector (list of numbers)
        vector2: second vector (list of numbers)
    
    Returns:
        Number - result of the dot product
        
    Raises:
        ValueError: if vectors have different lengths
        TypeError: if inputs are not lists of numbers
    """
    _validate_vector(vector1)
    _validate_vector(vector2)
    
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")
    
    result = 0
    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]
    
    return result


def vector_length(vector: List[float]) -> float:
    """
    Calculates the length (magnitude) of a vector.
    
    Args:
        vector: input vector (list of numbers)
    
    Returns:
        Number - length of the vector
    """
    _validate_vector(vector)
    
    sum_of_squares = 0
    for element in vector:
        sum_of_squares += element ** 2
    
    return math.sqrt(sum_of_squares)


def angle_between(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculates the angle between two vectors in radians.
    
    Args:
        vector1: first vector
        vector2: second vector
    
    Returns:
        Number - angle in radians
        
    Raises:
        ValueError: if one of the vectors is zero-length
    """
    
    length1 = vector_length(vector1)
    length2 = vector_length(vector2)
    
    if length1 == 0 or length2 == 0:
        raise ValueError("Vectors must not be zero-length")
    
    dot_prod = dot_product(vector1, vector2)
    cosine = dot_prod / (length1 * length2)
    
    # Ensure cosine is within valid range [-1, 1] to avoid floating point errors
    cosine = max(-1, min(1, cosine))
    
    return math.acos(cosine)



