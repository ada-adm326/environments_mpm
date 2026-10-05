import numpy as np
from scipy.ndimage import gaussian_filter
from sympy import Matrix


__all__ = ['rand_array', 'smooth_image', 'my_mat_solve']


def rand_array(shape):
    return np.random.rand(*shape)


def smooth_image(a, sigma=1):
    return gaussian_filter(a, sigma=sigma)


def my_mat_solve(A, b):
    """Solve the linear system A*x = b symbolically."""
    matrix_a = Matrix(A)
    vector_b = Matrix(b)

    if matrix_a.rows != matrix_a.cols:
        raise ValueError('A must be square.')
    if matrix_a.rows != vector_b.rows:
        raise ValueError('A and b must have compatible dimensions.')

    return matrix_a.LUsolve(vector_b)