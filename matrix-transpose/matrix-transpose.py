import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    a = np.array(A)
    rows, cols = a.shape
    result = np.zeros((cols, rows), dtype=int)
    for i in range(rows):
        for j in range(cols):
            result[j, i] = a[i, j]
    return result
    pass
