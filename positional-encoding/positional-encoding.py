import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]  
    i = np.arange(d_model)[np.newaxis, :] 
    angles = positions / np.power(base, (2 * (i // 2)) / d_model)
    PE[:, 0::2] = np.sin(angles[:, 0::2])   
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    return PE
    pass