import numpy as np

def _sin_pe(d_model: int,
            base: float, 
            pos: np.ndarray, 
            i_times_two: int) -> np.ndarray:
    return np.sin(pos / (base ** (i_times_two / d_model)))  # (seq_len,)

def _cos_pe(d_model: int,
            base: float, 
            pos: np.ndarray, 
            i_times_two_plus_one: int) -> np.ndarray:
    i_times_two = i_times_two_plus_one - 1
    return np.cos(pos / (base ** (i_times_two / d_model)))  # (seq_len,)

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here

    if d_model % 2 == 0:
        d_model_even = d_model
    else:
        d_model_even = d_model + 1

    pos = np.arange(seq_len)

    return np.concatenate(
        [
            np.stack([
                _sin_pe(d_model, base, pos, i * 2),
                _cos_pe(d_model, base, pos, i * 2 + 1),
            ], axis=-1)  # (seq_len, 2)
            for i in range(d_model_even // 2)
        ],
        axis=-1
    )[:, :d_model]

    