import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if len(seqs) == 0:
        return np.zeros((0,0),dtype=np.int32)
    max_len_from_inp = max(len(seq) for seq in seqs)

    if max_len:
        L = max(max_len_from_inp, max_len)
    else:
        L = max_len_from_inp

    padded = np.array(
        [
            seq + [pad_value] * (L-len(seq))
            for seq in seqs
        ]
    )

    if not max_len:
        return padded
    else:
        return padded[...,:max_len]