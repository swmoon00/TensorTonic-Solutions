import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    # X: (N,d), y: (N)
    N, dim = X.shape
    W = np.zeros(dim)
    b = 0

    step = 0
    while step < steps:
        probs = _sigmoid(X @ W + b)

        loss = 0
        for yi, probi in zip(y, probs):
            loss += yi * np.log(probi) + (1-yi) * np.log(probi)
        loss /= -N

        dl_dw = (probs-y) @ X / N
        dl_db = np.sum(probs-y) / N

        W -= lr * dl_dw
        b -= lr * dl_db
        
        step += 1

    return W, b