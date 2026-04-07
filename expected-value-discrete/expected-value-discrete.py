import numpy as np

def expected_value_discrete(x, p):
    x=np.array(x,dtype=float)
    p=np.array(p,dtype=float)
    if len(x)!=len(p):
        raise ValueError("x and p must have same length")
    if not np.isclose(np.sum(p),1.0):
        raise ValueError("Probabilities must sum to 1")
    if np.any(p<0):
        raise ValueError("Probabilites must be non-negative")
    return np.sum(x*p)