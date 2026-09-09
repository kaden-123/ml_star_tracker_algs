import torch
import numpy as np
import pandas as pd

#bi array of camera star vectors
#ri array of reference celestial star vectors
#assumes they're all normalized
def davenportq(bi, ri, weights=1.0):
    a = weights
    B = a * np.matmul(bi, ri.T)
    S = B + B.T
    z = np.array([B[1][2] - B[2][1], 
                  B[2][0] - B[0][2], 
                  B[0][1] - B[1][0]])
    
    weights = np.broadcast_to(np.asarray(weights, dtype=float), (len(bi),))

    B = (weights[:, None] * bi).T @ ri
    sigma = np.trace(B)
    S = B + B.T
    z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0],
    ])

    K = np.block([
        [np.array([[sigma]]), z[None, :]],
        [z[:, None], S - sigma * np.eye(3)],
    ])

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    q = eigenvectors[:, np.argmax(eigenvalues)]

    # q and -q describe the same rotation; this makes output deterministic.
    if q[0] < 0:
        q = -q

    return q / np.linalg.norm(q)