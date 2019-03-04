import numpy as np
from numba import jit

@jit(nopython=True)
def pearson_correlation(x, y):
    x -= np.mean(x)
    y -= np.mean(y)
    nom = np.sum((x * y))
    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom > 0:
        r = nom / denom
    else:
        r = 0.
    return r


@jit(nopython=True)
def compute_similarities(df, shape):
    distances = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if i == j:
                continue
            if distances[i, j] == 0 and distances[j, i] == 0:
                a = df[:, i]
                b = df[:, j]
                r = pearson_correlation(a, b)

                distances[i, j] = r
                distances[j, i] = r
    return distances