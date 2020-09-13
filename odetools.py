import numpy as np
import pandas as pd


def is_undirected(graph):
    return True if np.abs((graph - graph.T)).sum() == 0 else False


def is_square(mat):
    return True if mat.ndim == 2 and mat.shape[0] == mat.shape[1] else False


def odesolver(graph, inits, *, steps=1000, final=100, a, b):
    if not is_undirected(graph):
        raise NotImplementedError('Graph is directed')
    if not is_square(graph):
        raise IndexError('Adjacency matrix is not square')

    n = graph.shape[0]
    X0 = np.array([inits]).astype('float')  # TODO: check this
    X = X0.copy()
    # df = pd.DataFrame([inits], index=['t=0'])  # TODO: use ndarrays here instead, DataFrame to pass data after
    times = np.linspace(0, final, steps)
    h = times[1] - times[0]

    def make_kernel(p1, p2):
        def kernel(r):
            return np.minimum(p1*r+p2, 1-r)
        return kernel
    F = make_kernel(a, b)

    def compute_direction(x1, x2):  # TODO: generalize to n-d
        return 1 if x1 >= x2 else -1
    compute_direction = np.frompyfunc(compute_direction, 2, 1)

    for _ in times:
        (MX1, MX2) = np.meshgrid(X0, X0)
        dX = np.sum((1/n)*F(np.abs(MX1-MX2))*compute_direction(MX1, MX2).astype('float'), axis=0)  # TODO: fix
        X0 += h*dX
        X = np.append(X, X0, axis=0)

    return X


if __name__ == '__main__':
    graph = np.ones([10, 10])
    inits = np.arange(10)
    X = odesolver(graph, inits, a=0.5, b=0)
