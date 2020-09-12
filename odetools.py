from numpy import ndarray, abs, linspace
import pandas as pd


def is_undirected(graph):
    return True if abs((graph - graph.T)).sum() == 0 else False


def is_square(array):
    return True if array.ndim == 2 and array.shape[0] == array.shape[1] else False


def odesolver(graph, inits, *, steps=1000, final=4000):
    if not is_undirected(graph):
        raise NotImplementedError('Graph is directed')
    if not is_square(graph):
        raise IndexError('Adjacency matrix is not square')

    df = pd.DataFrame([inits], index=['t=0'])  # TODO: use ndarrays here instead, DataFrame to pass data after
    times = linspace(0, final, steps)
    h = times[1] - times[0]

    # TODO: finish this
