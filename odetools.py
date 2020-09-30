import numpy as np
import pandas as pd
import plottools
import block_model


# Sanity checks for graph inputs
def is_undirected(graph):
    return True if np.abs((graph - graph.T)).sum() == 0 else False


def is_square(mat):
    return True if mat.ndim == 2 and mat.shape[0] == mat.shape[1] else False


def generate_inits(graph, *, dims=1, sep=100, noise='uniform', scale=10):
    noise_dict = {'uniform': (lambda l: np.random.rand(*l)-1/2)}
    n_comms = len(graph.comms)
    n_nodes = np.sum(graph.comms)
    inits = np.zeros([1, n_nodes, dims])

    # generate inits spaced 'sep' apart by community, centered about 0, with noise
    inits[0, :, 0] = (np.array([pos for pos in range(n_comms) for times in range(graph.comms[pos])])*sep).astype('float')
    inits += noise_dict[noise]([1, n_nodes, dims])*scale
    inits -= inits.mean(axis=1)
    return inits


def flatten_nd(array):
    dims = sorted([num for num in array.shape if num > 1], reverse=True)
    new_array = array.reshape(dims)
    return new_array


def make_kernel(p1, p2):
    def kernel(r):
        return np.minimum(p1 * r + p2, 1 - r)
    return kernel


def odesolver(graph, inits, *, steps=1000, final=1000, a, b, adaptive=True, tol=.01):
    """
    Solves the aggregation equations with a prescribed interaction function and network structure.

    :param tol: Tolerance level for adaptive half-step method.
    :param bool adaptive: Flag specifying whether an adaptive or fixed timestep is taken. Ignore 'steps' and 'final'
                          if true and use 'tol'.
    :param SBMGraph graph: A stochastic block object (in block_model.py)
    :param list inits: Initial conditions. Must match with the row/col dim of GRAPH.
    :param int steps: Number of fixed-width time steps taken.
    :param int final: System of ODEs evolved to this point.
    :param float a: One parameter in kernel defined below.
    :param float b: as above.
    :return SolutionWrapper: An object (defined in plottools.py) that wraps the solution data with input parameters and
                            some useful methods. Refer to plottools.SolutionWrapper for more.

    Implementation
    --------------
    Equations are evolved via forward Euler with a fixed time step. Kernel function as defined in von Brecht et al.
    (2013).
    """
    n = graph.shape[0]
    adj = graph.adj.reshape(*graph.shape, 1)
    if not is_undirected(graph):
        raise NotImplementedError('Graph is directed')
    if not is_square(graph):
        raise IndexError('Adjacency matrix is not square')
    if n != inits.shape[1]:
        raise IndexError('Dim mismatch: initial conditions')

    X0 = np.array(inits).astype('float')  # to be safe, casting error below if this were 'int'
    X = X0.copy()
    times = np.linspace(0, final, steps)
    h = times[1] - times[0]
    F = make_kernel(a, b)

    def take_step(pos, step):
        n_nodes = pos.shape[1]
        n_dims = pos.shape[2]
        MX1 = np.zeros([n_nodes, n_nodes, n_dims])
        MX2 = np.zeros([n_nodes, n_nodes, n_dims])
        for idx in range(n_dims):
            (MX1[:, :, idx], MX2[:, :, idx]) = np.meshgrid(pos[:, :, idx], pos[:, :, idx])
        diffs = MX1-MX2

        # 1-D specific optimization
        if n_dims == 1:
            dpos = np.sum((1/n_nodes)*F(np.abs(diffs))*np.sign(diffs)*adj, axis=0)
        else:
            dist = np.sqrt(np.sum(diffs ** 2, axis=2, keepdims=True))
            for idx in range(n_nodes):
                dist[idx, idx] = 1
            dpos = (1/n_nodes)*np.sum(F(dist)*diffs/dist*adj, axis=0)
        return pos + step * dpos

    if adaptive is False:
        for _ in times:
            X0 = take_step(X0, h)
            X = np.append(X, X0, axis=0)

    else:
        t = 0
        h = 1
        times = [0]
        while t < final:
            X1_full = take_step(X0, h)
            Xh = (X1_full + X0)/2  # this is probably bad, should return X0 + (h/2)*dX instead, ideally
            X1_half = take_step(Xh, h/2)

            tau = X1_half - X1_full
            conv_norm = np.max(np.sqrt(np.sum(tau ** 2, axis=2)))
            if conv_norm < tol:
                X0 = X1_half + tau
                t = times[-1]+h
                times.append(t)
                X = np.append(X, X0, axis=0)
            h *= 0.9 * np.sqrt(tol / conv_norm)

    params_dict = {'graph': graph, 'inits': inits, 'times': times, 'a': a, 'b': b}
    return plottools.SolutionWrapper(params_dict, X)


if __name__ == '__main__':
    C = [100, 100]
    prob_array = np.array([[1, 1], [1, 1]])
    grp = block_model.SBMGraph(C, prob_array)

    init = generate_inits(grp, dims=1)
    sol = odesolver(grp, init, final=4000, a=0.5, b=0)
