import numpy as np
import pandas as pd
import plottools
import block_model


def is_undirected(graph):
    return True if np.abs((graph - graph.T)).sum() == 0 else False


def is_square(mat):
    return True if mat.ndim == 2 and mat.shape[0] == mat.shape[1] else False


def odesolver(graph, inits, *, steps=1000, final=100, a, b):
    """
    Solves the aggregation equations with a prescribed interaction function and network structure.

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

    In Progress
    -----------
    * TODO: Allow an adaptive time step with a tolerance mechanism for convergence.
    * TODO: Generalize code to extend to n>=2 dimensions.
    """
    inits = inits.reshape(-1)
    if not is_undirected(graph):
        raise NotImplementedError('Graph is directed')
    if not is_square(graph):
        raise IndexError('Adjacency matrix is not square')
    if graph.shape[0] != inits.shape[0]:
        raise IndexError('Dim mismatch: initial conditions')

    n = graph.shape[0]
    X0 = np.array([inits]).astype('float')  # to be safe, casting error below if this were 'int'
    X = X0.copy()
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
        dX_int = (1/n)*F(np.abs(MX1-MX2))*((MX1-MX2)/np.abs(MX1-MX2))*graph.adj  # not great, this throws error now
        for idx in range(n):
            dX_int[idx, idx] = 0
        dX = np.sum(dX_int, axis=0)
        X0 += h*dX
        X = np.append(X, X0, axis=0)

    return plottools.SolutionWrapper(graph, inits, steps, final, a, b, X)


if __name__ == '__main__':
    C = [250, 250]
    prob_array = np.array([[0.8, 0.05], [0.05, 0.8]])
    grp = block_model.SBMGraph(C, prob_array)

    com1 = -10 + np.random.rand(250)/10
    com2 = 10 + np.random.rand(250)/10
    init = np.append(com1, com2)
    # init = np.arange(500)/100
    sol = odesolver(grp, init, a=0.5, b=0)
