import numpy as np
import pandas as pd
import plottools
import block_model


# Sanity checks for graph inputs
def is_undirected(graph):
    return True if np.abs((graph - graph.T)).sum() == 0 else False


def is_square(mat):
    return True if mat.ndim == 2 and mat.shape[0] == mat.shape[1] else False


def generate_inits(graph, *, sep=100, noise='uniform', scale=10):  # do we actually need length here?
    noise_dict = {'uniform': (lambda l: np.random.rand(l)-1/2)}
    n_comms = len(graph.comms)
    n_nodes = np.sum(graph.comms)

    # generate inits spaced 'sep' apart by community, centered about 0, with noise
    inits = (np.array([pos for pos in range(n_comms) for times in range(graph.comms[pos])])*sep).astype('float')
    inits += noise_dict[noise](n_nodes)*scale
    inits -= inits.mean()
    return inits


def odesolver(graph, inits, *, steps=1000, final=1000, a, b, adaptive=True, tol=.001):
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
            return np.minimum(p1 * r + p2, 1 - r)

        return kernel

    F = make_kernel(a, b)

    def compute_direction(x1, x2):  # TODO: generalize to n-d
        return 1 if x1 >= x2 else -1

    compute_direction = np.frompyfunc(compute_direction, 2, 1)

    def take_step(pos, step):
        (MX1, MX2) = np.meshgrid(pos, pos)
        dpos_int = (1/n)*F(np.abs(MX1-MX2))*((MX1-MX2)/np.abs(MX1-MX2))*graph.adj  # not great, this throws error now
        for idx in range(n):
            dpos_int[idx, idx] = 0
        dpos = np.sum(dpos_int, axis=0)
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
            conv_norm = np.max(np.abs(tau))
            if conv_norm < tol:
                X0 = X1_half + tau
                t = times[-1]+h
                times.append(t)
                X = np.append(X, X0, axis=0)
            h *= 0.9 * np.sqrt(tol / conv_norm)

    return plottools.SolutionWrapper(graph, inits, steps, final, a, b, X)  # TODO: this should return times, not steps


if __name__ == '__main__':
    C = [500, 500]
    prob_array = np.array([[0.75, 0.15], [0.15, 0.75]])
    grp = block_model.SBMGraph(C, prob_array)

    init = generate_inits(grp)
    sol = odesolver(grp, init, a=0.5, b=0)
