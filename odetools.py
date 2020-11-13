import numpy as np
import plottools
import block_model
import sql_connector as sqlc


# Sanity checks for graph inputs
def is_undirected(graph):
    return True if np.abs(graph - graph.T).sum() == 0 else False


def is_square(mat):
    return True if mat.ndim == 2 and mat.shape[0] == mat.shape[1] else False


def generate_inits(graph, *, dims=1, sep=100, noise='uniform', scale=10):
    """Generates initial conditions (with noise) to use in the odesolver.

    :param SBMGraph graph: An SBM graph generated from block_model.py. Community data is used to separate initial
        conditions by community and dimensions of adjacency matrix are used for correct init dims.
    :param int dims: Number of spatial dimensions.
    :param float sep: Separation between each community and nearest other community.
    :param str noise: A given noise profile, selected from a dictionary. Allowable options: 'uniform'.
    :param float scale: Width of noise profile.
    :return ndarray: An array of dimensions (1,n_nodes,dims).
    """
    noise_dict = {'uniform': (lambda l: np.random.rand(*l)-1/2)}
    n_comms = len(graph.comms)
    n_nodes = np.sum(graph.comms)
    inits = np.zeros([1, n_nodes, dims])

    # generate inits spaced 'sep' apart by community, centered about 0, with noise
    raw_inits = []
    for x in range(n_comms):
        raw_inits.extend([x] * graph.comms[x])
    inits[0, :, 0] = raw_inits
    inits = inits * sep + scale * noise_dict[noise]([1, n_nodes, dims])
    inits -= inits.mean(axis=1)
    return inits


def flatten_nd(array):
    dims = sorted([num for num in array.shape if num > 1], reverse=True)
    new_array = array.reshape(dims)
    return new_array


def make_kernel(p1, p2):
    """Returns an interaction kernel (also used in odesolver) with the prescribed values for each parameter.

    Notes:
        The input to the kernel function is always a pairwise distance (scalar) quantity. This is readily vectorized in code
        of odesolver/take_step.
    """
    def kernel(r):
        return np.minimum(p1 * r + p2, 1 - r)
    return kernel


def odesolver(graph, inits, *, steps=1000, final=1000, a, b, adaptive=True, tol=.01):
    """Solves the aggregation equations with a prescribed interaction function and network structure.

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

    Implementation:
        Equations are evolved via forward Euler with a fixed time step. Kernel function as defined in von Brecht et al.
        (2013).

    In Progress:
        * TODO: restructure function (sub-f forward_euler, calls take_step...)
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
            dpos = np.sum((1/n_nodes) * F(np.abs(diffs)) * np.sign(diffs) * adj, axis=0)
        else:
            dist = np.sqrt(np.sum(diffs ** 2, axis=2, keepdims=True))
            for idx in range(n_nodes):
                dist[idx, idx] = 1
            dpos = (1/n_nodes) * np.sum(F(dist) * diffs/dist * adj, axis=0)
        return pos + step*dpos

    if adaptive is False:
        for i in times:
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


def run_simulation(*, resume=False, n_runs, n_nodes, a, b, inner_probs, outer_probs):
    """Runs simulation on a grid of inner/outer probabilities and record data in SQL table.

    :param resume: A flag indicating whether or not the most recent unfinished record in SQL table should be completed.
    :param n_runs: Number of runs per probability pair.
    :param n_nodes: Number of nodes to generate graph from.
    :param a: kernel parameter a.
    :param b: kernel parameter b.
    :param inner_probs: A sequence of inner probabilities for graph generation.
    :param outer_probs: A sequence of outer probabilities for graph generation.
    """
    # establish plan and link to database
    if not resume:
        p1, p2 = np.meshgrid(inner_probs, outer_probs)
        prob_pairs = list(zip(p1.flatten(), p2.flatten()))
        with sqlc.SQLConnector() as dbconn:
            dbconn.cursor.execute("SELECT MAX(sim_id) FROM simulations")
            (x,) = dbconn.cursor.fetchall()[0]
            sim_id = x+1 if x else 1
            dbconn.cursor.execute("INSERT INTO simulations "
                                  "VALUES (%s,%s,%s,%s,%s,%s,NULL)" % (sim_id, n_runs,
                                                                       n_nodes, 2, a, b))
            dbconn.cursor.execute("INSERT INTO communities "
                                  "VALUES ({0},1,{1}),({0},2,{1})".format(sim_id, n_nodes//2))
            query = ("INSERT INTO runs (sim_id,p_inner,p_outer) "
                     "VALUES ({}, %s, %s)".format(sim_id))
            for run in range(n_runs):
                dbconn.cursor.executemany(query, prob_pairs)
                dbconn.commit()

    # obtain plan from database
    with sqlc.SQLConnector() as dbconn:
        dbconn.cursor.execute("SELECT MAX(sim_id) FROM runs WHERE done IS NULL")
        (sim_id,) = dbconn.cursor.fetchall()[0]
        dbconn.cursor.execute("SELECT run_id,p_inner,p_outer FROM runs "
                              "WHERE sim_id = %s AND done IS NULL", (sim_id,))
        results = dbconn.cursor.fetchall()
    run_params = [(int(run), float(p_in), float(p_out)) for run, p_in, p_out in results]
    # run plan
    comms = [n_nodes//2, n_nodes - n_nodes//2]
    for run_id, p_in, p_out in run_params:
        graph = block_model.SBMGraph(comms, np.array([[p_in, p_out], [p_out, p_in]]))
        inits = generate_inits(graph)
        solution = odesolver(graph, inits, a=a, b=b)
        with sqlc.SQLConnector() as dbconn:
            dbconn.record_data(solution.inputs, solution.output[:, :, 0], sim_id, run_id)
            dbconn.cursor.execute("UPDATE runs SET done = CURRENT_TIMESTAMP() "
                                  "WHERE run_id = %s", (run_id,))
            dbconn.commit()


if __name__ == '__main__':
    C = [100, 100]
    prob_array = np.array([[1, 1], [1, 1]])
    grp = block_model.SBMGraph(C, prob_array)

    init = generate_inits(grp, dims=1)
    sol = odesolver(grp, init, final=4000, a=0.5, b=0)
