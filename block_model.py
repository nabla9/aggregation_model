from numpy.random import choice
from numpy import ndarray, cumsum, zeros, triu


def create_block_model(comm_list, prob_array):
    """
    A simple function to create a stochastic block model adjacency matrix.

    :param list comm_list: A 1D list of individuals assigned to respective communities.
    :param ndarray prob_array: A 2D array of interconnection probabilities. P[i,j] is the probability
                              of connection between individuals from comm i,j.
    :return ndarray adj: A symmetric adjacency matrix of 0's and 1's corresponding to edges between individuals.

    Implementation
    --------------
    The top half of the undirected adjacency matrix is sampled randomly. The lower half is populated by adding the
    transpose. Diagonal entries (self-edges) are set to 0 by convention.

    Notes
    --------------
    The graph generated here naturally "hard codes" community structure -- groups where the density of edges between
    members can differ significantly from the density of links with non-members.
    """
    comm_idx = cumsum(comm_list)
    n_nodes = comm_idx[-1]
    adj = zeros([n_nodes, n_nodes])

    for r_idx, r in enumerate(comm_idx):
        for c_idx, c in enumerate(comm_idx):
            p_conn = prob_array[r_idx][c_idx]
            if c < r:
                continue
            else:
                rl = comm_idx[r_idx-1] if r_idx != 0 else 0
                cl = comm_idx[c_idx-1] if c_idx != 0 else 0
                adj[rl:r, cl:c] = choice([0, 1], [r - rl, c - cl], p=[1 - p_conn, p_conn])

    # Zero sampled entries below/including the main diagonal and add transpose to fill in the rest
    adj = triu(adj, 1)
    adj += adj.transpose()
    return adj
