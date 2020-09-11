from numpy.random import choice
from numpy import array, cumsum, zeros, triu


def create_block_model(comm_list, prob_array, *, symmetric=True):
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

    adj = triu(adj, 1)
    adj += adj.transpose()
    return adj
