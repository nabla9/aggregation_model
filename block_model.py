from numpy.random import choice
from numpy import array, cumsum, zeros, triu


def create_block_model(comm_list, prob_array, *, symmetric=True):
    prob_array = [list(row) for row in prob_array]
    comm_idx = cumsum(comm_list)
    n_nodes = comm_idx[-1]
    adj_matr = zeros([n_nodes, n_nodes])

    last_row = 0
    last_col = 0
    for row_idx in comm_idx:
        p_list = prob_array.pop(0)
        for col_idx in comm_idx:
            p_conn = p_list.pop(0)
            if col_idx < row_idx:
                last_col = col_idx
                continue
            adj_matr[last_row:row_idx, last_col:col_idx] = choice([0, 1], [row_idx - last_row, col_idx - last_col], p=[1 - p_conn, p_conn])
            last_col = col_idx
        last_row = row_idx

    adj_matr = triu(adj_matr, 1)
    adj_matr += adj_matr.transpose()
    return adj_matr
