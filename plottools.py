import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SolutionWrapper:
    """
    A wrapper (returned from odetools.odesolver) to store state data from run.

    This stores all state parameters in a single container and provides some useful methods for plotting and analyzing
    results.

    In Progress
    -----------
    * TODO: look into letting odetools pass a dictionary itself instead of tons of pargs.
    * TODO: create method for plotting COM and individual state data over time
    * TODO: store simulated data in a dataframe with labels rather than an ndarray.
    * TODO: write a custom pickler method for saving runs
    """
    def __init__(self, graph_obj, inits, steps, final, a, b, state_data):
        self.inputs = {'graph': graph_obj, 'inits': inits, 'odeparams':
                       {'a': a, 'b': b, 'steps': steps, 'final': final}}
        self.output = state_data

    def compute_center(self):
        comlist = np.cumsum(self.inputs['graph'].comms)

        for idx, num in enumerate(comlist):
            lastnum = comlist[idx-1] if idx >= 1 else 0
            m_i = self.output[:, lastnum:comlist[idx]].mean(axis=1).reshape(-1, 1)

            M = m_i if idx == 0 else np.append(M, m_i, axis=1)

        return M

    def plot_state(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xlabel='time', ylabel='position',
                             title='individual state data')
        comms = self.inputs['graph'].comms

        def plot_comm(axes, k):
            comm_bds = np.cumsum(comms)
            idx_lower = comm_bds[k-1] if k > 0 else 0
            ax.plot(self.output[:, idx_lower:comm_bds[k]])

        for com in range(len(comms)):
            plot_comm(ax, com)

        return fig
