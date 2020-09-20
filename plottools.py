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
    * TODO: store simulated data in a dataframe with labels rather than an ndarray.
    """
    def __init__(self, params_dict, state_data):
        self.inputs = params_dict
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

        def plot_comm(axes, k, col):
            """
            A plotting method for state data **by community**.

            This method separates and color codes individuals by community and provides appropriate legend labels.

            :param AxesSubplot axes: a figure axes handle to plot into.
            :param int k: a community index.
            :param Union[str, list] col: a color, can either be a name or an RGBA list.

            In Progress
            -----------
            * TODO: create an option to plot community centers of mass over time.
            """
            comm_bds = np.cumsum(comms)
            idx_lower = comm_bds[k-1] if k > 0 else 0
            axes.plot(self.output[:, idx_lower:comm_bds[k]], color=col)

            lines = axes.get_lines()
            a_line = lines[-1]
            a_line.set_label('community %s' % k)  # Set only one label in legend per community

        # Color code communities, pick colors far apart on cmap
        cmap = plt.get_cmap('jet')
        col_list = np.linspace(0, 256, len(comms))
        # Compute COMs and plot with state data
        center_array = self.compute_center()
        for com in range(len(comms)):
            plot_comm(ax, com, cmap(col_list[com]))
            ax.plot(center_array[:, com], '--k')
        ax.get_lines()[-1].set_label('COM')

        fig.legend()
        return fig
