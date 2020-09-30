import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sql_connector as sqlc
import odetools


class SolutionWrapper:
    """
    A wrapper (returned from odetools.odesolver) to store state data from run.

    This stores all state parameters in a single container and provides some useful methods for plotting and analyzing
    results.

    In Progress
    -----------
    * TODO: store simulated data in a dataframe with labels rather than an ndarray.
    """
    def __init__(self, params_dict, state_data):
        self.inputs = params_dict
        self.output = state_data
        self._ndims = state_data.shape[2]

    def compute_center(self):
        comm_endidx = np.cumsum(self.inputs['graph'].comms)
        for idx, num in enumerate(comm_endidx):
            prev = comm_endidx[idx-1] if idx >= 1 else 0
            m_i = self.output[:, prev:comm_endidx[idx], :].mean(axis=1, keepdims=True)
            M = m_i if idx == 0 else np.append(M, m_i, axis=1)
        return M

    def plot_state(self, time=None):
        """
        Plots state data (output) of simulation runs in 1D or 2D. Either calls _plot_state_1D or _plot_state_2D. These
        sub-functions should not be called directly.
        :param time: Plots a particular time slice for a 2D simulation run. Required in 2D, does nothing in 1D. 
        :return fig: A handle to the current figure window.
        """
        if self._ndims == 1:
            return self._plot_state_1D()
        elif self._ndims == 2:
            return self._plot_state_2D(time)
        else:
            raise NotImplementedError('number of dimensions > 2')

    def _plot_state_1D(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xlabel='time', ylabel='position', title='individual state data')
        comms = self.inputs['graph'].comms

        def plot_comm_1D(axes, k, col):
            """
            A plotting method for state data **by community**.

            This method separates and color codes individuals by community and provides appropriate legend labels.

            :param AxesSubplot axes: a figure axes handle to plot into.
            :param int k: a community index.
            :param Union[str, list] col: a color, can either be a name or an RGBA list.
            """
            comm_endidx = np.cumsum(comms)
            prev = comm_endidx[k-1] if k > 0 else 0
            axes.plot(self.output[:, prev:comm_endidx[k], 0], color=col)

            lines = axes.get_lines()
            lines[-1].set_label('community %s' % k)  # Set only one label in legend per community

        # Color code communities, pick colors far apart on cmap
        cmap = plt.get_cmap('jet')
        col_list = np.linspace(0, 256, len(comms))
        # Compute COMs and plot with state data
        M = self.compute_center()
        for com in range(len(comms)):
            plot_comm_1D(ax, com, cmap(col_list[com]))
            ax.plot(M[:, com, 0], '--k')
        ax.get_lines()[-1].set_label('COM')

        fig.legend()
        return fig
    
    def _plot_state_2D(self, time):
        if time is None:
            raise ValueError('must pass numeric time value')
        idx = np.argmin(np.abs(np.array(self.inputs['times']) - time))
        time = self.inputs['times'][idx]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, title='individual state data @ time=%s' % time)
        comms = self.inputs['graph'].comms

        def plot_comm_2D(axes, k, idx, col):
            comm_endidx = np.cumsum(comms)
            prev = comm_endidx[k - 1] if k > 0 else 0
            axes.scatter(self.output[idx, prev:comm_endidx[k], 0], self.output[idx, prev:comm_endidx[k], 1],
                         color=col, label='community %s' % k)

        cmap = plt.get_cmap('jet')
        col_list = np.linspace(0, 256, len(comms))
        M = self.compute_center()
        for com in range(len(comms)):
            plot_comm_2D(ax, com, idx, cmap(col_list[com]))
            ax.scatter(M[idx, com, 0], M[idx, com, 1], 50, color=cmap(col_list[com]), marker='x')

        fig.legend()
        return fig

    def record_data(self):
        if self._ndims != 1:
            raise NotImplementedError
        with sqlc.SQLConnector() as dbconn:
            dbconn.record_data(self.inputs, self.output[:, :, 0])
