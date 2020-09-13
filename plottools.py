import pandas as pd
import numpy as np


class SolutionWrapper:
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
