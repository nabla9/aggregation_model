import mysql.connector as mysql
import numpy as np
import json
import traceback


class SubstringFound(Exception):
    pass


class DatabaseError(Exception):
    pass


class SQLAccessError(DatabaseError):
    pass


def read_config(name, group):
    cfgdict = {}
    with open(name, 'r') as cfg:
        try:
            for line in cfg:
                if ('[' + group + ']') in line:
                    raise SubstringFound()
        except SubstringFound:
            for line in cfg:
                if line == '\n':
                    break
                arg = line.split('=')
                arg = [thing.strip() for thing in arg]
                cfgdict[arg[0]] = arg[1] if not arg[1].isdigit() else int(arg[1])  # TODO: make this work with decimals
    return cfgdict


class SQLConnector:
    def __init__(self, *, cfgfile='agg.conf'):
        cfg = read_config(cfgfile, 'SQL')
        self._connect = mysql.connect(**cfg)
        self._cursor = self._connect.cursor()
        self._cfgfile = cfgfile  # Store cfgfile path in case of reconnect

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):  # TODO: handle exceptions here
        self.close()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)

    def __str__(self):
        return '%s object, connected: %s' % (self.__class__, self._connect.is_connected())

    def _flushresults(self):
        try:
            self.fetchall()
        except mysql.errors.InterfaceError:
            pass

    def close(self):
        self._cursor.close()
        self._connect.close()

    def reconnect(self):
        self.__init__(cfgfile=self._cfgfile)

    def fetchall(self):
        return self._cursor.fetchall()

    def record_data(self, params, data):
        self._flushresults()
        self._cursor.execute('SELECT MAX(sim_id) FROM simulations')
        (x,) = self.fetchall()[0]

        # Insert record into simulations table
        sim_id = x + 1 if x else 1
        graph = params['graph']
        graph_str = repr(graph).replace(".0, ", ",")
        n_nodes = graph.shape[0]
        n_comms = len(graph.comms)
        ker_a = params['a']
        ker_b = params['b']
        query = ("INSERT INTO simulations (sim_id,graph,n_nodes,n_comms,ker_a,ker_b) "
                 "VALUES (%s, '%s', %s, %s, %s, %s)" % (sim_id, graph_str, n_nodes, n_comms, ker_a, ker_b))
        self._cursor.execute(query)

        # Insert record into simcomms table
        query = ("INSERT INTO simcomms (sim_id,comm_id,comm_nodes) VALUES "
                 + ','.join(str((sim_id, comm_id, comm_nodes)) for comm_id, comm_nodes in enumerate(graph.comms)))
        self._cursor.execute(query)
        self._connect.commit()

        # Insert data into simdata table
        times = params['times']
        for idx, row in enumerate(data):
            query = ("INSERT INTO simdata VALUES "
                     + ','.join(str((sim_id, node, times[idx], state)) for node, state in enumerate(row)))
            self._cursor.execute(query)
        self._connect.commit()

    def get_graph(self, sim_id):
        self._flushresults()

        try:
            query = 'SELECT graph FROM simulations WHERE sim_id = %s' % sim_id
            self._cursor.execute(query)
            fetched_obj = self._cursor.fetchall()
            return np.array(json.loads(fetched_obj[0][0]))
        except IndexError:
            raise SQLAccessError('No record in table with sim_id = %s' % sim_id)
