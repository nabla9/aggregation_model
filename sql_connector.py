import mysql.connector as mysql
import numpy as np
import json
import traceback
from block_model import SBMGraph


class SubstringFound(Exception):
    pass


class DatabaseError(Exception):
    pass


class SQLAccessError(DatabaseError):
    pass


def read_config(name, group):
    """
    Reads a grouped config file and returns a dictionary with each key=value line corresponding to a (key,value) item.

    :param str name: Path to config file. A file name alone, by default, opens in current directory.
    :param str group: A specific group to read in the config file.
    :return dict: A dictionary containing each key=value line as items.

    Examples
    --------
    Suppose my_config.cfg contains the following group:
        [group1]
        name = 'Adam'
        hobby = 'Skiing'
        age = 30

    Then, read_config('my_config.cfg','group1') will return the dictionary {'name':'Adam','hobby':'Skiing','age':30}.

    Notes
    -----
    This currently only works with string keys and values that are strings or integers.
    """
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
    """
    A SQL connection wrapper. Offers functionality useful for recording and querying simulation data.
    """
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

    def commit(self):
        self._connect.commit()

    def execute(self, query):
        self._cursor.execute(query)

    def reconnect(self):
        self.__init__(cfgfile=self._cfgfile)

    def fetchall(self):
        return self._cursor.fetchall()

    def record_data(self, params, data, sim_id=None, run_id=None):
        """
        A method to record simulation data (generated from odetools.odesolver) and inputs.

        :param params: A set of input parameters (including graph).
        :param data: Simulation state data (outputs).
        :param int sim_id: A specified sim_id to insert records into correct place in SQL table.
        :param int run_id: A specified run_id.

        Notes
        -----
        Database must first be initialized via the code in init.sql before this method will function properly.
        """
        self._flushresults()
        if sim_id is None:
            self._cursor.execute('SELECT MAX(sim_id) FROM simulations')
            (x,) = self.fetchall()[0]
            sim_id = x + 1 if x else 1

        # Insert record into simulations table
        self._cursor.execute("SELECT * FROM simulations WHERE sim_id = %s" % sim_id)
        self.fetchall()
        if self._cursor.rowcount == 0:
            graph = params['graph']
            n_nodes = graph.shape[0]
            n_comms = len(graph.comms)
            ker_a = params['a']
            ker_b = params['b']
            query = ("INSERT INTO simulations (sim_id,n_nodes,n_comms,ker_a,ker_b) "
                     "VALUES (%s, %s, %s, %s, %s)" % (sim_id, n_nodes, n_comms, ker_a, ker_b))
            self._cursor.execute(query)

        # Insert record into runs table if necessary
        if run_id is None:
            query = "INSERT INTO runs (sim_id,p_inner,p_outer,done) VALUES (%s,-1,-1,CURRENT_TIMESTAMP())"
            self._cursor.execute(query, (sim_id,))
            self._connect.commit()
            self._cursor.execute('SELECT run_id FROM runs WHERE sim_id = %s', (sim_id,))
            (run_id,) = self.fetchall()[0]

        # Insert record into simcomms table
        self._cursor.execute("SELECT * FROM communities WHERE sim_id = %s" % sim_id)
        self.fetchall()
        if self._cursor.rowcount == 0:
            query = ("INSERT INTO communities (sim_id,comm_id,comm_nodes) VALUES "
                     + ','.join(str((sim_id, comm_id, comm_nodes)) for comm_id, comm_nodes in enumerate(graph.comms)))
            self._cursor.execute(query)
            self._connect.commit()

        # Insert graph into graphs table
        graph = params['graph']
        graph_str = repr(graph).replace(".0, ", ",")
        query = "INSERT INTO graphs SELECT sim_id,run_id,p_inner,p_outer,%s FROM runs WHERE run_id = %s"
        self._cursor.execute(query, (graph_str, run_id))
        self._connect.commit()

        # Insert data into simdata table
        times = params['times']
        for idx, row in enumerate(data):
            query = ("INSERT INTO statedata VALUES "
                     + ','.join(str((sim_id, run_id, node, times[idx], state)) for node, state in enumerate(row)))
            self._cursor.execute(query)
        self._connect.commit()

    def get_graph(self, sim_id, run_id=1):
        """
        A method to recover the random adjacency matrix used to generate a particular simulation run.

        :param int sim_id: An identifier for the particular simulation.
        :param int run_id: An identifier for the particular run.
        :return SBMGraphFromDB:

        Implementation
        --------------
        record_data stores a given adjacency matrix in JSON format; the original graph object can reconstructed from
        this data. The "simulations" table is queried for the graph, and the "simcomms" table is queried for its
        associated community information.
        """
        class SBMGraphFromDB(SBMGraph):
            def __init__(self, adj_mat, comm):
                self._adj = adj_mat
                self._com = comm

        self._flushresults()
        try:
            query = 'SELECT graph FROM graphs WHERE sim_id = %s AND run_id = %s' % (sim_id, run_id)
            self._cursor.execute(query)
            fetched_obj = self._cursor.fetchall()
            adj = np.array(json.loads(fetched_obj[0][0]))

            query = 'SELECT comm_nodes FROM communities WHERE sim_id = %s' % sim_id
            self._cursor.execute(query)
            fetched_obj = self._cursor.fetchall()
            comms = [num for (num,) in fetched_obj]

            return SBMGraphFromDB(adj, comms)
        except IndexError:
            raise SQLAccessError('No record in table with sim_id = %s' % sim_id)
