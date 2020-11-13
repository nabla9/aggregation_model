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
    """Reads a grouped config file and returns a dictionary with each key=value line corresponding to a (key,value) item.

    :param str name: Path to config file. A file name alone, by default, opens in current directory.
    :param str group: A specific group to read in the config file.
    :return dict: A dictionary containing each key=value line as items.

    Examples:
        Suppose my_config.cfg contains the following group:
            [group1]
            name = 'Adam'
            hobby = 'Skiing'
            age = 30

        Then, read_config('my_config.cfg','group1') will return the dictionary {'name':'Adam','hobby':'Skiing','age':30}.

    Notes:
        This currently only works with string keys.
    """
    cfgdict = {}
    with open(name, 'r') as cfg:
        try:
            for line in cfg:
                if ('[%s]' % group) in line:
                    raise SubstringFound()
        except SubstringFound:
            for line in cfg:
                if line == '\n':
                    break
                arg = line.split('=')
                arg = [thing.strip() for thing in arg]
                # Try to read value as int/float if possible, otherwise accept as string
                if arg[1].isdigit():
                    val = int(arg[1])
                else:
                    try:
                        val = float(arg[1])
                    except ValueError:
                        val = arg[1]
                cfgdict[arg[0]] = val
    return cfgdict


class SQLConnector:
    """Wraps a SQL connection.

    This class offers functionality useful for recording and querying simulation data.
    """
    def __init__(self, *, cfgfile='agg.conf'):
        cfg = read_config(cfgfile, 'SQL')
        self._connection = mysql.connect(**cfg)
        self._cursor = self.cnx.cursor()
        self._cfgfile = cfgfile  # Store cfgfile path in case of reconnect

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):  # TODO: handle exceptions here
        self.close()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)

    def __getattr__(self, item):
        return getattr(self.cnx, item)

    def __str__(self):
        return '%s object, connected: %s' % (self.__class__, self.is_connected())

    def _flushresults(self):
        try:
            self.cursor.fetchall()
        except mysql.errors.InterfaceError:
            pass

    def close(self):
        self.cursor.close()
        self.cnx.close()

    @property
    def cnx(self):
        return self._get_connection()

    def _get_connection(self):  # TODO: add "is connected" logic here
        return self._connection

    @property
    def cursor(self):
        return self._get_cursor()

    def _get_cursor(self):
        return self._cursor

    def reconnect(self):
        self.__init__(cfgfile=self._cfgfile)

    def record_data(self, params, data, sim_id=None, run_id=None):
        """Records simulation data (generated from odetools.odesolver) and inputs.

        :param params: A set of input parameters (including graph).
        :param data: Simulation state data (outputs).
        :param int sim_id: A specified sim_id to insert records into correct place in SQL table.
        :param int run_id: A specified run_id.

        Notes:
            Database must first be initialized via the code in init.sql before this method will function properly.
        """
        self._flushresults()
        graph = params['graph']
        if sim_id is None:
            self.cursor.execute('SELECT MAX(sim_id) FROM simulations')
            (x,) = self.cursor.fetchall()[0]
            sim_id = x + 1 if x else 1

        # Insert record into simulations table
        self.cursor.execute("SELECT * FROM simulations WHERE sim_id = %s", (sim_id,))
        self.cursor.fetchall()
        if self.cursor.rowcount == 0:
            n_nodes = graph.shape[0]
            n_comms = len(graph.comms)
            ker_a = params['a']
            ker_b = params['b']
            query = ("INSERT INTO simulations (sim_id,n_nodes,n_comms,ker_a,ker_b) "
                     "VALUES (%s, %s, %s, %s, %s)")
            self.cursor.execute(query, (sim_id, n_nodes, n_comms, ker_a, ker_b))

        # Insert record into runs table if necessary
        if run_id is None:
            query = ("INSERT INTO runs (sim_id,p_inner,p_outer,done) "
                     "VALUES (%s,-1,-1,CURRENT_TIMESTAMP())")
            self.cursor.execute(query, (sim_id,))
            self.commit()
            self.cursor.execute('SELECT run_id FROM runs WHERE sim_id = %s', (sim_id,))
            (run_id,) = self.cursor.fetchall()[0]

        # Insert record into simcomms table
        self.cursor.execute("SELECT * FROM communities WHERE sim_id = %s" % sim_id)
        self.cursor.fetchall()
        if self.cursor.rowcount == 0:
            # values = ",".join(str((sim_id, comm_id, comm_nodes))
            #                  for comm_id, comm_nodes in enumerate(graph.comms))
            # query = "INSERT INTO communities (sim_id,comm_id,comm_nodes) VALUES %s" % values
            values = [(sim_id, comm_id, comm_nodes) for comm_id, comm_nodes in enumerate(graph.comms)]
            query = "INSERT INTO communities (sim_id,comm_id,comm_nodes) VALUES (%s,%s,%s)"
            self.cursor.executemany(query, values)  # TODO: parametrize above values?
            self.commit()

        # Insert graph into graphs table
        graph_str = repr(graph).replace(".0, ", ",")
        query = "INSERT INTO graphs SELECT sim_id,run_id,p_inner,p_outer,%s FROM runs WHERE run_id = %s"
        self.cursor.execute(query, (graph_str, run_id))

        # Insert data into simdata table
        times = params['times']
        for idx, row in enumerate(data):
            values = [(sim_id, run_id, node, times[idx], state) for node, state in enumerate(row)]
            query = "INSERT INTO statedata VALUES (%s, %s, %s, %s, %s)"
            # values = ",".join(str((sim_id, run_id, node, times[idx], state))
            #                  for node, state in enumerate(row))
            # query = "INSERT INTO statedata VALUES %s" % values
            self.cursor.executemany(query, values)
        self.commit()

    def get_graph(self, sim_id, run_id=1):
        """Recovers the random adjacency matrix used to generate a particular simulation run.

        :param int sim_id: An identifier for the particular simulation.
        :param int run_id: An identifier for the particular run.
        :return SBMGraphFromDB:

        Implementation:
            record_data stores a given adjacency matrix in JSON format; the original graph object can be reconstructed
            from this data. The "simulations" table is queried for the graph, and the "simcomms" table is queried
            for its associated community information.
        """
        class SBMGraphFromDB(SBMGraph):
            def __init__(self, adj_mat, comm):
                self._adj = adj_mat
                self._com = comm

        self._flushresults()
        try:
            # query = 'SELECT graph FROM graphs WHERE sim_id = %s AND run_id = %s' % (sim_id, run_id)
            query = 'SELECT graph FROM graphs WHERE sim_id = %s AND run_id = %s'
            self.cursor.execute(query, (sim_id, run_id))
            fetched_obj = self.cursor.fetchall()
            adj = np.array(json.loads(fetched_obj[0][0]))

            # query = 'SELECT comm_nodes FROM communities WHERE sim_id = %s' % sim_id
            query = 'SELECT comm_nodes FROM communities WHERE sim_id = %s'
            self.cursor.execute(query, (sim_id,))
            fetched_obj = self.cursor.fetchall()
            comms = [num for (num,) in fetched_obj]

            return SBMGraphFromDB(adj, comms)
        except IndexError:
            raise SQLAccessError('No record in table with sim_id = %s' % sim_id)
