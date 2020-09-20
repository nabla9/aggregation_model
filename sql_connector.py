import mysql.connector as mysql


class SubstringFound(Exception):
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

    def close(self):
        self._cursor.close()
        self._connect.close()

    def fetchall(self):
        self._cursor.fetchall()

    def record_data(self, data):
        try:
            self.fetchall()
        except mysql.errors.InterfaceError:
            pass
        self._cursor.execute('SELECT MAX(sim_id) FROM simulations')
        (x,) = self.fetchall()[0]
        sim_id = x if x else 1  # TODO: change back to x+1 here later

        # TODO: temporary, change this later
        sim_id = [sim_id]*1000
        times = np.arange(1001)
        nodes = np.arange(1,1001)
        for idx, row in enumerate(data):
            times_ar = [times[idx]]*1000
            data = zip(sim_id, nodes, times_ar, row)
            data_str = [str(tup) for tup in data]
            commit_state = 'INSERT INTO simdata VALUES ' + ', '.join(data_str)
            self._cursor.execute(commit_state)
            self._connect.commit()

