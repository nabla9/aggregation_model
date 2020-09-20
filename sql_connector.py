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
