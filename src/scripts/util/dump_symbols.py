#!/usr/bin/python3
import sys, json, os

import context
from classes.config import Config
from classes.binary import Binary
from classes.database import Database
from classes.symbol import Symbol

cfg = Config()


if __name__ == '__main__':
    db = Database()

    proj = {
        'name' : 1,
        'path' : 1,
        'bin_name': 1,
        'vaddr': 1,
        'compiler' : 1,
        'optimisation' : 1,
        'size' : 1, 'hash': 1, 'opcode_hash': 1, 
        'vex' : 1,
        'callers': 1,
        'callees': 1,
    }

    query = Database.gen_query({}, projection=proj)
    symbols = db.get_symbols('symbols_stripped', query) 

    fname = cfg.desyl + "/res/" + "symbols_stripped.json"
    print("[+] Saving to {}".format(fname))
    with open(fname, 'w') as f:
        for symb in symbols:
            f.write( json.dumps( symb.to_dict() ) + "\n"  )



