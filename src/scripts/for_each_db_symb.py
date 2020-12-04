#!/usr/bin/python3
import sys
import math
import os

from multiprocessing.pool import ThreadPool
import functools
import itertools

sys.path.insert(0, "../classes/")
from binary import Binary
from database import Database
from symbol import Symbol

def symbol_hasher_thread(binary, col=''):
    assert(isinstance(binary, str))
    db_ref = Database()
    db = db_ref.client

    symbols = db["symbols"] if COL == "" else db["symbols_" + COL]
    res = symbols.find({"path": binary})

    for s in res:
        #s = Symbol.fromDatabase(db, symb['_id'])
        symb = Symbol.fromDict( s )
        symb._gen_raw_hash()
        symb.save_to_db(db, col)

if __name__ == "__main__":

    COL = ""
    #get all symbols
    db_ref = Database()
    db = db_ref.client

    symbols = db["symbols"] if COL == "" else db["symbols_" + COL]
    res = symbols.distinct("path")

    paths = []
    for p in res:
        paths.append( p )
    print("[+] Binaries in DB: " + str(len(paths)))
    print("[+] Starting symbol processing!")


    pool = ThreadPool(processes=32)
    pool.map( symbol_hasher_thread, paths )
    #(_infer_symbol, zip(repeat(known_symbols), unknown[i*32:(i+1)*32] ))
