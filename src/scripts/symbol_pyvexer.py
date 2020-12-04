#!/usr/bin/python3
import sys
import math
import os
import socket, pickle
import glob
import binascii
import pprint
import r2pipe
import progressbar
from threading import Thread, Lock
import pyvex
import archinfo
import json

from multiprocessing.pool import ThreadPool

from classes.binary import Binary
from classes.database import Database
from classes.symbol import Symbol

pbar_config = [ '[ ', progressbar.Counter(), '] ', ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
]

MAX_PROCESSES = 32 

def symbol_pyvexer_thread(symbs, collection_name, pbar):
    db_ref = Database()
    db = db_ref.client

    for symb in symbs:
        s = Symbol.fromJSON(symb)
        s.gen_hashes()
        s.save_to_db(db, collection_name)
        global processed
        processed += 1
        pbar.value  = processed
        pbar.update()


def symbol_fill_bytes(symb):
    pipe = r2pipe.open(symb.path, ['-2', '-n', '-z'])
    #The best method so far is to use r2. pyelftools is not up to the job
    #symb_bytes = json.loads( pipe.cmd("px %d @ %d" %(symbol.size, symb.vaddr)) )
    symb_hex_dump = pipe.cmd("px {}@{}".format(str(symb.size), str(symb.vaddr)))

    symb_hex = ""
    for line in symb_hex_dump.split('\n')[1:]:
        #print(line)
        lf = line.split(' ')
        symb_bytes = lf[1:10]
        symb_hex += "".join(symb_bytes)

    assert( len(symb_hex) == symb.size * 2)
    symb.bytes = binascii.unhexlify(symb_hex)
    assert( type(symb.bytes) == type(b''))
    #print("Requesting vex info!")
    pipe.quit()
    symb.gen_vex_features()

def symbol_vexer_threaded():
    #get all symbols
    db_ref = Database()
    db = db_ref.client

    collection_name = "symbols"
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]

    symbols = db[collection_name]
    sfc = 0
    sfc_thresh = MAX_PROCESSES

    while True: 
        #symb = Symbol.fromJSON( symbols.find_one({ 'vex' : None }) )
        #symb = Symbol.fromJSON( symbols.find_one({ 'vex' : None , 'size' : {'$lt' : 5000 } }) )
        res =  symbols.find_one({ 'vex' : {} , 'size' : {'$lt' : 5000 } })
        if not res:
            break
        symb = Symbol.fromJSON(res)

        #print("Requesting vex info!")
        symbol_fill_bytes(symb)
        #print("Got vex info!")
        symb.save_to_db(db, collection_name)

        sys.stdout.write('.')
        sfc += 1
        if sfc > sfc_thresh:
            sfc = 0
            sys.stdout.flush()

if __name__ == "__main__":

    pool = ThreadPool(processes=MAX_PROCESSES)

    pool_res = []
    for i in range(MAX_PROCESSES):
        async_res = pool.apply_async(symbol_vexer_threaded, ())
        pool_res.append(async_res)

    for res in pool_res:
        res.get()
