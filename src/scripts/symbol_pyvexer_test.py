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

sys.path.insert(0, "../classes/")
from binary import Binary
from database import Database
from symbol import Symbol

pbar_config = [ '[ ', progressbar.Counter(), '] ', ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
]

MAX_PROCESSES = 1

def symbol_pyvexer_thread(symbs, pbar):
    db_ref = Database()
    db = db_ref.client

    for symb in symbs:
        s = Symbol.fromJSON(symb)
        s.gen_hashes()
        s.save_to_db(db)
        global processed
        processed += 1
        pbar.value  = processed
        pbar.update()


def symbol_fill_bytes(symb):
    pipe = r2pipe.open(symb.path, ['-2'])
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
    print(symb_hex)
    sys.exit(-1)
    symb.bytes = binascii.unhexlify(symb_hex)
    assert( type(symb.bytes) == type(b''))
    #print("Requesting vex info!")
    pipe.quit()
    #symb.gen_vex_features()




def symbol_vexer_threaded():
    #get all symbols
    db_ref = Database()
    db = db_ref.client
    symbols = db.symbols

    while True: 
        #symb = Symbol.fromJSON( symbols.find_one({ 'vex' : None }) )
        #symb = Symbol.fromJSON( symbols.find_one({ 'vex' : None , 'size' : {'$lt' : 5000 } }) )
        symb = Symbol.fromJSON( symbols.find_one({ 'vex' : None , 'size' : {'$gt' : 5000 } }) )
        if not symb:
            break

        #print("Requesting vex info!")
        symbol_fill_bytes(symb)
        #print("Got vex info!")
        #symb.save_to_db(db)
        sys.exit(-1)

        sys.stdout.write('.')
        sys.stdout.flush()

if __name__ == "__main__":
        symbol_vexer_threaded()
