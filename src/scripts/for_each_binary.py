#!/usr/bin/python3
import sys
import math
import os
import glob
import progressbar
import pprint
from threading import Thread, Lock

sys.path.insert(0, "../classes/")
from binary import Binary
from database import Database
from symbol import Symbol

NUM_THREADS = 32
processed = 0

pbar_config = [ '[ ', progressbar.Counter(), '] ', ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
]

def symbol_hasher_thread(symbs, pbar):
    db_ref = Database()
    db = db_ref.client

    for symb in symbs:
        s = Symbol.fromDatabase(db, symb['_id'])
        s.gen_hashes()
        s.save_to_db(db)
        global processed
        processed += 1
        pbar.value  = processed
        pbar.update()

#Symbols that cannot be converted to vex IR
def unvexable_symbols(db, bin_path):
    query = [
        { '$match' : { 'path' : bin_path } },
        { '$match' : { 'size' : { '$gte': 5000 } } }
    ]
    symbs = []
    res = db.symbols.aggregate( query )
    for res_it in res:
        symb = { "name": res_it['name'], "vaddr": res_it['vaddr'] }
        symbs.append( symb )
    return symbs

#Symbols that can be converted to vex IR
def vexable_symbols(db, bin_path):
    query = [
        { '$match' : { 'path' : bin_path } },
        { '$match' : { 'size' : { '$lt': 5000 } } }
    ]
    symbs = []
    res = db.symbols.aggregate( query )
    for res_it in res:
        symb = { "name": res_it['name'], "vaddr": res_it['vaddr'] }
        symbs.append( symb )
    return symbs


def ANGRIZE(data):
    address = "../../res/python2_angr.unix.socket"
    sock = socket.socket(socket.AF_UNIX, socket.AF_STREAM)
    s.connect( address )
    s.send( json.dumps( data ) )
    res = s.recv(2 ** 16)
    return res

if __name__ == "__main__":

    #get all symbols
    db_ref = Database()
    db = db_ref.client

    query = [
        { '$group' : { '_id' : '$path' } } 
    ]
    res = db.symbols.aggregate( query )
    for res_it in res:
        path = res_it['_id']
        white_list = vexable_symbols(db, path)
        black_list = unvexable_symbols(db, path)
        data = {
                "bin_path": path,
                "white_list": white_list,
                "black_list": black_list
        }
        pprint.pprint( data )
        break

    sys.exit(0)


    """


    unhashed_symbs = symbols.find({'hash' :  ""  })
    nsymbs = unhashed_symbs.count()
    print("[+] Unhashed symbols in DB: " + str(nsymbs))
    tl = []

    step = math.ceil( nsymbs / NUM_THREADS )
    with progressbar.ProgressBar(widgets=pbar_config,max_value=symbols.count()) as pbar:
        for s in [unhashed_symbs[i:i + step] for i in range(0, nsymbs, step)]:
            t = Thread(target=symbol_hasher_thread, args=(s, pbar))
            t.start()
            tl.append(t)

        for t in tl:
            t.join()

        pbar.value = pbar.max_value
        pbar.finish()
    """
