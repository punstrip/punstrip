#!/usr/bin/python3
import sys
import math
import os
import glob
import progressbar
from threading import Thread, Lock

from classes.binary import Binary
from classes.database import Database
from classes.symbol import Symbol

NUM_THREADS = 16
processed = 0

pbar_config = [ '[ ', progressbar.Counter(), '] ', ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
]

def symbol_hasher(symbs, collection_name):
    db_ref = Database()
    db = db_ref.client

    for symb in symbs:
        s = Symbol.fromDatabase(db, collection_name, symb['_id'])
        s.gen_hashes()
        print(s.to_str_full())
        s.save_to_db(db, collection_name)



def symbol_hasher_thread(symbs, collection_name, pbar):
    db_ref = Database()
    db = db_ref.client

    for symb in symbs:
        s = Symbol.fromDatabase(db, collection_name, symb['_id'])
        s.gen_hashes()
        #print(s.to_str_full())
        s.save_to_db(db, collection_name)
        global processed
        processed += 1
        pbar.value  = processed
        pbar.update()


if __name__ == "__main__":

    #get all symbols
    db_ref = Database()
    db = db_ref.client
    collection_name = "symbols"

    if len(sys.argv) > 1:
        collection_name = sys.argv[1]

    print("[+] Hashing symbols from collection {}.".format(collection_name))
    
    unhashed_symbs = db[collection_name].find({'hash' :  ""  })
    nsymbs = unhashed_symbs.count()
    print("[+] Unhashed symbols in DB: " + str(nsymbs))
    if nsymbs == 0:
        print("[+] Done")
        sys.exit(0)

    tl = []

    for symb in unhashed_symbs:
        symbol_hasher([symb], collection_name)


    """
    step = math.ceil( nsymbs / NUM_THREADS )
    with progressbar.ProgressBar(widgets=pbar_config,max_value=nsymbs) as pbar:
        for s in [unhashed_symbs[i:i + step] for i in range(0, nsymbs, step)]:
            t = Thread(target=symbol_hasher_thread, args=(s, collection_name, pbar))
            t.start()
            tl.append(t)

        for t in tl:
            t.join()

        pbar.value = pbar.max_value
        pbar.finish()
    """
