#!/usr/bin/python3
import multiprocessing
import progressbar
import logging
import itertools

import context
from classes.config import Config
from classes.database import Database
import perform_analysis

cfg = Config()
logger = logging.getLogger( cfg.logger )
logging.basicConfig(level=logging.INFO , format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


pbar_config = [ ' [ ',  progressbar.Counter(format='%(value)d / %(max_value)d'), ' ] ',  progressbar.Percentage(), ' [', progressbar.Timer(),
                progressbar.Bar(), ' (', progressbar.ETA(), ')'
]

query_config = {
    'linkages' :        [ "dynamic" ],
    'compilers' :       [ "gcc", "gcc" ],
    'optimisations':    [ "1", "2" ],
}

symbol_projection = { 'size' : 1, 'name' : 1, 'path': 1, 'hash': 1, 'opcode_hash': 1, 'cfg': 1,
    'vex.statements': 1, 'vex.operations': 1, 'vex.expressions': 1, 'vex.ntemp_vars': 1,
    'vex.temp_vars': 1, 'vex.sum_jumpkinds': 1, 'vex.jumpkinds': 1, 'vex.ninstructions': 1,
    'vex.constants': 1, 'callers': 1, 'callees': 1, 'bin_name': 1, 'vaddr': 1
}

NUM_PROCESSES = 32

global similarity_hashmap
global MPPool
global PBar

def compute_sim_cb(res):
    global PBar
    PBar.value += len(res)

def compute_sim_err_cb(err):
    logger.error("You fucked up!")
    logger.error(err)

def async_compute_similarity( symb_pair ):
    a, b = symb_pair
    sim = a.similarity(b)
    return (sim, perform_analysis.check_similarity_of_symbol_name(a.name, b.name))

#recursive design hits recursion limit
def compare_symbols( symbols, max_len ):
    global MPPool
    global similarity_hashmap
    global PBar


    for jt in range(0, max_len):
        #start comparison of 1 rows of top right corner of adjacency matrix

        PBar.max_value += max_len - (jt+1)
        #similarity_hashmap[ (i, it) ] = MPPool.apply_async( async_compute_similarity, (symbols[i], symbols[it]), callback=compute_sim_cb, error_callback=compute_sim_err_cb)
        similarity_hashmap[ jt ] = MPPool.map_async( async_compute_similarity, zip( itertools.repeat(symbols[jt]) , [ symbols[i] for i in range(jt+1, max_len) ] ),
                callback=compute_sim_cb, error_callback=compute_sim_err_cb, chunksize=max_len)

if __name__ == "__main__":
    log_file = logging.FileHandler(cfg.desyl + "/res/generate_similarities.log")
    logger.addHandler(log_file)
    logger.setLevel(logging.INFO)

    db_ref = Database()
    query = Database.gen_query( query_config, projection=symbol_projection, sample=2000 )
    logger.debug("Using query : {}".format(query))
    symbols = db_ref.get_symbols("symbols", query)

    MAX_LEN = len(symbols)
    logger.info("Fetched and parsed symbols!")
    logger.info("{} Symbols. N**2 -N/2 combinations == {}".format( MAX_LEN, ((MAX_LEN ** 2)-MAX_LEN)/ 2))

    #compute symbols^2 similarities
    #only compute top right corner of similarity matrix
    global similarity_hashmap
    global MPPool
    global PBar

    similarity_hashmap = {}
    MPPool = multiprocessing.Pool(NUM_PROCESSES)
    PBar = progressbar.ProgressBar(widgets=pbar_config,max_value=0)

    compare_symbols( symbols, MAX_LEN)
    PBar.update()

    results = []

    for k, v in similarity_hashmap.items():
        while True:
            v.wait(1) #1s timeout
            if v.ready():
                #sim_vec = r.get()
                results.append( v.get() )
                break
            PBar.update()

    PBar.value = PBar.max_value
    PBar.update()
    PBar.finish()
    logger.info("Finished Processing!")

    gen_sim_file = cfg.desyl + "/res/gen_sim.json"
    logger.info("Writing to {}".format( gen_sim_file ))
    with open(gen_sim_file, 'w') as f:
        for res_t in results:
            for res in res_t:
                f.write("{}\t{}\n".format( res[0].tolist(), [1] if res[1] else [0] ))
        


