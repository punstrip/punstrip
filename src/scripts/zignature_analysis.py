#!/usr/bin/python3
import sys, os
import logging
import glob, gc
import copy
import progressbar
import traceback
import re, json
import r2pipe

#from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat

import context
from classes.config import Config
from classes.binary import Binary
from classes.symbol import Symbol
from classes.database import Database

NUM_PROCESSES = 10
global pool
global pbar
global results

pbar_config = [ ' [ ',  progressbar.Counter(format='%(value)d / %(max_value)d'), ' ] ',  progressbar.Percentage(), ' [', progressbar.Timer(),
                progressbar.Bar(), ' (', progressbar.ETA(), ')'
]

def find_symbol(db, bin_path, address, size):
    return db.client.symbols.find_one({ 'path' : bin_path, 'vaddr': address, 'size': size})

def config_to_bin_paths(config, stripped):
    bins=[]
    bins_path = "bin-stripped" if stripped else "bin"

    for a in config['archs']:
        if a == "x86_64":
            mod_bins_path = copy.deepcopy(bins_path)
        elif a == "ARMv7":
            mod_bins_path = bins_path.replace("bin", "bin-armv7")

        for b in config['bin_names']:
            for c in config['compilers']:
                for o in config['optimisations']:
                    for l in config['linkages']:
                        print(mod_bins_path)
                        bins.append("{}/{}/{}/{}/o{}/{}".format( cfg.corpus, mod_bins_path, l, c, o, b))

    return bins


def r2_infer_symbols(train_config, test_config):

    ( train_query_config, min_projection ) = train_config
    ( test_query_config, min_projection ) = test_config

    #print(train_config)

    assert( "bin_names" in train_query_config.keys() )
    assert( "bin_names" in test_query_config.keys() )

    #build list of bin_paths for training data
    train_bins  = config_to_bin_paths(train_query_config, False)
    test_bins   = config_to_bin_paths(test_query_config, True)

    train_sdbs = list(map(lambda x: x + ".sdb", train_bins))

    #print(train_sdbs)
    #sys.exit()
    
    logger.info("Starting infer process")
    total_correct, total_incorrect, total_symbols = 0, 0, 0
    for test_b in test_bins:
        if not os.path.isfile(test_b):
            continue

        logger.info("Inferring symbols in {}".format(test_b))
        correct, incorrect, num_symbols = _r2_infer_symbols( test_b, train_sdbs )
        total_correct += correct
        total_incorrect += incorrect
        total_symbols += num_symbols
        __calculate_f1(correct, incorrect, num_symbols)
        logger.info("Correct: {}, Incorrect: {}, Total symbols: {}".format( correct, incorrect, total_symbols ))
        sys.exit()

    return total_correct, total_incorrect, total_symbols

def __calculate_f1(correct, incorrect, total, loggerout=logging.INFO):
    """
        here total is the total amount of correct symbols
    """
    tp = float(correct)
    tn = 0.0
    fp = float(incorrect)
    fn = float(total - correct)

    loggerout("TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
    
    precision   = tp / (tp+fp)
    recall      = tp / (tp+fn) 
    f1 = (2 * precision * recall) / ( precision + recall )

    loggerout("Precision : {}, Recall : {}, F1 : {}".format( precision, recall, f1))


def _r2_infer_symbols(bin_path, sdb_paths, names=False):
    assert( "stripped" in bin_path)
    unstripped_bin_path = bin_path.replace("bin-stripped", "bin")
    matched_correct, matched_incorrect = 0, 0

    logger.info("Opening {}".format(bin_path))
    pipe = r2pipe.open(bin_path, ["-2"])

    optimized_sets = set()
    for zig_db in sdb_paths:
        logger.debug("Using sbd file: {}".format(zig_db))

        optimized_sets.add( os.path.dirname( zig_db ))
        if not os.path.isfile(zig_db):
            continue

        #logger.debug("r2 loading zignatures from {}".format(zig_db))
        #pipe.cmd("zo {}".format(zig_db))

    for db in optimized_sets:
        logger.debug("r2 loading zignatures from {}/r2.sdb".format(db))
        pipe.cmd("zo {}/r2.sdb".format(db))


    #print("Saving all training zignatures to {}/r2.sdb".format(os.path.dirname(  sdb_paths[0] ) ))
    #pipe.cmd('zos {}/r2.sdb'.format( os.path.dirname( sdb_paths[0] ) )) 
    #sys.exit()

    #match zignatures
    logger.debug("matching zignatures!")
    pipe.cmd("z/")
    logger.debug("matched zignatures!")

    db = Database()
    logger.info("Checking correctness of results!")

    #res = pipe.cmd("fj~sign")
    res = pipe.cmd("fj")
    matches = json.loads(res)
    matched = set()
    for match in matches:
        #not a ymbol match
        if "sym." not in match['name']:
            continue

        #dynamic function matched
        if "sym.imp." in match['name']:
            continue

        #{"name":"sign.bytes.sym.__libc_csu_init_0","size":101,"offset":9040},{"name":"sign.bytes.main_0","size":427,"offset":8607}
        #print(match)
        symbol = find_symbol(db, unstripped_bin_path, match['offset'], match['size'])

        if not symbol:
            #print("Match incorrect")
            matched_incorrect += 1
            continue

        #limit matched to symbols in names
        if names:
            if symbol['name'] not in names:
                continue

        if symbol['name'] in match['name']:
            logger.info("[+] Matched {} :: {}".format(symbol['name'], match['name']))
            matched_correct += 1
        else:
            matched_incorrect += 1
            logger.info("Match incorrect. r2 match: {}. Correct symbol name: {}".format( match['name'], symbol['name'] ) )
        
    pipe.quit()

    #get total number of symbols for bin
    unstripped_path = bin_path.replace('/bin-stripped/', '/bin/')
    total_symbols = db.client.symbols.find({'path': unstripped_path }).count()
    #print(unstripped_path)
    #print(total_symbols)
    #sys.exit()

    return matched_correct, matched_incorrect, total_symbols

if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    train_bins  = cfg.train['bin_names']
    test_bins   = cfg.test['bin_names']

    configs = [
                [ 
                    { 'optimisations' : [ 'g' ], 'linkages' : [ 'dynamic' ], 'compilers' : [ 'gcc' ], 'bin_names' : train_bins, 'archs': [ 'ARMv7']     },
                    { 'optimisations' : [ 'g' ], 'linkages' : [ 'dynamic' ], 'compilers' : [ 'gcc' ], 'bin_names' : test_bins, 'archs': [ 'x86_64' ]     }
                ]
            ]
    for train, test in configs:
        logger.critical("Train config: {}".format(train))
        logger.critical("Test config: {}".format(test))
        zig_correct, zig_incorrect, total_symbols = r2_infer_symbols((train,{}), (test,{}))
        logger.critical("[+] Radare2 zignatures inferred {} correctly and misclassified {} symbols out of {}.".format( zig_correct, zig_incorrect, total_symbols))
        __calculate_f1(zig_correct, zig_incorrect, total_symbols, loggerout=logger.critical)

