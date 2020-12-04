#!/usr/bin/python3
import logging
import re
import numpy as np
import json
import functools, itertools
from io import BytesIO
import binascii
import sys, random
from multiprocessing import Pool
from annoy import AnnoyIndex
from tqdm import tqdm

import context
from classes.config import Config
from classes.database import Database
from classes.symbol import Symbol
import classes.counter
import classes.utils
import crf.crfs
import classes.callgraph

def gen_new_symbol_indexes(db):
    distinct_symbols = set( db.distinct_symbol_names() )
    symbs_from_rels = db.flatten_callees_callers()
    distinct_symbols = list( distinct_symbols.union( set(symbs_from_rels) ) )
    dim = len(distinct_symbols)
    logger.info("Found {} unique symbol names!".format( dim ))

    symbol_to_index = dict( map( lambda x: [distinct_symbols[x], x], range(dim)) )
    index_to_symbol = dict( map( lambda x: [x, distinct_symbols[x]], range(dim)) )
    return distinct_symbols, symbol_to_index, index_to_symbol


def gen_new_symbol_indexes_from_xrefs(db):
    binaries = db.get_number_of_xrefs()
    binaries_with_xrefs = set( {k: v for k, v in binaries.items() if v > 0}.keys() )

    N = len(binaries_with_xrefs)
    config.logger.info("Found {} binaries with XREFS".format(N))
    config.logger.info("Splitting into training and testing set")

    """
    k = int(N * 0.95)
    config.logger.info("Generating random sample of {} binaries for training sample".format(k))
    training_bins = random.sample(binaries_with_xrefs, k)

    config.logger.info("Taking difference for {} testing binaries".format(N - k))
    testing_bins = binaries_with_xrefs.difference( training_bins )

    config.logger.info("Saving testing and training bins")
    classes.utils.save_py_obj( config, training_bins, "training_bins" )
    classes.utils.save_py_obj( config, testing_bins, "testing_bins" )
    classes.utils.save_py_obj( config, training_bins + testing_bins, "unknown_bins" )


    config.logger.info("Generating set of unknown symbols and symbols in xrefs")
    res = db.run_mongo_aggregate([ { "$match": { "path" : { "$in" : list(training_bins) } } }, { "$project" : { "callees" : 1, "callers": 1, "name": 1} } ] )
    """
    res = db.run_mongo_aggregate([ { "$match": { "path" : { "$in" : list(binaries_with_xrefs) } } }, { "$project" : { "callees" : 1, "callers": 1, "name": 1} } ] )

    symbols = set([])
    for r in tqdm(res):
        for s in r['callers'] + r['callees'] + [ r['name'] ]:
            if s not in symbols:
                symbols.add(s)

    dim = len(symbols)
    distinct_symbols = list( symbols )

    logger.info("Found {} unique symbol names!".format( dim ))

    """
    symbol_to_index = dict( map( lambda x: [distinct_symbols[x], x], range(dim)) )
    index_to_symbol = dict( map( lambda x: [x, distinct_symbols[x]], range(dim)) )
    """
    return distinct_symbols, symbol_to_index, index_to_symbol

def build_and_save_symbol_name_indexes(db, config):
    config.logger.info("Generating distinct symbols and indexes...")
    #symbol_names, name_to_index, index_to_name = gen_new_symbol_indexes(db)
    symbol_names, name_to_index, index_to_name = gen_new_symbol_indexes_from_xrefs(db)

    classes.utils.save_py_obj( config, symbol_names, "symbol_names")
    classes.utils.save_py_obj( config, name_to_index, "name_to_index")
    classes.utils.save_py_obj( config, index_to_name, "index_to_name")

if __name__ == "__main__":

    config = classes.config.Config()
    logger = config.logger
    logger.setLevel(logging.INFO)
    db = classes.database.Database(config)

    #build_and_save_symbol_name_indexes(db, config)
    ##crf.crfs.build_all_cgs(db)

    training_bins = classes.utils.load_py_obj( config, "training_bins" )
    testing_bins = classes.utils.load_py_obj( config, "testing_bins" )
    #classes.callgraph.build_cgs_from_paths(list(training_bins)+list(testing_bins), CORPUS_NAME2INDEX=True)
    classes.callgraph.build_cgs_from_paths(list(training_bins))
