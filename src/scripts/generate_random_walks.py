#!/usr/bin/python3
import sys, os, re
import logging
import glob, gc
import progressbar
from tqdm import tqdm
import random
import json
import IPython
import psycopg2
import functools
import pandas as pd
from joblib import Parallel, delayed

import context
from classes.config import Config
from classes.symbol import Symbol
from classes.database import Database, PostgresDB
from classes.callgraph import Callgraph
import classes.utils

def par_gen_random_walks(chunks):
    config = classes.config.Config()
    pdb = classes.database.PostgresDB(config)
    walks = []

    pdb.connect()
    for path in chunks:
        cg = Callgraph(config, pdb, path)
        for walk in cg.random_walk_iter():
            walks.append(walk)

    random.shuffle(walks)
    return walks

def generate_random_walks(config):
    """
    """
    db = PostgresDB(config)
    db.connect()

    paths = list(db.binaries())
    #walks = par_gen_random_walks([next(paths)])
    #IPython.embed()
    #sys.exit(0)

    chunks = classes.utils.n_chunks(paths, 256)
    results = Parallel(n_jobs=64, verbose=1, backend="multiprocessing")(map(delayed(par_gen_random_walks), chunks))
    walks = functools.reduce(lambda x, y: x + y, results, [])
    random.shuffle(walks)
    return walks

def export_random_walks(config, rwalks, fname='random_walks'):
    classes.utils.pickle_save_py_obj(config, rwalks, "random_walks")
    with open('/tmp/random.walks.txt', 'w') as f:
        for walk in rwalks:
            f.write(' '.join(walk) + '\n')

if __name__ == "__main__":
    config = Config(level=logging.INFO)
    config.logger.info("[+] Generating random walks")
    walks = generate_random_walks(config)
    export_random_walks(config, walks)

    print("Done!")
    IPython.embed()
