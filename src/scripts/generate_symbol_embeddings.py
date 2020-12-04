#!/usr/bin/python3
import sys, os, re
import logging
import glob, gc
import progressbar
from tqdm import tqdm
import json
import IPython
import psycopg2
import functools
import pandas as pd
from joblib import Parallel, delayed

import context
from classes.config import Config
from classes.symbol import Symbol
import classes.crf
from classes.database import Database, PostgresDB
import classes.experiment

def par_gen_symbol_embeddings(chunks):
    config = classes.config.Config()
    exp = classes.experiment.Experiment(config)
    exp.load_settings()

    names, embeddings = [], []
    binaries = []
    for r in chunks:
        bin_id, binary_path, name, arguments, heap_arguments, tls_arguments, local_stack_bytes, num_args, ret, closure, callees, callers, sha256, opcode_hash, asm_hash, size, binding, vex, cfg, tainted_flows, dynamic_imports = r

        ##TODO: Need to include dynamic calls/callees to to_vec function
        known_functions = classes.crf.CRF.calculable_knowns | set(dynamic_imports)

        s = classes.symbol.Symbol(config, name=name, closure=closure, local_stack_bytes=local_stack_bytes,
                arguments=arguments, heap_arguments=heap_arguments, num_args=num_args, binding=binding, cfg=cfg, callers=callers, 
                callees=callees, opcode_hash=bytes(opcode_hash), hash=bytes(sha256), vex=vex, tainted_flows=tainted_flows,
                size=size)

        embedding = s.to_vec(exp, KNOWN_FUNCS=known_functions)
        names.append(name)
        embeddings.append(embedding)
        binaries.append(binary_path)

    return names, embeddings, binaries

def generate_symbol_embeddings(config):
    """
        returns [names], [embeddings], [binary_path]
    """
    db = classes.database.PostgresDB(config)
    db.connect()

    curr = db.conn.cursor()
    #curr.execute("SELECT binary_id, public.binary.path, public.binary_functions.name, arguments, heap_arguments, tls_arguments, local_stack_bytes, num_args, return, closure, callees, callers, public.binary_functions.sha256, opcode_hash, asm_hash, public.binary_functions.size, binding, vex, cfg, tainted_flows, public.binary.dynamic_imports FROM public.binary_functions RIGHT JOIN public.binary ON binary_id=public.binary.id WHERE binding = 'GLOBAL'")
    curr.execute("SELECT binary_id, public.binary.path, public.binary_functions.name, arguments, heap_arguments, tls_arguments, local_stack_bytes, num_args, return, closure, callees, callers, public.binary_functions.sha256, opcode_hash, asm_hash, public.binary_functions.size, binding, vex, cfg, tainted_flows, public.binary.dynamic_imports FROM public.binary_functions LEFT JOIN public.binary ON binary_id=public.binary.id")

    """
        We copy objects into python because we can't pickle memory view objects from Postgres
    """
    symbol_buffer = []
    for r in tqdm(curr.fetchall(), desc="Loading symbols from database"):
        datum = [ r[i] for i in range(12) ] + [ bytes(r[12]), bytes(r[13]), bytes(r[14]) ] + [ r[i] for i in range(15, len(r)) ]
        symbol_buffer.append(datum)

    ##single threaded
    #par_gen_symbol_embeddings(symbol_buffer)
    #sys.exit(0)

    chunks = classes.utils.n_chunks(symbol_buffer, 256)
    results = Parallel(n_jobs=120, verbose=1, backend="multiprocessing")(map(delayed(par_gen_symbol_embeddings), chunks))
    _names, _embeddings, _binaries = zip(*results)
    names = functools.reduce(lambda x, y: x + y, _names, [])
    embeddings = functools.reduce(lambda x, y: x + y, _embeddings, [])
    binaries = functools.reduce(lambda x, y: x + y, _binaries, [])
    return names, embeddings, binaries

def export_symbol_embeddings(config, names, embeddings):
    classes.utils.pickle_save_py_obj(config, [names, embeddings], "symbol.desyl.embeddings")

if __name__ == "__main__":
    config = Config(level=logging.INFO)
    if len(sys.argv) == 2:
        config.logger.info("Enabling DEBUG output")
        config.logger.setLevel(logging.DEBUG)

    config.logger.info("[+] Generating symbol embeddings for current experiment settings")
    names, embeddings, binaries = generate_symbol_embeddings(config)
    config.logger.info("[+] Exporting symbol embeddings to DILL file")
    export_symbol_embeddings(config, names, embeddings)

    ##creates a dataframe with all info stored
    df              = pd.DataFrame(names, columns=['name'])
    bin_df          = pd.DataFrame(binaries, columns=['binary'])
    embeddings_df   = pd.DataFrame({'embedding': embeddings})
    df['binary']    = bin_df['binary']
    df['embedding'] = embeddings_df['embedding']
    df.to_pickle('/tmp/symbol_embeddings_df')

    print("Done!")
    IPython.embed()
