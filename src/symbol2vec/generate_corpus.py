import os, sys
import tensorflow as tf
import numpy as np
import random, logging
import itertools, tqdm
import tqdm
import collections, functools
import math
import context
import classes.utils
import classes.callgraph
import classes.config
import classes.experiment
import classes.database
from sklearn.model_selection import train_test_split
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd
import IPython

stop_symbols = set([ 'stack_chk_fail', 'stack_chk_fail_local', '', 'errno_location' ])

def symbols_from_binary(db, bin_path):
    coll_name=db.config.database.collection_name+db.config.database.symbol_collection_suffix
    res = db.run_mongo_aggregate([{'$match': {'path': bin_path}}, {'$project': {'path': 1, 'name': 1, 'callers':1, 'callees': 1}}], coll_name=coll_name)
    paths, names, callers, callees = [], [], [], []
    for r in res:
        paths.append(r['path'])
        names.append(r['name'])
        callers.append(r['callers'])
        callees.append(r['callees'])
    return pd.DataFrame.from_dict({'name': names, 'path': paths, 'callers': callers, 'callees': callees })

if __name__ == '__main__':

    conf = classes.config.Config()
    db   = classes.database.Database(conf)
    E = classes.experiment.Experiment(conf)
    E.load_settings()

    binaries = set(db.distinct_binaries())

    global_sentences = []
    for path in tqdm.tqdm(binaries):
        df = symbols_from_binary(db, path)

        ##walk all 3 steps
        sentences = []
        for i in df.index:
            for callee in df['callees'][i]:
                #skip self calls
                if callee == df['name'][i]:
                    continue
                for callee2_records in df.loc[df['name'] == callee]['callees'].values:
                    for callee2 in callee2_records:
                        #skip self calls
                        if callee2 == callee:
                            continue
                        sentences.append( [ df['name'][i], callee, callee2 ] )
                        global_sentences += sentences

    with open('/tmp/fastText_corpus.txt', 'w') as f:
        for sentence in global_sentences:
            f.write(' '.join(sentence) + "\n")

