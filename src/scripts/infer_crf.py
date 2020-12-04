#!/usr/bin/python3
import sys
import unittest
import logging
import networkx as nx
import numpy as np
import scipy
import os
import glob
import copy
import math
import gc
import itertools
from joblib import Parallel, delayed
import random
import IPython
import tqdm

from networkx.drawing.nx_agraph import write_dot

import context
from classes.config import Config
from classes.database import Database, PostgresDB, RedisDB
import classes.annoydb
from classes.experiment import Experiment
from classes.crf import CRF
import classes.callgraph
from classes.binary import Binary
from classes.symbol import Symbol
from classes.crf_greedy_inference import GreedyCRFInference
from classes.crf_sum_product_belief import CRFSumProductBeliefPropagation
import time

def infer_crf(config, db, Exp, callgraph, learned):
    """
        Train a CRF from a callgraph
    """
    factor_rels = {}
    #constraints = np.ones( (Exp.name_vector_dims, ), dtype=np.float64 )

    ll_rels, ln_rels = learned

    crf = CRF(config, Exp, callgraph.knowns, callgraph.unknowns, callgraph.G, ln_rels, ll_rels, factor_rels )

    sp_crf = CRFSumProductBeliefPropagation(crf)

    sp_crf.infer()
    sp_crf.check_accuracy(top_n=5, confidence=0.2)

    return sp_crf

if __name__ == '__main__':
    config = Config()
    db = Database(config)
    pdb = PostgresDB(config)
    pdb.connect()
    exp = Experiment(config)
    exp.load_settings()

    np.seterr(over='raise')

    #load fingerprint classifier
    #clf = classes.utils.pickle_load_py_obj(config, "rfc.clf")
    adb = classes.annoydb.AnnoyDB(config)
    adb.load()
    binaries = classes.utils.pickle_load_py_obj(config, 'test_binaries')
    ##randomly shuffle binaries for this epoch
    random.shuffle(binaries)

    knc_clf         = classes.utils.pickle_load_py_obj(config, "knc_clf")
    pca             = classes.utils.pickle_load_py_obj(config, "pca")
    clf_pipeline    = lambda x: classes.callgraph.Callgraph.clf_proba_inf(knc_clf, exp, pca.transform(x))


    learned = exp.load_experiment_key('crf_relationships')
    #IPython.embed()

    #callgraphs = map(delayed(classes.callgraph.build_crf_for_binary_adb), itertools.repeat(pdb), itertools.repeat(exp), itertools.repeat(adb), binaries)
    #callgraphs = map(delayed(classes.callgraph.build_crf_for_binary_clf), itertools.repeat(pdb), itertools.repeat(exp), itertools.repeat(clf), binaries)
    callgraphs = map(delayed(classes.callgraph.Callgraph.from_clf), itertools.repeat(config), itertools.repeat(exp), itertools.repeat(clf_pipeline), itertools.repeat(pdb), binaries)
    for i, (func, binary) in enumerate(tqdm.tqdm(zip(callgraphs, binaries))):
        print("{} : Testing CRF on binary {}".format(i, binary))

        callgraph = func[0](*func[1:][0])
        crf = infer_crf(config, db, exp, callgraph, learned)
        IPython.embed()

    print("Finished inference.")
