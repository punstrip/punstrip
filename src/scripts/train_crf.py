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
import posix_ipc
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

NEW_MODEL       = False

def init_model_weights(Exp): 
    """
    Initialize a new CRF model weights. 
    Retruns label-label and label-node weightings
    """
    fp_thetas = np.ones(( Exp.name_vector_dims, ))

    callee_rels         = scipy.sparse.csc_matrix(( Exp.name_vector_dims, Exp.name_vector_dims), dtype=np.float64)
    caller_rels         = scipy.sparse.csc_matrix(( Exp.name_vector_dims, Exp.name_vector_dims), dtype=np.float64)

    #callee_rels        = np.ndarray(( Exp.name_vector_dims, Exp.name_vector_dims), dtype=np.float64)
    #caller_rels        = np.ndarray(( Exp.name_vector_dims, Exp.name_vector_dims), dtype=np.float64)
    #known_callee_rels  = np.ndarray(( Exp.known_name_vector_dims, Exp.name_vector_dims), dtype=np.float64)


    """
        known_caller relationship if x has a know that calls it
        known_callee relationship if x calls a known
    """
    ln_rels = { 
            'fingerprint'   : fp_thetas,
    }
    ll_rels = { 
        'caller' : caller_rels, 
        'callee' : callee_rels
    }

    for d in range(1, 4):
        known_callee_rels   = scipy.sparse.csc_matrix(( Exp.known_name_vector_dims, Exp.name_vector_dims), dtype=np.float64)
        known_caller_rels   = scipy.sparse.csc_matrix(( Exp.known_name_vector_dims, Exp.name_vector_dims), dtype=np.float64)
        thetas = {
            'known_caller_{}'.format(d)  : known_caller_rels,
            'known_callee_{}'.format(d)  : known_callee_rels
        }
        ln_rels.update(thetas)

    return ll_rels, ln_rels

def interpolate_weights(old, new, alpha=1e-2):
    """
        SGD between old and new weights
    """
    old_ll_rels, old_ln_rels = old
    new_ll_rels, new_ln_rels = new

    for rel in new_ln_rels:
        diff = old_ln_rels[rel] - new_ln_rels[rel]
        old_ln_rels[rel] -= alpha * diff

    for rel in new_ll_rels:
        diff = old_ll_rels[rel] - new_ll_rels[rel]
        old_ll_rels[rel] -= alpha * diff

    learned = old_ll_rels, old_ln_rels
    return learned



def train_crf(config, db, Exp, callgraph, learned, alpha=1e-3):
    """
        Train a CRF from a callgraph
    """
    #knowns      = list(filter(lambda x,G=callgraph: G.nodes[x]['func'] and not G.nodes[x]['text_func'], callgraph.nodes()))
    #unknowns    = list(filter(lambda x,G=callgraph: G.nodes[x]['func'] and G.nodes[x]['text_func'], callgraph.nodes()))

    if len(callgraph.unknowns) <= 0:
        return learned

    ##constraints. Do not allow knows to be represented
    constraints = np.ones( (Exp.name_vector_dims, ), dtype=np.float64 )

    ll_rels, ln_rels = learned

    crf = CRF(config, Exp, callgraph.knowns, callgraph.unknowns, callgraph.G, ln_rels, ll_rels, constraints=constraints)
    sp_crf = CRFSumProductBeliefPropagation(crf)

    #print("Created CRF.. About to train")
    #IPython.embed()

    sp_crf.train(alpha)
    #sp_crf.infer()
    #sp_crf.marginals_from_fps()
    sp_crf.check_accuracy(top_n=5, confidence=0.1)

    print("Trained CRF model")
    #IPython.embed()
    learned = sp_crf.ll_relationships, sp_crf.ln_relationships
    return learned, sp_crf

if __name__ == '__main__':
    posix_ipc.Semaphore('crf_updates').unlink()
    sem = posix_ipc.Semaphore('crf_updates', flags=posix_ipc.O_CREAT, initial_value=1)
    config = Config()
    db = Database(config)
    pdb = PostgresDB(config)
    pdb.connect()
    exp = Experiment(config)
    exp.load_settings()

    np.seterr(over='raise')

    learned = None
    if NEW_MODEL:
        print("Generating new model")
        learned = init_model_weights(exp)
        sem.acquire()
        exp.update_experiment_key('crf_relationships', learned)
        sem.release()
        print("Done. please disable new model generation and rerun this script")
        sys.exit()
        #raise RuntimeError("New model not supported in multi-process training, please create a blank new model first")

    #load fingerprint classifier
    #clf = classes.utils.pickle_load_py_obj(config, "rfc.clf")
    adb = classes.annoydb.AnnoyDB(config)
    adb.load()
    #train_df = classes.utils.pickle_load_py_obj('train_df')
    binaries = classes.utils.pickle_load_py_obj(config, 'train_binaries')
    #binaries = classes.utils.pickle_load_py_obj(config, 'test_binaries')
    ##randomly shuffle binaries for this epoch
    random.shuffle(binaries)
    #binaries = [ '/dbg_elf_bins/myproxy/usr/bin/myproxy-store' ]
    #binaries = binaries[:1]
    #binaries = ['/dbg_elf_bins/umview/usr/lib/umview/umbinwrap' ]

    """
    SHARED_MEMORY_WEIGHTS_FOLLOWER = False
    if not SHARED_MEMORY_WEIGHTS_FOLLOWER:
        sem.acquire()
        config.logger.info("Loading CRF model weights...")
        learned = exp.load_experiment_key('crf_relationships')
        sem.release()
    else:
        desc = classes.utils.load_py_obj(config, "crf_model_weights_desc")
    """
    #learned = init_model_weights(exp)
    learned = exp.load_experiment_key('crf_relationships')

    CHUNK_SIZE = 1

    knc_clf         = classes.utils.pickle_load_py_obj(config, "knc_clf")
    pca             = classes.utils.pickle_load_py_obj(config, "pca")
    clf_pipeline    = lambda x: classes.callgraph.Callgraph.clf_proba_inf(knc_clf, exp, pca.transform(x))


    #callgraph = classes.callgraph.Callgraph.from_clf(config, exp, clf_pipeline, pdb, path)

    g_callgraph = None
    #callgraphs = map(delayed(classes.callgraph.build_crf_for_binary_adb), itertools.repeat(pdb), itertools.repeat(exp), itertools.repeat(adb), binaries)
    #callgraphs = map(delayed(classes.callgraph.build_crf_for_binary_clf), itertools.repeat(pdb), itertools.repeat(exp), itertools.repeat(clf), binaries)
    callgraphs = map(delayed(classes.callgraph.Callgraph.from_clf), itertools.repeat(config), itertools.repeat(exp), itertools.repeat(clf_pipeline), itertools.repeat(pdb), binaries)
    for i, (func, binary) in enumerate(tqdm.tqdm(zip(callgraphs, binaries))):
        print("{} : Training CRF on binary {}".format(i, binary))

        #if SHARED_MEMORY_WEIGHTS_FOLLOWER:
        #    weights = classes.crf_sum_product_belief.CRFSumProductBeliefPropagation.load_weights_from_shared_memory(desc)

        callgraph = func[0](*func[1:][0])
        learned, crf = train_crf(config, db, exp, callgraph, learned, alpha=1e-3)
        if (i+1) % 5 == 0:
            CRF.save_relationships(exp, learned)
        print("Trained CRF")
        #IPython.embed()
        #sys.exit()
        continue

        if g_callgraph == None:
            g_callgraph = callgraph
        else:
            g_callgraph.update(callgraph)

        if (i+1) % CHUNK_SIZE != 0:
            continue

        if len(g_callgraph.unknowns) > 0:
            """
            print("Waiting for model weights semaphore lock...")
            sem.acquire()
            config.logger.info("Loading CRF model weights...")
            learned = exp.load_experiment_key('crf_relationships')
            sem.release()
            """
            learned, crf = train_crf(config, db, exp, g_callgraph, learned, alpha=1e-3)
            #desc = crf.weights_to_shared_memory()
            #classes.utils.save_py_obj(config, desc, 'crf_model_weights_desc')

            """
            sem.acquire()
            config.logger.info("Loading CRF model weights...")
            new_old_learned = exp.load_experiment_key('crf_relationships')
            ##save backup increase we are killed during write
            CRF.save_relationships(exp, learned, key='crf_relationships.backup')
            ##interpolate between learned and new_old_learned
            learned = interpolate_weights(new_old_learned, learned, alpha=1e-2)
            config.logger.info("Saving model weights")
            CRF.save_relationships(exp, learned)
            sem.release()
            """

        ##release previous resources
        del g_callgraph
        g_callgraph = None
        gc.collect()

    print("Finished training.")
    IPython.embed()
    """
    print("Finished training.")
    print("Interpolating final weights")
    sem.acquire()
    config.logger.info("Loading CRF model weights...")
    new_old_learned = exp.load_experiment_key('crf_relationships')
    ##interpolate between learned and new_old_learned
    learned = interpolate_weights(new_old_learned, learned, alpha=1e-2)
    config.logger.info("Saving model weights")
    CRF.save_relationships(exp, learned)
    sem.release()
    """
