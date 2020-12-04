#!/usr/bin/python3
import sys
import unittest
import logging
import networkx as nx
import numpy as np
import scipy
import copy
import random
import IPython

from networkx.drawing.nx_agraph import write_dot

import context
from classes.config import Config
from classes.database import Database
from classes.experiment import Experiment
from classes.crf import CRF
from classes.NLP import NLP
import classes.callgraph
from classes.binary import Binary
from classes.symbol import Symbol
from classes.crf_greedy_inference import GreedyCRFInference
from classes.crf_sum_product_belief import CRFSumProductBeliefPropagation
import time
import IPython

if __name__ == '__main__':
    ###########################################################################
    USE_LOCALLY_CACHED_CALLGRAPH    = True
    REUSE_SAVED_RELATIONSHIPS       = True
    MODE                            = "FULL" # "FULL", "RANDOM_SPLIT"
    train_binary_path               = '/root/mirai'
    test_binary_path                = '/root/mirai.orig'
    RANDOM_SPLIT_TEST_UNKNOWNS      = 500
    TRAIN_CRITERIA                  = { 'path' : train_binary_path }
    TEST_CRITERIA                   = { 'path' : test_binary_path }
    ###########################################################################

    ##raise overflow warnings
    np.seterr(over='raise')

    config = Config()
    db = Database(config)
    nlp = NLP(config)
    Exp = Experiment(config)
    Exp.load_settings()

    config.logger.info("Loading fingerprint model...")
    #clf         = Exp.load_experiment_key('fingerprint_model')
    clf          = classes.utils.load_py_obj(config, 'rfc')

    #######################Load callgraph###################################### 
    config.logger.info("Constructing callgraph with fingerprints for inference")
    if not USE_LOCALLY_CACHED_CALLGRAPH:
        callgraph = classes.callgraph.build_crf_for_binary(db, Exp, clf, test_binary_path, debug=True)
        classes.utils.save_py_obj(config, callgraph, 'callgraph')
    else:
        callgraph = classes.utils.load_py_obj(config, 'callgraph')
    ########################################################################### 

    ###Load CRF relationships##################################################
    try: 
        if not REUSE_SAVED_RELATIONSHIPS:
            raise Exception("Error, generate new relationships")

        config.logger.info("Loading CRF model weights...")
        ll_rels, ln_rels = Exp.load_experiment_key('crf_relationships')

        #config.logger.info("Found preexisting trained CRF relationships")
        #ll_rels, ln_rels = classes.utils.py_obj_from_bytes( crf_relationships )
    except:
        config.logger.info("Generating new CRF relationships")
        fp_thetas = scipy.ones(( Exp.name_vector_dims,), dtype=np.float64) / Exp.name_vector_dims
        ln_rels = { 'fingerprint' : fp_thetas }

        #ll_rels = { 
        #    'caller' : { 'f' : call_rels, 'w': call_thetas },
        #    'callee' : { 'f' : call_rels, 'w': call_thetas }
        #}
        ll_rels = CRF.callgraph_to_relationships(Exp, callgraph)
    ###########################################################################


    ##load symbols from database
    symbols = db.get_symbols_from_binary( test_binary_path )


    ######
    ###### Calculate true unknowns that exit in test set.
    ######
    ##true unknowns in binary, dynamic imports loaded through relationships in callgraph
    true_unknowns = set(map(lambda x: x.name, filter(lambda x: x.size > 0 and x.binding == "GLOBAL", symbols)))


    ###
    symbol_names_in_train   = set(map(lambda x: x['name'], db.run_mongo_aggregate([{ '$match' : TRAIN_CRITERIA}, {'$project': { "name" :1 }}])))
    symbol_names_in_test    = set(map(lambda x: x['name'], db.run_mongo_aggregate([{ '$match' : TEST_CRITERIA}, {'$project': { "name" :1 }}])))

    true_unknowns = true_unknowns.union(symbol_names_in_train).union(symbol_names_in_test)




    ##TODO, Add data relationships
    funcs   = list(filter(lambda x, cg=callgraph: cg.nodes[x]['func'], callgraph.nodes)) 
    unknowns= list(filter(lambda x, cg=callgraph: cg.nodes[x]['text_func'], funcs))
    knowns  = list(filter(lambda x, cg=callgraph: not cg.nodes[x]['text_func'], funcs))

    ##extract fingerprints from callgraph
    node_fingerprints = dict(map(lambda x: [ x, callgraph.nodes[x]['node_potential'] ], callgraph.nodes()))
    ln_values = { 'fingerprint' : node_fingerprints }

    ## randomly split nodes into known/unknown
    if MODE == "RANDOM_SPLIT":
        TEST_UNKNOWNS = RANDOM_SPLIT_TEST_UNKNOWNS

        all_funcs = random.sample(funcs, len(funcs))
        #randomly split into 2
        knowns, unknowns = list(classes.utils.n_chunks(all_funcs, 2))

        knowns += unknowns[:-TEST_UNKNOWNS]
        unknowns = unknowns[-TEST_UNKNOWNS:]

        constraints = scipy.ones( (Exp.name_vector_dims, ), dtype=scipy.float64 )

    ##do full binary inference as if we did not known any symbols (apart from 
    #dynamically imported functions
    elif MODE == "FULL":

        ##disallow a symbol name if it is already known
        constraints = scipy.ones( (Exp.name_vector_dims, ), dtype=scipy.float64 )
        for known in knowns:
            constraints[ Exp.to_index('name_vector', known), 0 ] = 0

    else:
        raise RuntimeError("Unknown CRF Inference mode: {}".format(MODE))

    ###########################################################################
    ##################Here comes the magic, dependencies loaded################
    ###########################################################################

    ###########################################################################
    ##finally create CRF for binary
    config.logger.info("Building CRF model!")
    crf = CRF(config, Exp, knowns, unknowns, ln_values, callgraph, ln_rels, ll_rels, constraints=constraints)
    crf.remove_irrelevant()
    crf.save_to_db()
    crf.hide_unknowns()
    IPython.embed()
    ###########################################################################

    #crf.save('/tmp/{}.hidden'.format("mirai"))
    #fg = crf.generate_factor_graph()
    #CRF.save_model(fg.model, '/tmp/{}.hidden.fg'.format("mirai"))

    """
    #IPython.embed()
    fg = crf.generate_factor_graph()
    crf.save('/tmp/{}'.format("mirai"))
    crf.hide_unknowns()
    CRF.save_model(fg.model, '/tmp/mirai.fg')

    CRF.save_model(crf.generate_factor_graph().model, '/tmp/mirai.hidden.fg')
    #TODO compute marginals
    #opt_score, opt_name = crf.compute_max_marginal(1)
    #print(opt_score, opt_name)
    """

    ###########################################################################
    #CRF Greedy Inference
    """
    gCRF = GreedyCRFInference(crf)
    gCRF.infer()
    gCRF.check_accuracy()
    sys.exit(0)
    """

    #layman marginals exact inference
    #crf.check_accuracy()
    #crf.compute_max_marginals()
    #crf.save('/tmp/mirai.inferred')
    #crf.check_accuracy()
    #sys.exit(0)


    ###########################################################################
    #CRF LBP Inference
    sp_crf = CRFSumProductBeliefPropagation(crf)
    sp_crf.hide_unknowns()


    sp_crf.check_accuracy()
    sp_crf.load_from_db()
    sp_crf.infer()
    sp_crf.check_accuracy()
    IPython.embed()
    sys.exit()
    sp_crf.train()
    config.logger.info("Trained! Saving to database")
    sp_crf.save_to_db()
    sp_crf.check_accuracy()
    sp_crf.check_accuracy_extended(nlp, true_unknowns)
    IPython.embed()

    sys.exit()
    print("About to infer...")
    IPython.embed()
    sp_crf.infer()
    sp_crf.check_accuracy()

