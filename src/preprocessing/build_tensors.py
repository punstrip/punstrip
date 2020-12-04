#!/usr/bin/python3
import logging
import re, gc
import numpy as np
import json
import functools, itertools
from io import BytesIO
import binascii
import sys
from multiprocessing import Pool
from annoy import AnnoyIndex
import scipy as sp
from tqdm import tqdm
import collections

import context
from classes.config import Config
from classes.database import Database
import classes.utils
import classes.callgraph



def build_n_d_sparse_tensor(d, GG, name_to_index):
    #scipy sparse does not support D > 2
    assert(d > 2)
    dim = len(name_to_index.keys())
    shape = (dim,) * (d-1)
    cg = {}

    for G in GG:
        for node in G.nodes():
            #skip data xrefs
            if not G.nodes[node]['func']:
                continue

            callees = list(filter(lambda x: G[node][x]['call_ref'], list(G.successors(node))))
            if len(callees) < d-1:
                ##doesn't call as many dimensions as we are looking for
                continue
            
            #order in tensor matters
            for perm in itertools.combinations(callees, d-1):
                #indexes = tuple(map(lambda x: name_to_index[x], (node,) + perm))
                if node not in cg:
                    cg[ node ] = collections.Counter()

                cg[node][ frozenset(perm) ] += 1

    return cg

def build_factors(GG, name_to_index):
    cg = {}
    cgr = {}

    for G in GG:
        for node in G.nodes():
            #skip data xrefs
            if not G.nodes[node]['func']:
                continue

            callees = list(filter(lambda x: G[node][x]['call_ref'], list(G.successors(node))))

            if len(callees) > 0:
                #order in tensor matters
                fs = frozenset(callees)
                if fs not in cgr:
                    cgr[fs] = collections.Counter()

                if node not in cg:
                    cg[ node ] = collections.Counter()

                cg[node][fs ] += 1
                cgr[fs][node] += 1

    return cg, cgr



if __name__ == "__main__":

    config = classes.config.Config()
    logger = config.logger
    logger.setLevel(logging.INFO)
    db = classes.database.Database(config)

    #load new mapping
    name_to_index = classes.utils.load_py_obj(config, 'name_to_index')
    index_to_name = classes.utils.load_py_obj(config, 'index_to_name')

    training_bins = classes.utils.load_py_obj( config, "training_bins" )
    testing_bins = classes.utils.load_py_obj( config, "testing_bins" )

    GG = classes.callgraph.mp_load_cgs(config, training_bins)

    cg, cgr = build_factors(GG, name_to_index)
    classes.utils.save_py_obj(config, cg, "ff_factors")
    classes.utils.save_py_obj(config, cgr, "ff_rfactors")

    import IPython
    IPython.embed()
    sys.exit()

    for i in range(3, 30):
        i = 30
        print("Building {}d feature functions".format(i))
        cg = build_n_d_sparse_tensor(i, GG, name_to_index)
        print("Saving {}d feature functions".format(i))
        classes.utils.save_py_obj(config, cg, "ff_"+str(i)+"d")
        break
        del cg
        gc.collect()
    import IPython
    IPython.embed()
