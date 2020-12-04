#!/usr/bin/python3
import sys
import unittest
import logging
import networkx as nx
import numpy as np
import scipy
import copy

import context
from classes.config import Config
from classes.crf import CRF
from classes.crf_sum_product_belief import CRFSumProductBeliefPropagation

import IPython

class TestCRFSumProductBeliefPropagation(unittest.TestCase):
    config = Config()
    #db_ref = Database()
    #db = db_ref.client


    knowns = [ "trish", "bela", "mum" ]
    unknowns = [ "james", "stevie", "emily dad" , "emily"]
    name_to_index = { v:k for k, v in enumerate( knowns + unknowns ) }
    callgraph = nx.DiGraph()
    callgraph.add_edge("james", "emily")
    callgraph.add_edge("mum", "james")
    callgraph.add_edge("trish", "emily")
    callgraph.add_edge("stevie", "bela")
    callgraph.add_edge("emily dad", "emily")

    fp_thetas = scipy.zeros((len(name_to_index.keys()), 1))
    fp_thetas[ name_to_index['james'], 0] = 0.5
    fp_thetas[ name_to_index['emily'], 0] = 10.3

    empty_fp = scipy.zeros((len(name_to_index.keys()), 1))
    james_fp = copy.deepcopy(empty_fp)
    james_fp[ name_to_index['james'], 0 ] = 0.5

    ln_rels = { 'fingerprint' : fp_thetas }
    ln_values = { 'fingerprint' : { 'james' : james_fp, 'emily' : empty_fp }}

    call_rels = scipy.zeros((len(name_to_index.keys()), len(name_to_index.keys())))
    call_rels = CRF.assign_symmetric_rel(call_rels, 'james', 'emily', name_to_index, 1)
    call_rels = CRF.assign_symmetric_rel(call_rels, 'james', 'mum', name_to_index, 1)
    call_rels = CRF.assign_symmetric_rel(call_rels, 'emily', 'trish', name_to_index, 1)
    call_rels = CRF.assign_symmetric_rel(call_rels, 'emily dad', 'emily', name_to_index, 1)
    call_rels = CRF.assign_symmetric_rel(call_rels, 'stevie', 'bela', name_to_index, 1)

    call_thetas = scipy.zeros((len(name_to_index.keys()), len(name_to_index.keys())))
    call_thetas = CRF.assign_symmetric_rel(call_thetas, 'james', 'emily', name_to_index, 3)
    call_thetas = CRF.assign_symmetric_rel(call_thetas, 'james', 'mum', name_to_index, 9)
    call_thetas = CRF.assign_symmetric_rel(call_thetas, 'emily', 'trish', name_to_index, 8)
    call_thetas = CRF.assign_symmetric_rel(call_thetas, 'emily dad', 'emily', name_to_index, 4)
    call_thetas = CRF.assign_symmetric_rel(call_thetas, 'stevie', 'bela', name_to_index, 6)

    ll_rels = { 'control_flow' : { 'f' : call_rels, 'w': call_thetas } }

    def fixed_messages(self):
        crf = CRF(self.config, self.knowns, self.unknowns, self.ln_values, self.callgraph, self.ln_rels, self.ll_rels, self.name_to_index)
        crf_bp = CRFSumProductBeliefPropagation(crf)
        crf_bp.infer()


if __name__ == '__main__':
    unittest.main()
