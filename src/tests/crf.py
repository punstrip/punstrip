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

import IPython

class TestCRF(unittest.TestCase):
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


    def test_save_model(self):
        crf = CRF(self.config, self.knowns, self.unknowns, self.ln_values, self.callgraph, self.ln_rels, self.ll_rels, self.name_to_index)
        crf.save('/tmp/family')

        IPython.embed()

    def test_funcs_iter(self):
        crf = CRF(self.config, self.knowns, self.unknowns, self.ln_values, self.callgraph, self.ln_rels, self.ll_rels, self.name_to_index)
        knowns = set(self.knowns)
        unknowns = set(self.unknowns)

        for i in crf.model_known_funcs_iter():
            print(CRF.node_to_string(crf.model, i))
            self.assertTrue( crf.model.node[i]['name'] in knowns )
            knowns.remove( crf.model.node[i]['name'] )
        self.assertEqual(len(knowns), 0)


        for i in crf.model_unknown_funcs_iter():
            print(CRF.node_to_string(crf.model, i))
            self.assertTrue( crf.model.node[i]['name'] in unknowns )
            unknowns.remove( crf.model.node[i]['name'] )
        self.assertEqual(len(unknowns), 0)


    def test_nodes_iter(self):
        crf = CRF(self.config, self.knowns, self.unknowns, self.ln_values, self.callgraph, self.ln_rels, self.ll_rels, self.name_to_index)
        knowns = set(self.knowns)
        unknowns = set(self.unknowns)
        ln_values = copy.deepcopy(self.ln_values)


        for i in crf.model_knowns_iter():
            print(CRF.node_to_string(crf.model, i))
            if crf.model.nodes[i]['type'] == 'func':
                self.assertTrue( crf.model.node[i]['name'] in knowns )
                knowns.remove( crf.model.node[i]['name'] )
            else:
                self.assertEqual(crf.model.node[i]['type'], 'fingerprint')
                #get node fingerprint connects to
                nodes = list(crf.model.neighbors(i))
                ##fingerprint node should only connect to 1 other node
                self.assertEqual(len(nodes), 1)
                
                self.assertTrue( crf.model.node[nodes[0]]['name'] in ln_values['fingerprint'].keys() )
                del ln_values['fingerprint'][crf.model.node[nodes[0]]['name']] 

        self.assertEqual(len(knowns), 0)
        self.assertEqual(len(ln_values['fingerprint'].items()), 0)


        for i in crf.model_unknowns_iter():
            print(CRF.node_to_string(crf.model, i))
            #print(crf.model.nodes[i])
            self.assertTrue(crf.model.node[i]['type'] == 'func')
            self.assertTrue( crf.model.node[i]['name'] in unknowns )
            unknowns.remove( crf.model.node[i]['name'] )

        self.assertEqual(len(unknowns), 0)

    def test_max_marginal(self):
        crf = CRF(self.config, self.knowns, self.unknowns, self.ln_values, self.callgraph, self.ln_rels, self.ll_rels, self.name_to_index)
        crf.hide_unknowns()
        crf.save('/tmp/family.hidden')
        #TODO compute marginals
        #opt_score, opt_name = crf.compute_max_marginal(1)
        #print(opt_score, opt_name)
        crf.compute_max_marginals()
        crf.save('/tmp/family.inferred')



    def test_factor_graph(self):
        crf = CRF(self.config, self.knowns, self.unknowns, self.ln_values, self.callgraph, self.ln_rels, self.ll_rels, self.name_to_index)
        #crf.hide_unknowns()
        fg = crf.generate_factor_graph()
        CRF.save_model(fg.model, '/tmp/family.fg')


if __name__ == '__main__':
    unittest.main()
