#!/usr/bin/python3
import sys
import unittest
import logging
import networkx as nx
import numpy as np

import context
from classes.config import Config
from classes.crf import CRF
from classes.crf_greedy_inference import GreedyCRFInference

class TestCRFGreedyInference(unittest.TestCase):
    #db_ref = Database()
    #db = db_ref.client

    def test_top_n_max_marginals(self):
        config = Config()
        knowns = [ "james", "trish", "bela" ]
        unknowns = [ "mum", "stevie", "emily dad" , "emily"]
        callgraph = nx.DiGraph()
        callgraph.add_edge("james", "emily")
        callgraph.add_edge("mum", "james")
        callgraph.add_edge("trish", "emily")
        callgraph.add_edge("stevie", "bela")
        callgraph.add_edge("emily dad", "emily")

        name_to_index = { k:v for v, k in enumerate( knowns + unknowns ) }
        d = len(name_to_index.keys())
        empt = np.zeros((d, 1))
        edfp = np.zeros((d, 1))
        edfp[ name_to_index["emily dad"] , 0] = 0.90
        fingerprints = { "james" : empt, "emily": edfp, "emily dad": edfp }
        relationships = { '1_caller' : {
                            'mum' : { 'james' : 10 },
                            'trish' : { 'emily' : 5 },
                            'stevie' : { 'bela' : 7 },
                            'emily dad' : { 'emily' : 3 }
                          },

                          '1_callee' : {
                            'emily' : { 'emily dad' : 8 },
                            'james' : { 'mum' : 10 },
                            'emily' : { 'trish' : 5 },
                            'bela' : { 'stevie' : 7 }
                            },

                          'fingerprint' : {
                            'emily dad' : 100.2,
                            'emily' : 0.3
                          }
                        }

        crf = CRF(config, knowns, unknowns, callgraph, fingerprints, relationships, name_to_index)
        crf.hide_unknowns()

        clf = GreedyCRFInference(crf)
        best_y = clf.infer(top_n=10)
        fname = "/tmp/family.jointly.inferred"
        clf.logger.info("Best Score: {}, saving to {}".format( best_y[0][0], fname ))
        CRF.save_model(best_y[0][1], fname)



    def test_infer(self):
        pass


if __name__ == '__main__':
    unittest.main()
