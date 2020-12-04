#!/usr/bin/pyhton3
import copy
import random
import math
import os, sys, functools
import networkx as nx
import pygraphviz
from networkx.drawing import nx_agraph
from networkx.drawing.nx_agraph import write_dot
import logging
from threading import Thread, Lock
import numpy as np
import scipy
import IPython

import context
from classes.config import Config
import classes.utils


class FactorGraph:
        fn_shape = { 'factor': 'box', 'variable' : 'circle' }

        def __init__(self, config):
            """
                Daphne Koller defintiion:
                    A factor graph F is an undirected graph containing two types of nodes: variable nodes (denoted as ovals) 
                    and factor nodes (denoted as squares). The graph only contains edges between variable nodes and factor nodes.
                    A factor graph F is parameterized by a set of factors, where each factor V_psi is associated with preciscely
                    one factor psi, whose scope is the set of variables that are neighbors of V_psi in the graph. A distribution
                    P factorizes over F if it can be represented as a set of factors of this form.


                    The factor graph depicts the factorization of the markov model unambiguously.
                    The may be multiple valid factor graph! 

                    e.g.    X -- X
                             \  /
                               X

                    could be X ---@--- X
                              \       /
                               @     @
                                \   /
                                  X

                    or      X   X
                             \ /
                              @
                              |
                              X
            """
            classes.utils._desyl_init_class_(self, config)
            self.logger.debug("Building FactorGraph...")

            """
                A factor graph G = (X, F, E) consists of variable vertices (X), factor vertcies (F) and edges (E)
            """
            self.factor_nodes   = set([])
            self.variable_nodes = set([])
            self.factors        = set([])
            self._factor_counter = 0
            self.model = nx.Graph()

        def to_dot(self, fname='/tmp/factor_graph.dot'):
            write_dot(self.model, fname)

        def _build_fg(self, crf, ignore_fingerprints=True):
            self._factor_counter = max(crf.model.nodes()) + 1
            self.model = nx.Graph()
            self.factors, self.factor_nodes = set([]), set([])

            for node in crf.model_unknowns_iter():
                ntype = 'variable'
                node_name = crf.model.nodes[node]['name']

                self.model.add_node(node, type=ntype, 
                        known=crf.model.nodes[node]['known'], shape=FactorGraph.fn_shape[ntype], 
                        style='filled', fillcolor=crf.model.nodes[node]['fillcolor'], name=crf.model.nodes[node]['name'])
                self.variable_nodes.add(node)

                if not ignore_fingerprints:
                    ##add single fingerprint node for unknown nodes! stored in callgraph
                    assert('fingerprint' in crf.callgraph.nodes[node_name])
                    fp=crf.callgraph.nodes[node_name]['fingerprint']
                    if not isinstance(fp, np.ndarray):
                        fp = np.asarray(crf.callgraph.nodes[node_name]['fingerprint'].todense())[0]

                    fp = fp.reshape(-1)

                    factor_name = '{}_fingerprint'.format(node)
                    self.model.add_node(self._factor_counter, 
                            label=factor_name, 
                            type='factor', 
                            known=True, 
                            relationship='fingerprint',
                            ff=fp, 
                            shape=FactorGraph.fn_shape['factor'], style='filled', 
                            fillcolor='red')
                    self.model.add_edge(node, self._factor_counter)
                    self.factors.add(factor_name)
                    self.factor_nodes.add(self._factor_counter)
                    self._factor_counter += 1

                for d in range(1, 4):
                    for v in self.dth_callee_from_node(crf, d, node):
                        self.add_pairwise_factors(crf, d, node, v, 'callee')

                    for u in self.dth_caller_from_node(crf, d, node):
                        self.add_pairwise_factors(crf, d, u, node, 'caller')

                #known_callees = self.generic_known_callee_factor(crf, node)
                #self.add_generic_factor(crf, node, known_callees, 'callee_factor')

            #check fg for errors
            self.__check_edges()

        def generic_known_callee_factor(self, crf, node):
            """
                Emit frozen_set of all known_callees
            """
            known_callees = set([])
            cg_name = crf.model.nodes[node]['name']
            for u, v in crf.callgraph.out_edges(cg_name):
                assert(u == cg_name)
                if v in crf.knowns:
                    known_callees.add(v)
            return frozenset(known_callees)

        def dth_callee_from_node(self, crf, d, node):
            assert(d > 0)
            cg_name = crf.model.nodes[node]['name']
            ##find nodes dth callee away e.g. a -> b -> c, (d=2, node=a) -> c
            for u, v in crf.callgraph.out_edges(cg_name):
                assert(cg_name == u)
                crf_node = crf.name_to_nodeid[v]
                if d-1 == 0:
                    yield crf_node
                    continue
                yield from self.dth_callee_from_node(crf, d-1, crf_node)

        def dth_caller_from_node(self, crf, d, node):
            assert(d > 0)
            ##find nodes dth callee away e.g. a -> b -> c, (d=2, node=c) -> a
            cg_name = crf.model.nodes[node]['name']
            for u, v in crf.callgraph.in_edges(cg_name):
                assert(v == cg_name)
                crf_node = crf.name_to_nodeid[u]
                if d-1 == 0:
                    yield crf_node
                    continue
                yield from self.dth_caller_from_node(crf, d-1, crf_node)

        def add_pairwise_factors(self, crf, d, u, v, rel):
            ##add a single known_call factor
            if not crf.model.nodes[u]['known'] and crf.model.nodes[v]['known']:
                mrel = 'known_callee_{}'.format(d)
                factor_name = crf.model.nodes[v]['name'] + '_' + mrel
                known_index = crf.Exp.to_index('known_name_vector', crf.model.nodes[v]['name'])
                self.model.add_node(self._factor_counter, 
                        label=factor_name, 
                        type='factor', 
                        known=True, 
                        known_index=known_index,
                        relationship=mrel,
                        ff=crf.Exp.to_vec('known_name_vector', [crf.model.nodes[v]['name']]))
                self.model.add_edge(u, self._factor_counter)
                self.factors.add(factor_name)
                self.factor_nodes.add(self._factor_counter)
                self._factor_counter += 1

            elif crf.model.nodes[u]['known'] and not crf.model.nodes[v]['known']:
                mrel = 'known_caller_{}'.format(d)
                factor_name = crf.model.nodes[u]['name'] + '_' + mrel
                known_index = crf.Exp.to_index('known_name_vector', crf.model.nodes[u]['name'])
                self.model.add_node(self._factor_counter, 
                        name=factor_name, 
                        type='factor', 
                        known=True, 
                        known_index=known_index,
                        relationship=mrel,
                        ff=crf.Exp.to_vec('known_name_vector', [crf.model.nodes[u]['name']]))
                self.model.add_edge(v, self._factor_counter)
                self.factors.add(factor_name)
                self.factor_nodes.add(self._factor_counter)
                self._factor_counter += 1

            elif not crf.model.nodes[u]['known'] and not crf.model.nodes[v]['known']:
                if d > 1:
                    ##do not add nth callers/callees between unknowns
                    return
                ##add label-label factor between 2 variable nodes

                ##only add this edge from multgraph
                #if it hasn't been added before
                direction   = "{}->{}".format(u, v) 
                factor_name = "{}_{}".format(rel, direction)
                #color = CRF.rels_colors[rel]
                color = 'red'

                self.model.add_node(self._factor_counter, type='factor', relationship=rel, direction=direction, label=factor_name, shape=FactorGraph.fn_shape['factor'], fixedsize=True, width=0.2, height=0.2, style='filled', bgcolor=color, color=color, start=u, end=v)
                self.model.add_edge(u, self._factor_counter)
                self.model.add_edge(self._factor_counter, v)
                self.factors.add(factor_name)
                self.factor_nodes.add(self._factor_counter)
                self._factor_counter += 1

        def add_generic_factor(self, crf, u, V, rel):
            factor_name = crf.model.nodes[v]['name'] + '_' + rel
            """
                Create new set of feature functions for this factor if it doesn't exist
            """
            if V not in crf.factor_relationships:
                ff = np.ndarray((crf.Exp.name_vector_dims, ), dtype=np.float64)
                crf.factor_relationships[V] = ff

            self.model.add_node(self._factor_counter, 
                    label=factor_name, 
                    type='factor', 
                    known=True, 
                    relationship=rel,
                    factor_key=V)
            self.model.add_edge(u, self._factor_counter)
            self.factors.add(factor_name)
            self.factor_nodes.add(self._factor_counter)
            self._factor_counter += 1

        @staticmethod
        def fromCRF(crf):
            fg = FactorGraph(crf.config)
            fg._build_fg(crf)
            return fg

        def __check_edges(self):
            for u, v in self.model.edges():
                u_node_type = self.model.nodes[u]['type']
                v_node_type = self.model.nodes[v]['type']

                if u_node_type not in FactorGraph.fn_shape.keys():
                    raise Exception("Error, node {} is of unknown type - `{}`".format(u, u_node_type))
                if v_node_type not in FactorGraph.fn_shape.keys():
                    raise Exception("Error, node {} is of unknown type - `{}`".format(v, v_node_type))

                if u_node_type == v_node_type:
                    raise Exception("Error, node {} of type `{}` connects to node {} of type `{}`. Types need to be different".format(u, u_node_type, v, v_node_type))


