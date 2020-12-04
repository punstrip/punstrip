#!/usr/bin/pyhton3
import copy
import random
import math
import re
import os, sys, functools
import networkx as nx
from networkx.drawing import nx_agraph
#import pydot
#from networkx.drawing.nx_pydot import write_dot
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
import logging
from threading import Thread, Lock
import numpy as np
import scipy

import context
from classes.config import Config
import classes.utils
import classes.NLP
import classes.factor_graph

import IPython


class CRF:
        rels_colours = { 'callee' : 'blue', 'fingerprint' : 'orange', 'data_flow' : 'black', 'shared_memory_access' : 'pink', 'caller' : 'yellow', 'constants' : 'purple', 'known_call' : 'brown' }
        ##known are grey, unknown are white
        node_colours = { True: 'grey', False: 'white' }
        ##c runtime functions
        calculable_knowns = set([ 'init', 'fini', 'csu_init', 'csu_fini', 'start' , 'libc_csu_init', 'libc_csu_fini', 'libc_start', 'deregister_tm_clones', 'register_tm_clones', 'rtld_init', 'main',
            'do_global_dtors_aux', 'frame_dummy', 'frame_dummy_init_array_entry', 'do_global_dtors_aux_fini_array_entry', 'init_array_end', 'init_array_start', 'start_main', 'libc_start_main'])

        @classmethod
        def fromInstance(cls, crf):
            return cls(crf.config, crf.Exp, crf.knowns, crf.unknowns, crf.callgraph, crf.fingerprints, crf.relationships, crf.orig_constraints)

        def __init__(self, config, Exp, known_funcs, unknown_funcs, callgraph, label_node_rels, label_label_rels, factor_rels, constraints=None):
            """
            :param knowns: set of known node in callgraph set([string])
            :param unknowns: set of unknown node in callgraph set([string])
            :param callgraph: networkx DiGraph of callgraph
            :param ln_relatonships: label-node relationships dict('relationship name') -> scipy.sparse.matrix Each rel is a name, row vector representing theta for each name in name_to_index for relationship
            :param ll_relationships: dict('relationship_name') -> dict{ 'f' :  scipy.sparse.matrix, 'w' : scipy.sparse.matrix }. Symmetric matrix connecting all nodes to each other. f is the feature function values, theta is an DxD matrix with weightings, also symmetric.
            :param name_to_index: dict[name] -> name_index
            :param constraints: A scipy.sparse.matrix row vector with elements in {0,1} indexed by name_to_index to identify if a name is allowed
                0 -> disabled, 1-> enabled
            :return: A CRF instance
            :rtype: classes.CRF

            Constraints is a list of functions that the function cannot be.
            self.constraints is the list of function names we can choose from

            Relationships connect two nodes and  maintain x -> y.  x -> y == y -> x e.g. 'connected_by_nth_call'
            ll_relationships and ln_relationships are stored in a matrix. The matrices are index by self.name_to_index. 
            ll_relationships are undirected and thus symmetric. Elements in the matrix represent the theta weighting for each connection 
            ln_relationships are also undirected.
            """

            ##move unknowns with these names into knowns

            self.assumed_knowns = set(filter(lambda x: x in CRF.calculable_knowns, set(unknown_funcs).union(set(known_funcs))))
            self.knowns = set(known_funcs).union(self.assumed_knowns)
            self.unknowns = set(filter(lambda x: x not in CRF.calculable_knowns, unknown_funcs))
            self.callgraph = callgraph
            self.ll_relationships = label_label_rels
            self.ln_relationships = label_node_rels
            self.factor_relationships = factor_rels
            self.Exp = Exp
            #Special unknown function name
            #self.index_to_name = { v : k for k, v in name_to_index.items() }
            #self.dim = len(name_to_index.keys())

            if isinstance(constraints, type(None)):
                constraints = scipy.ones( (self.Exp.name_vector_dims,), scipy.float64 )
                for known in known_funcs:
                    constraints[ Exp.to_index('name_vector', known) ] = 0
            elif not isinstance(constraints, np.ndarray):
                raise Exception("Invalid formation for constraints. Should be a numpy ndarray. You passed `{}`".format(type(constraints)))
                
            self.constraints = constraints.reshape(-1)
            self.model = None

            self.available_ll_relationships = set(self.ll_relationships.keys())
            self.available_ln_relationships = set(self.ln_relationships.keys())
            self.available_factor_relationships = set(self.factor_relationships.keys())
            
            classes.utils._desyl_init_class_(self, config)
            self.logger.debug("Creating new CRF with:")
            self.logger.debug("\t{} label-label relationships".format(self.available_ll_relationships))
            self.logger.debug("\t{} label-node relationships".format(self.available_ln_relationships))
            self.logger.debug("\t{} knowns".format(len(self.knowns)))
            self.logger.debug("\t{} unknowns".format(len(self.unknowns)))
            self.logger.debug("\teach unknown random variable has {} possible assignments.".format(self.Exp.name_vector_dims))


            self.logger.debug("\tType checking initalisation variables...")
            self._type_check_init_vars()
            self.logger.debug("\tChecking ll relationships...")
            #self._check_ll_relationships()

            #transform knowns, unknowns, callgraph, relationships into a undirected graphical model
            self.logger.debug("Building CRF model...")
            self._build_model()
            self.knowns = set(self.model_knowns_iter())

        def _check_ll_relationships(self):
            """
                Checks the relationship variables. 
                Forces all feature functions to be 0 or 1 and symmetric.
                Forces all omega matrices to be symmetric.
            """
            for rel, d in self.ll_relationships.items():
                #if scipy.transpose(d['f']) != d['f']:
                #    raise Exception("Feature function matrix for relationship `{}` is not symmetric".format(rel))
                r, c = d['f'].nonzero()
                for i, j in zip(r, c):
                    if d['f'][i, j] != 0 and d['f'][i, j] != 1:
                        raise Exception("Feature function matrix for relationship `{}` at position [{},{}] is not in {{0,1}}".format(rel, i, j))
                    if d['f'][j, i] != d['f'][i, j]:
                        raise Exception("Feature function matrix for relationship `{}` is not symmetric at position [{},{}]".format(rel, i, j))

                """
                r, c = d['w'].nonzero()
                for i, j in zip(r, c):
                    if d['w'][j, i] != d['w'][i, j]:
                        raise Exception("Theta matrix for relationship `{}` is not symmetric at position [{},{}]".format(rel, i, j))
                """

        def _type_check_init_vars(self):
            ln_relationships_desc = "ln_relationships should be a dictionary of relationship types to a row vector of thetas for each name in name_to_index."
            ll_relationships_desc = "ll_relationships should be a dictionary of dictionaries. The outer dict key specifies the relationship type. The inner dictionary contains the keys `f` and `w` which correspond to the feature function square matrix and theta weighting square matrix for every name in name_to_index * name_to_index" 

            if not isinstance(self.ln_relationships, dict):
                raise Exception(ln_relationships_desc)

            if not isinstance(self.ll_relationships, dict):
                raise Exception(ll_relationships_desc)

        def _build_model(self):
            """
                Builds internal CRF model from callgraph and relationships between knowns and unknowns
                self.model is built with graphviz +styles +colors!
            """
            self.model = nx.Graph()
            #nodes are identified by node ids, not names so we can easily modify names
            node_id_counter = 0
            name_to_nodeid = {}

            ##include disconnected nodes
            ##add nodeid counter for nodes that don't have callers
            for n in self.callgraph.nodes:
                node = self.callgraph.nodes[n]
                name_to_nodeid[n] = node_id_counter
                known_status = node['func'] and not node['text_func']
                if n in self.assumed_knowns:
                    known_status = True

                self.model.add_node(node_id_counter, name=n, known=known_status, fillcolor=CRF.node_colours[known_status],style='filled',type='func')
                node_id_counter += 1

            for u, v in self.callgraph.edges():
                self.model.add_edge(name_to_nodeid[u], name_to_nodeid[v], rel='caller', direction='{}->{}'.format(name_to_nodeid[u], name_to_nodeid[v]), color=CRF.rels_colours['caller'])

            self.name_to_nodeid = name_to_nodeid

        def model_known_funcs_iter(self, model=None):
            """
                Return an iterator to knowns functions in model
                :return: returns the CRF models node for each known function
            """
            if not model:
                model = self.model

            for node in model.nodes():
                if model.nodes[node]['known'] and model.nodes[node]['type'] == 'func':
                    yield node

        def model_unknown_funcs_iter(self, model=None):
            """
                Return an iterator to unknown functions in model
                :return: returns the CRF models node for each unknown function
            """
            if not model:
                model = self.model

            for node in model.nodes():
                if not model.nodes[node]['known'] and model.nodes[node]['type'] == 'func':
                    yield node

        def model_knowns_iter(self, model=None):
            """
                Return an iterator to knowns in model
                :return: returns the CRF models node for each known
            """
            if not model:
                model = self.model

            for node in model.nodes():
                if model.nodes[node]['known']:
                    yield node

        def model_unknowns_iter(self, model=None):
            """
                Return an iterator to unknowns in model
                :return: returns the CRF models node for each unknown
            """
            if not model:
                model = self.model

            for node in model.nodes():
                if not model.nodes[node]['known']:
                    yield node

        @staticmethod
        def node_to_string(model, nodeid):
            nodestr = "( {:^5} : {:^14} : {:^7} : {:^13} )".format(nodeid, model.nodes[nodeid]['name'], "known" if model.nodes[nodeid]['known'] else "unknown", model.nodes[nodeid]['type'])
            return nodestr

        def hide_unknowns(self):
            for node in self.model_unknowns_iter():
                name = self.model.nodes[node]['name']
                self.model.nodes[node]['label'] = 'NAME_UNKNOWN'
                self.callgraph.nodes[name]['label'] = 'NAME_UNKNOWN'



        @staticmethod
        def save_model(model, fname):
            """
            Saves model to a file using the dot format
            :param fname: string filename
            """
            #write_dot(self.callgraph, fname + ".dot")
            write_dot(model, fname + ".dot")
            #nx.write_gexf(self.model, fname + ".gexf")

        def remove_irrelevant(self):
            """
                Remove nodes that do not interact with unknown nodes!
                Remove edges between knowns!
            """
            for node in list(self.model_knowns_iter()):
                REMOVE_THIS_NODE = True
                for u, v in set(self.model.edges(node)):
                    if not self.model.nodes[u]['known'] or not self.model.nodes[v]['known']:
                        REMOVE_THIS_NODE = False
                        break
                if REMOVE_THIS_NODE:
                    self.callgraph.remove_node( self.model.nodes[node]['label'] )
                    self.model.remove_node(node)

            ##now remove edges between knowns
            for node in list(self.model_knowns_iter()):
                for u, v in list(self.model.edges(node)):

                    if self.model.nodes[u]['known'] and self.model.nodes[v]['known']:
                        uname = self.model.nodes[u]['label']
                        vname = self.model.nodes[v]['label']

                        ## we are modifying the current graph
                        ## check if edge still exists
                        if vname in self.callgraph[uname]:
                            self.callgraph.remove_edge(uname, vname)
                        if v in self.model[u]:
                            self.model.remove_edge(u, v)

        def save(self, fname):
            """
            Saves model to a file using the dot format
            :param fname: string filename
            """
            write_dot(self.callgraph, fname + ".cg.dot")
            write_dot(self.model, fname + ".crf.dot")
            #nx.write_gexf(self.model, fname + ".crf.gexf")

        def save_to_db(self):
            """
                This overwrites the (trained) model for the current experimental
                settings
            """
            ##save weights, convert to csc_matrix first
            #learned = self.ll_relationships, self.ln_relationships

            learned = self.ll_relationships, self.ln_relationships
            CRF.save_relationships(self.Exp, learned)

            ##callgraph and model
            ##this might be huge!
            """
            graphs = self.callgraph, self.model
            self.Exp.update_experiment_key('graphs', graphs)

            if hasattr(self, 'factor_graph'):
                self.Exp.update_experiment_key('factor_graph', self.factor_graph )
            """

        @staticmethod
        def save_relationships(Exp, learned, key='crf_relationships'):

            ##compress matrix to sparse
            ll_rels, ln_rels = learned
            ##cannot copy, too much ram required
            #ll_rels = copy.deepcopy(orig_ll_rels)
            for rel in ll_rels:
                if type(ll_rels[rel]) != scipy.sparse.csc_matrix:
                    ll_rels[rel] = scipy.sparse.csc_matrix( ll_rels[rel] )

            learned = ll_rels, ln_rels
            Exp.update_experiment_key(key, learned)


        @staticmethod
        def load_relationships(Exp):
            ll_rels, ln_relationships = Exp.load_experiment_key('crf_relationships')

            for rel in ll_rels:
                if type(ll_rels[rel]['w']) == scipy.sparse.csc_matrix:
                    ll_rels[rel]['w'] = ll_rels[rel]['w'].todense()

            return ll_rels, ln_relationships

        def load_from_db(self):
            """
                This overwrites the (trained) model for the current experimental
                settings
            """
            ##load weights
            #self.ll_relationships, self.ln_relationships = self.Exp.load_experiment_key('crf_relationships')
            learned = CRF.load_relationships(self.Exp)
            self.ll_relationships, self.ln_relationships = learned

            ##callgraph and model
            ##this might be huge!
            #self.callgraph, self.model = self.Exp.load_experiment_key('graphs')

            #self.factor_graph = self.Exp.load_experiment_key('factor_graph')
            #self.logger.info("Loading CRF from database!")

        def clone(self):
            return copy.deepcopy(self)

        def __eq__(self, other):
            return self.data == other.data

        def __str__(self):
            return self.to_json()

        def max_sum(self, G):
            """
                Custom optimization sub-routine to calculate the max sum of feature functions defined in relationships
                Permutate all permutations of nodes in G and maximize the feature function values
                :param G: CRF subgraph model
                :return: score for current CRF subgraph G


                This function is optimized not to do error checking, assume keys in dict exist
            """
            score = 0.0
            for u, v in G.edges():
                for e in G[u][v]:
                    rel = G[u][v][e]['rel']
                    #only looking for single param relationship
                    if rel in self.available_ln_relationships:
                        ln_node = u if G.nodes[u]['type'] == rel else v
                        unknown_node = v if G.nodes[u]['type'] == rel else u

                        uname = G.nodes[unknown_node]['name']
                        uind = self.name_to_index[ uname ]
                        theta_fp = self.ln_relationships[rel][ uind, 0 ]
                        ##multiply weighting by probability from PMF
                        score += (theta_fp * self.nodeid_to_ln_value[rel][ln_node][ uind, 0 ] )
                        continue

                    uname, vname = G.nodes[u]['name'], G.nodes[v]['name']
                    uind, vind = self.Exp.to_index('name_vector', uname), self.Exp.to_index('name_vector', vname) 
                    f_uv        = self.ll_relationships[rel]['f'][uind, vind]
                    theta_uv    = self.ll_relationships[rel]['w'][uind, vind]
                    ##NB: f_uv should always be 1 or 0, theta_uv should be t or 0. Could skip multiplication, just use theta_uv
                    score += (f_uv * theta_uv)
            return score


        def __max_marginal_custom(self, G, node_lst, node_index=0):
            """
                Maximize the marginal of the first node in node_list by iterating over all assignments to all nodes
                Recursive function that walks along the node lst tracks with node_index
                :param constraints: A list of possible values

                Allow multiple functions being called the same thing by using unique node ids for graph nodes
            """

            #don't allow duplicates in node id list
            if len(set(node_lst)) != len(node_lst):
                return -math.inf

            #leaf of binary expansion. evaluate G
            if (node_index + 1) >= len(node_lst):
                return self.max_sum(G), G

            #don't iterate over known functions
            if G.nodes[ node_lst[node_index] ]['known']:
                return self.__max_marginal_custom(G, node_lst, node_index=node_index+1)


            max_score = -math.inf
            #max_graph = copy.deepcopy(G)
            max_graph = G

            """
            ##for all combinations between node_id 0 and node_index
            constraints = copy.deepcopy(self.constraints)
            if node_index > 0:
                #get relationship between start node and here
                rels = list(map(lambda x: x['rel'], G[ node_lst[0] ][ node_lst[node_index] ].values()))
                #possible names of functions that expose relationship from node0 -> node_index
                rels_constraints = set([])
                master_node = G.nodes[node_lst[0]]['name']

                for rel in rels:
                    ##ignore fingerprint relationship for building name constraints
                    if rel in self.available_ln_relationships:
                        continue

                    self.logger.info("Using Relationship `{}` for `{}`".format(rel, master_node))
                    if rel not in self.available_ll_relationships:
                        self.logger.warn("Relationship `{}` not in known relationships".format(rel))
                        continue
                    assert(rel in self.ll_relationships)
                    ##constraints too harsh, if function doe snot exhibit seen relationship, we cannot infer it
                    #constraints *= self.ll_relationships[rel]['f'][:, self.Exp.to_index('name_vector', master_node)].reshape(self.Exp.name_vector_dims, 1)
                        

                
                #self.logger.debug("{} possible relationships between `{}` and unknown for relationship `{}`.".format(len(possible_names), master_node, rel))
            """

            r, c = self.constraints.nonzero()
            possible_names = set(map(lambda x: self.Exp.name_vector[x], r))

            #if no possible functions or not known
            if len(possible_names) == 0:
                raise Exception("Constraints dictate this node can not be any function name I know...")

            for v in possible_names:
                nodeid = node_lst[node_index]
                G.nodes[nodeid]['name'] = v
                score, graph = self.__max_marginal_custom(G, node_lst, node_index=node_index+1)
                if score > max_score:
                    max_score = score
                    max_graph = copy.deepcopy(graph)

            return max_score, max_graph



        def _max_marginal_scipy(self, G, node_id):
            """
                Convert out problem into a scalr function of multiple variables. Pass to scipy optimize
                :return: list of tuples (node_id, opt_name, score)
            """
            node_ids = list(self.model_unknowns_model_iter(model=G))
            num_unknowns = len(node_ids)
            #initial guess of function names, incremental order
            X = np.arrange(num_unknowns)
            #set bounds between function names
            bounds = [ (0, self.Exp.name_vectr_dims) ] * num_unknowns

            X_opt, scores = scipy.optimize.maximize(CRF.sum_product, X, args=(G, self.relationships, node_ids, Exp,))
            return zip(node_ids, list(map(lambda x: self.Exp.name_vector[x], X_opt)), scores)

        def compute_max_marginal(self, node_id):
            """
            Compute the max marginal for the given node id
            :param node id: node id to compute max marginal for
            :return: returns the optimial name and the score
            :rtype: tuple
            """
            #create subgraph with only nodes connecting to nodeid
            subg_nodes = set([])
            ##node_id might be disconnected
            subg_nodes.add(node_id)
            for u, v in self.model.edges(nbunch=node_id):
                subg_nodes.add(u)
                subg_nodes.add(v)


            G = nx.MultiGraph(self.model.subgraph(list(subg_nodes)))
            #remove edges that do not involve node id
            for u, v in list(G.edges()):
                if u != node_id and v != node_id:
                    G.remove_edge(u, v)

            ##remove node_id from subgraph
            #node_list = list( set(self.model_knowns_iter(G)) - set([node_id]) )
            node_list = list( set(G.nodes()) - set([node_id]) )

            #iterate of all combinations of other nodes in set
            self.logger.info("computing_max_marginal :: marginalising `{}` nodes".format(len(G.nodes())))
            self.logger.info("Finding the maximum marginal for node `{}`".format(node_id))
            self.logger.info(node_list)
            #opt_score, opt_graph = self.__max_marginal_custom(copy.deepcopy(G), [node_id] + node_list )
            opt_score, opt_graph = self.__max_marginal_custom(copy.deepcopy(G), node_list )
            opt_name = opt_graph.nodes[node_id]['name']
            return opt_score, opt_name


        def compute_max_marginals(self):
            """
                Compute the max-marginals for all unknowns in the model
            """
            for node in self.model_unknown_funcs_iter():
                opt_score, opt_name = self.compute_max_marginal(node)
                self.logger.info("{} : {} : {}".format(node, opt_name, opt_score))
                if opt_score > 0:
                    #self.model.nodes[node]['name'] = opt_name
                    self.model.nodes[node]['label'] = opt_name

        def generate_factor_graph(self):
            """
                Generate a factor graph of the CRF
            """
            fg = classes.factor_graph.FactorGraph.fromCRF(self)
            return fg

        @staticmethod
        def assign_symmetric_rel(Exp, mat, name_a, name_b, value):
            """
                Shortcut method to assign relationship in symmertic matrix
                :param mat: Matrix
                :param name_a: First name
                :param name_b: Other name
                :param name_to_index: index mapping names tro indecies in matrix
                :param value: value of relationship
                :return matrix: Modified matrix with symmetric relationship
                :rtype: matrix
            """
            ia = Exp.to_index('name_vector', name_a)
            ib = Exp.to_index('name_vector', name_b)
            mat[ ia, ib ] = value
            mat[ ib, ia ] = value
            return mat

        @staticmethod
        def callgraph_to_relationships(Exp, cg):
            """
                Generate a ll relationship matrices from a callgraph
            """

            callee_rels   = scipy.sparse.lil_matrix(( Exp.name_vector_dims, Exp.name_vector_dims), dtype=np.float64)
            callee_thetas = scipy.sparse.lil_matrix(( Exp.name_vector_dims, Exp.name_vector_dims), dtype=np.float64)

            caller_rels   = scipy.sparse.lil_matrix(( Exp.name_vector_dims, Exp.name_vector_dims), dtype=np.float64)
            caller_thetas = scipy.sparse.lil_matrix(( Exp.name_vector_dims, Exp.name_vector_dims), dtype=np.float64)

            for u, v in cg.edges():
                #callee
                CRF.assign_symmetric_rel( Exp, callee_rels, u, v, 1 )
                prev_value = callee_thetas[ Exp.to_index('name_vector', u), Exp.to_index('name_vector', v) ]
                #CRF.assign_symmetric_rel( Exp, callee_thetas, u, v, prev_value + 1 )
                CRF.assign_symmetric_rel( Exp, callee_thetas, u, v, 0 )
                
                #caller
                CRF.assign_symmetric_rel( Exp, caller_rels, v, u, 1 )
                prev_value = caller_thetas[ Exp.to_index('name_vector', v), Exp.to_index('name_vector', u) ]
                #CRF.assign_symmetric_rel( Exp, caller_thetas, v, u, prev_value + 1 )
                CRF.assign_symmetric_rel( Exp, caller_thetas, v, u, 0 )

            ##convert to csr matrices
            callee_rels   = scipy.sparse.csr_matrix(callee_rels)
            callee_thetas = scipy.sparse.csr_matrix(callee_thetas)

            caller_rels   = scipy.sparse.csr_matrix(caller_rels)
            caller_thetas = scipy.sparse.csr_matrix(caller_thetas)

            return { 
                'caller' : { 'f' : caller_rels, 'w': caller_thetas }, 
                'callee' : { 'f' : callee_rels, 'w': callee_thetas }
            }

        @staticmethod
        def add_relationships_from_callgraph(Exp, cg, ll_rels):
            """
                Generate a ll relationship matrices from a callgraph
            """
            assert( np.shape(ll_rels['caller']['f']) == (Exp.name_vector_dims, Exp.name_vector_dims) )

            """
            callee_rels   = scipy.sparse.lil_matrix( ll_rels['callee']['f'])
            callee_thetas = scipy.sparse.lil_matrix( ll_rels['callee']['w'])

            caller_rels   = scipy.sparse.lil_matrix( ll_rels['caller']['f'])
            caller_thetas = scipy.sparse.lil_matrix( ll_rels['caller']['w'])
            """

            callee_rels   = scipy.sparse.lil_matrix(ll_rels['callee']['f'])
            callee_thetas = ll_rels['callee']['w']

            caller_rels   = scipy.sparse.lil_matrix(ll_rels['caller']['f'])
            caller_thetas = ll_rels['caller']['w']

            ##convert to dense

            if type(callee_thetas) == scipy.sparse.csr.csr_matrix:
                callee_thetas = callee_thetas.todense()
            if type(caller_thetas) == scipy.sparse.csr.csr_matrix:
                caller_thetas = caller_thetas.todense()

            for u, v in cg.edges():
                #callee
                CRF.assign_symmetric_rel( Exp, callee_rels, u, v, 1 )
                #prev_value = callee_thetas[ Exp.to_index('name_vector', u), Exp.to_index('name_vector', v) ]
                #CRF.assign_symmetric_rel( Exp, callee_thetas, u, v, prev_value + 1 )
                #CRF.assign_symmetric_rel( Exp, callee_thetas, u, v, 0 )
                
                #caller
                CRF.assign_symmetric_rel( Exp, caller_rels, v, u, 1 )
                #prev_value = caller_thetas[ Exp.to_index('name_vector', v), Exp.to_index('name_vector', u) ]
                #CRF.assign_symmetric_rel( Exp, caller_thetas, v, u, prev_value + 1 )
                #CRF.assign_symmetric_rel( Exp, caller_thetas, v, u, 0 )

            ##convert to csr matrices
            callee_rels   = scipy.sparse.csr_matrix(callee_rels)
            #callee_thetas = scipy.sparse.csr_matrix(callee_thetas)

            caller_rels   = scipy.sparse.csr_matrix(caller_rels)
            #caller_thetas = scipy.sparse.csr_matrix(caller_thetas)

            return { 
                'caller' : { 'f' : caller_rels, 'w': caller_thetas }, 
                'callee' : { 'f' : callee_rels, 'w': callee_thetas }
            }

        def check_accuracy(self, confidence=0.0):
            tp, fp = 0, 0
            tn, fn = 0, 0
            for node in list(self.model_unknowns_iter()):
                corr_name   = self.model.nodes[node]['name']
                inf_name    = self.model.nodes[node]['label']

                connected_knowns = 0
                connected_unknowns = 0
                for v in self.model[node]:
                    if 'marginal' not in self.model.nodes[v]:
                        print("node has no marginal")
                        IPython.embed()
                    if np.max(self.model.nodes[v]['marginal']) < confidence:
                        fn += 1
                        self.logger.debug("{:<40}--{:<3} NOT CONFIDENT {:<3}->{:>40}".format(corr_name, connected_knowns, connected_unknowns, inf_name))
                        continue

                    if self.model.nodes[v]['name'] in self.knowns:
                        connected_knowns+=1
                    else:
                        connected_unknowns+=1

                #self.logger.debug("{:<40}->{:>40}".format(corr_name, inf_name))
                self.logger.debug("{:<40}--{:<3} {:<3}->{:>40}".format(corr_name, connected_knowns, connected_unknowns, inf_name))
                if corr_name == inf_name:
                    tp += 1
                else:
                    fp += 1

            for name in list(self.assumed_knowns):
                self.logger.debug("{:<40}--ASS ASS->{:>40}".format(name, name))
                tp += 1

            accuracy = 0.0
            if (tp+tn+fp+fn) > 0:
                accuracy = (tp+tn) / (tp+tn+fp+fn)
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = (2 * p * r) / (p + r)
            self.logger.debug("TP : {}, FP: {}, FN: {}, FP: {}, ACC: {}".format(tp, fp, fn, fp, accuracy))
            self.logger.debug("P: {}, R: {}, F1: {}".format(p, r, f1))



        def check_accuracy_extended(self, nlp, unkns):
            """
                check accuracy of a subset of unknowns (GLOBAL SYMBOLS, SIZE > 0)
                Using nlp matching

                :param nlp:     classes.nlp.NLP
                :param unkns:   Subset of unknowns we actually care about inferring
                In this case, unkns is a iteratble set of names of unknowns, not the nodes
            """
            tp, fp = 0, 0
            tn, fn = 0, 0
            #for node in unkns:
            for node in self.model.nodes:
                if self.model.nodes[node]['name'] not in unkns:
                    continue

                corr_name   = self.model.nodes[node]['name']
                inf_name    = self.model.nodes[node]['label']

                connected_knowns = 0
                connected_unknowns = 0
                for v in self.model[node]:
                    if self.model.nodes[v]['name'] in self.knowns:
                        connected_knowns+=1
                    else:
                        connected_unknowns+=1

                #self.logger.debug("{:<40}->{:>40}".format(corr_name, inf_name))
                self.logger.debug("{:<40}--{:<3} {:<3}->{:>40}".format(corr_name, connected_knowns, connected_unknowns, inf_name))
                if nlp.check_word_similarity(corr_name, inf_name):
                    tp += 1
                else:
                    fp += 1

            accuracy = 0.0
            if (tp+tn+fp+fn) > 0:
                accuracy = (tp+tn) / (tp+tn+fp+fn)
            self.logger.debug("TP : {}, FP: {}, ACC: {}".format(tp, fp, accuracy))

        def check_accuracy_top_n_nlp(clf, n, nlp, E, unkns):
            tp, tn, fp, fn = 0, 0, 0, 0

            for node in self.model.nodes:
                if self.model.nodes[node]['name'] not in unkns:
                    continue

                corr_name   = self.model.nodes[node]['name']
                m = self.compute_bp_marginal(node)
                top_n = m.argsort()[-n:][::-1]

                CORRECT = False
                for ind in top_n:
                   if nlp.check_word_similarity(corr_name, E.name_vector[ind]):
                       CORRECT = True
                       break

                connected_knowns = 0
                connected_unknowns = 0
                for v in self.model[node]:
                    if self.model.nodes[v]['name'] in self.knowns:
                        connected_knowns+=1
                    else:
                        connected_unknowns+=1

                if CORRECT:
                    tp += 1
                else:
                    fp += 1
                    inf_name   = self.model.nodes[node]['label']
                    self.logger.debug("{:<40}--{:<3} {:<3}->{:>40}".format(corr_name, connected_knowns, connected_unknowns, "inf_name"))

            accuracy = 0.0
            if (tp+tn+fp+fn) > 0:
                accuracy = (tp+tn) / (tp+tn+fp+fn)

            return accuracy, tp, fp

        @staticmethod
        def _scipy_minimize(x, x_mapping, crf):
            """
                Function to be used with scipy optimize
                Need to set input vars through an indircet mapping to feature function 
                weights

                ##nb: rel_type = { ll, ln }

                x_mapping = [
                    ( rel_type, rel, index )
                ]
            """
            ##set x parameters according to mapping
            for i, e in enumerate(x):
                rel_type, rel, ind = x_mapping[i]
                crf[rel_type][rel]['w'][ind[0], ind[1]] = e
            return CRFSumProductBeliefPropagation._score_network(crf)

        @staticmethod
        def _score_relationship(w, c, dim):
            """
                c is the count of how many times x->y
            """

            unknown = scipy.ones( (1, dim) )
            ##elemenet wise multiplication
            e = w * c
            score = unknown @ e

        @staticmethod
        def _score_network(crf):
            """
                Evaluate a score for the current model parameters
                Score is proportional to correct_name == label_name

                NB: CRF is serialized into a dict for multiprocessing
            """
            score = 0.0
            direction_regex = re.compile(r'(\d+)->(\d+)')
            ##TODO score label-node fingerprint relationships
            for u in crf['model'].nodes:
                if not crf['model'].nodes[u]['known']:
                    continue

                for v in crf['model'][u]:
                    for edge in crf['model'][u][v]:
                        rel         = crf['model'][u][v][edge]['rel']
                        if rel not in crf['available_ll_relationships']:
                            raise RuntimeError("Error, unknown relationship: {}".format(rel))

                        direction   = crf['model'][u][v][edge]['direction']
                        m = direction_regex.match(direction)
                        if not m:
                            raise RuntimeError("Unknown direction format: {}".format(direction))

                        start       = int(m.group(1))
                        end         = int(m.group(2))

                        start_name  = crf['model'].nodes[start]['name']
                        end_name    = crf['model'].nodes[end]['name']

                        start_index = crf['Exp_index_cache'][start_name]
                        end_index   = crf['Exp_index_cache'][end_name]

                        ff_score = crf['ll_relationships'][ rel ]['w'][start_index, end_index]
                        score += ff_score

            penalty = 0.0
            for rel in crf['ll_relationships'].keys():
                penalty += np.sum( np.square( crf['ll_relationships'][rel]['w'] ), axis=(0,1) )

            ##we want to maximize, (score - penalty), so minimize inverse
            #if score-penalty == 0.0:
            #    return math.inf
            #return 1.0/(score - penalty)
            return -(score - penalty)
                        
        def train(self):
            """
            train crf using loop belief propagation to infer results
            Need to maximize log likelihood
            """
            if len(list(self.model_unknowns_iter())) > 0:
                raise RuntimeError("Error, you called train on a CRF with unknown nodes in it")

            MAX_EPOCHS = 10

            self.logger.debug("Building weights mapping for all relationships")
            ##randomly sample all possible weights
            ALL_WEIGHTS = []
            for rel, rel_struct in self.ll_relationships.items():
                r, c = rel_struct['f'].nonzero()
                for r_i, c_i in zip(r, c):
                    weight_mapping = 'll_relationships', rel, [r_i, c_i]
                    ALL_WEIGHTS.append( weight_mapping )


            for epoch in range(MAX_EPOCHS):
                self.logger.debug("Starting epoch {}/{}".format(1+epoch, MAX_EPOCHS))
                rnd_sample = random.sample(ALL_WEIGHTS, len(ALL_WEIGHTS))
                count = 0
                with multiprocess.Pool(self.config.analysis.THREAD_POOL_THREADS) as p:
                    #chunks = list(classes.utils.chunks_of_size(rnd_sample, CONCURRENT_WEIGHTS))
                    chunks = list(classes.utils.n_chunks(rnd_sample, self.config.analysis.THREAD_POOL_THREADS))

                    ##cannot serialize rlock class in python for use in multiprocessing
                    non_rlock_crf_obj = copy.copy(self.__dict__)
                    for rlock_containing_obj in ['logger', 'config', 'Exp', 'factor_graph', 'messages', 'callgraph', 'gui_thread', 'model' ]:
                        del non_rlock_crf_obj[rlock_containing_obj]

                    mp_safe_crfs = []
                    for i in tqdm.tqdm(range(len(chunks))):
                        x = copy.deepcopy(non_rlock_crf_obj)
                        x.update({'model': self.model})
                        x.update({'Exp_index_cache': self.Exp.to_index_cache('name_vector')})
                        mp_safe_crfs.append(x)

                    #r = list(tqdm.tqdm(p.starmap(CRFSumProductBeliefPropagation.mp_train_subset, zip(mp_safe_crfs, chunks),chunksize=1), total=len(chunks)))
                    r = list(tqdm.tqdm(p.imap_unordered(CRFSumProductBeliefPropagation.mp_train_subset, zip(mp_safe_crfs, chunks),chunksize=1), total=len(chunks)))
                    for res, x_mapping in r:
                        #if res.success:
                        CRFSumProductBeliefPropagation._set_mapping_results(self, x_mapping, res.x)

            self.logger.debug("Finish training...")
            IPython.embed()

        @staticmethod
        #def mp_train_subset(crf, x_mapping):
        def mp_train_subset( args ):
            crf, x_mapping = args
            bounds = [(0, 100)] * len(x_mapping)
            options = { 'maxiter' : 10 }
            #options = { 'maxfev' : 10**5 }
            method = ''

            current_weights = list(map(lambda x,crf=crf: crf[x[0]][x[1]]['w'][x[2][0], x[2][1]] , x_mapping))
            res = scipy.optimize.minimize( CRFSumProductBeliefPropagation._scipy_minimize, x0=np.array(current_weights), bounds=bounds, args=(x_mapping, crf), options=options)
            if not res.success:
                print("Error optimising subproblem")
                print(res)
                #IPython.embed()
                #raise RuntimeError("Error optimising subproblem")
            return res, x_mapping

        @staticmethod
        def _set_mapping_results(crf, x_mapping, res):
            for i in range(len(res)):
                rel_type, rel, ind = x_mapping[i]
                getattr(crf, rel_type)[rel]['w'][ind[0], ind[1]] = res[i]


