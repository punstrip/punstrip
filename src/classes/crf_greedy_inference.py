#!/usr/bin/pyhton3
import copy
import random
import math
import os, sys, functools
import networkx as nx
from networkx.drawing import nx_agraph
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
import logging
from threading import Thread, Lock
import collections

import context
from classes.config import Config
import classes.utils
import classes.NLP
from classes.crf import CRF

import IPython


class GreedyCRFInference(CRF):
    def __init__(self, crf):
        super().__init__(crf.config, crf.Exp, crf.knowns, crf.unknowns, crf.ln_values, crf.callgraph, crf.ln_relationships, crf.ll_relationships, crf.constraints)

    def infer(self, top_n=5):
        unknowns = list(self.model_unknown_funcs_iter())
        nunknowns = len(unknowns)
        self.logger.info("Computing top {} marginals for {} unknowns...".format(top_n, nunknowns))
        self.compute_top_n_marginals(top_n, unknowns)
        self.logger.info("Inferring {} unknown nodes using greedy algorithm".format(nunknowns))

        #iterate through all permutations of top_n complexity
        joint_top_n = top_n
        best_y = self.__max_joint_custom(self.model, unknowns, joint_top_n)
        self.logger.info("Inferred maximum joint probability")
        return best_y


    def compute_top_n_marginals(self, top_n, unknown_node_ids):
        """
            Compute and save the top_n marginal scores
            :param top_n: Number of top scores to save
            :param unknown_node_ids: list of unknown node ids
        """
        #store top n probabilities for each unknown node
        self.top_marginals = { k:[] for k in unknown_node_ids }
        for nodeid in unknown_node_ids:
            best_scores = self.compute_max_marginal(nodeid, top_n)
            self.top_marginals[ nodeid ] = best_scores

    def compute_max_marginal(self, node_id, top_n):
        """
        Compute the max marginal for the given node id
        :param node id: node id to compute max marginal for
        :return: returns the optimial name and the score
        :rtype: tuple
        """
        #create subgraph with only nodes connecting to nodeid
        subg_nodes = set([])
        for u, v in self.model.edges(nbunch=node_id):
            subg_nodes.add(u)
            subg_nodes.add(v)

        G = nx.Graph(self.model.subgraph(list(subg_nodes)))
        #remove edges that do not involve node id
        for u, v in list(G.edges()):
            if u != node_id and v != node_id:
                G.remove_edge(u, v)

        ##remove knowns from subgnode
        node_list = list( set(self.model_knowns_iter(G)) - set([node_id]) )

        #iterate of all combinations of other nodes in set
        self.logger.info("computing_max_marginal :: marginalising `{}` nodes".format(len(G.nodes())))
        self.logger.info("Finding the maximum marginal for node `{}`".format(node_id))
        highest = self.__max_marginal_custom(copy.deepcopy(G), [node_id] + node_list, top_n )
        highest_node_id = list(map(lambda x,node_id=node_id: x[1].nodes[node_id]['name'], highest))
        highest_scores = list(zip(highest_node_id, list(map(lambda x: x[0], highest))))
        print(highest_scores)
        ##create the highest scores of unique names
        seen = set([])
        unique_highest_scores = []
        for name, score in highest_scores:
            if name not in seen and score > 0.0:
                seen.add(name)
                unique_highest_scores.append( (name, score) )

        #print(unique_highest_scores)
        return unique_highest_scores



    def __max_marginal_custom(self, G, node_lst, top_n, node_index=0):
        """
            Maximize the marginal of the first node in node_list by iteration of all assignments to all nodes
            Recursive function that walks along the node lst tracks with node_index
            :param constraints: A list of possible values

            Allow multiple functions being called the same thing by using unique node ids for graph nodes
        """

        #don't allow duplicates in node id list
        if len(set(node_lst)) != len(node_lst):
            return collections.deque( )


        #leaf of binary expansion. evaluate G
        if (node_index + 1) == len(node_lst):
            a = collections.deque()
            a.append( (self.max_sum(G), copy.deepcopy(G)) )
            return a

        #store highest top_n configs
        highest = collections.deque(maxlen=(top_n*2)+1)
        highest.append( (-math.inf, copy.deepcopy(G)) )

        ##for all combinations between node_id 0 and node_index
        possible_names = self.constraints
        if node_index > 0:
            #get relationship between start node and here
            rels = list(map(lambda x: x['rel'], G[ node_lst[0] ][ node_lst[node_index] ].values()))
            #possible names of functions that expose relationship from node0 -> node_index
            rels_constraints = set([])
            master_node = G.nodes[node_lst[0]]['name']
            for rel in rels:
                ##ignore fingerprint relationship for building name constraints
                if rel == 'fingerprint':
                    continue

                self.logger.info("Using Relationship `{}` for `{}`".format(rel, master_node))
                if rel not in self.relationships:
                    self.logger.warn("Relationship `{}` not in known relationships".format(rel))
                    continue
                assert(rel in self.relationships)
                if master_node in self.relationships[rel]:
                    rels_constraints = rels_constraints.union( set(self.relationships[rel][master_node].keys()) )

            possible_names = possible_names.intersection(rels_constraints)
            self.logger.debug("{} possible relationships between `{}` and unknown for relationship `{}`.".format(len(possible_names), master_node, rel))

        #if no possible functions or not known
        if len(possible_names) == 0:
            possible_names = set(['NAME_UNKNOWN'])

        for v in possible_names:
            nodeid = node_lst[node_index]
            G.nodes[nodeid]['name'] = v
            r_highest = self.__max_marginal_custom(G, node_lst, top_n, node_index=node_index+1)
            highest.extend(r_highest)
            tmp = sorted(highest, key=lambda x: x[0])[-top_n:][::-1]
            highest.clear()
            highest.extend(tmp)

        return highest


    def __max_joint_custom(self, G, node_lst, top_n, node_index=0):
        """
            Maximize the marginal of the first node in node_list by iteration of all assignments to all nodes
            Recursive function that walks along the node lst tracks with node_index
            :param constraints: A list of possible values

            Allow multiple functions being called the same thing by using unique node ids for graph nodes
        """

        #don't allow duplicates in node id list
        if len(set(node_lst)) != len(node_lst):
            return collections.deque()


        #leaf of binary expansion. evaluate G
        if (node_index + 1) == len(node_lst):
            a = collections.deque()
            a.append( (self.max_sum(G), copy.deepcopy(G)) )
            return a

        #store highest top_n configs
        highest = collections.deque(maxlen=(top_n*2)+1)
        highest.append( (-math.inf, copy.deepcopy(G)) )

        ##for all combinations between node_id 0 and node_index
        node_id = node_lst[node_index]
        possible_names = set(map(lambda x: x[0], self.top_marginals[node_id]))

        #if no possible functions or not known
        if len(possible_names) == 0:
            possible_names = set(['NAME_UNKNOWN'])

        for v in possible_names:
            nodeid = node_lst[node_index]
            G.nodes[nodeid]['name'] = v
            r_highest = self.__max_joint_custom(G, node_lst, top_n, node_index=node_index+1)
            highest.extend(r_highest)
            tmp = sorted(highest, key=lambda x: x[0])[-top_n:][::-1]
            highest.clear()
            highest.extend(tmp)

        return highest

