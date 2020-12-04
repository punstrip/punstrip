#!/usr/bin/pyhton3
import copy
import random
import math
import re
from io import StringIO, BytesIO
import os, sys, functools
import networkx as nx
from multiprocessing import shared_memory
from networkx.drawing import nx_agraph
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
import logging
from threading import Thread, Lock
import collections
import numpy as np
import scipy 
import tqdm
import multiprocess
import itertools

from PIL import Image, ImageTk 
import tkinter as tk 
import numpy as np

import PIL
import PIL.Image
import PIL.ImageTk
from threading import Thread
import queue
from queue import Queue
import time
from sklearn.preprocessing import normalize

import context
from classes.config import Config
import classes.utils
import classes.NLP
from classes.crf import CRF
import classes.pmfs
from classes.belief_propagation_message import BPMessage
from classes.belief_propagation_message_store import MessageStore
from classes.gui_thread import GUIThread

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed

import IPython


class CRFSumProductBeliefPropagation(CRF):
    VISUALISATION = False
    valid_msg_colors = { True : 'green', False: 'red' }
    def __init__(self, crf):
        super().__init__(crf.config, crf.Exp, crf.knowns, crf.unknowns, crf.callgraph, crf.ln_relationships, crf.ll_relationships, crf.constraints)
        self._dim = self.Exp.name_vector_dims
        self.gui_thread = None

        ##nodes are recreated here from callgraph

        ## store mesages in cache
        self.messages = MessageStore()

        #store factor graph, use fg for computing messages
        self.factor_graph = self.generate_factor_graph()

    def normalise_weight_matrices(self):
        """
        Applies row normalisation to all weight matrices
        """
        for rel in self.ln_relations:
            normalize( self.ln_relations[rel], norm='l2', copy=False)
        for rel in self.ll_relations:
            normalize( self.ll_relations[rel]['w'], norm='l2', copy=False)

    def apply_tikinov_regularization(self):
        """
            Performs tikinov regularization on all weight matrices


            I don't want to subtract mean and we are using sparse matrices, effecient computation with majority of non-zero elements
            subtracting mean wuld roduce a row of all nonzeros
        """
        MAX_VAL = 32

        for rel_type in ['ln_relationships', 'll_relationships']:
            for rel, w in getattr(self, rel_type).items():
                store = getattr(self, rel_type)
                if isinstance(w, np.ndarray):
                    #self.ln_relationships[rel] = classes.pmfs.PMF.normalise_numpy_density(np.power(w, 2) / (2 * (np.mean(w)**2)))
                    store[rel] = classes.pmfs.PMF.normalise_n(np.power(w, 2), MAX_VAL)
                else:
                    store[rel] = classes.pmfs.PMF.normalise_n(w.power(2), MAX_VAL)

    def _train_back_propagation(self, node, alpha):
        ##for each clique in graph, SGD weightings
        #belief = self.compute_bp_marginal(node)
        ##math.e is the range of the "true" marginal. using [0,1] is almost linear for e^x
        error = self.compute_bp_marginal_error(node)
        if 'name' not in self.model.nodes[node]:
            raise RuntimeError('node {} is not in model nodes!'.format(node))

        ##for each contibuting factor, apply back prop
        for bp_msg in self.messages.to_node_iter(node):
            start, end  = bp_msg.direction
            belief      = bp_msg.value
            true_marginal = self.Exp.to_vec('name_vector', [ self.model.nodes[node]['name'] ])
            diff        = np.log(belief) - true_marginal

            rel = self.factor_graph.model.nodes[start]['relationship']
            ##update weightings for this relationship
            if rel in self.available_ln_relationships:
                ####get predicted value and reduce fingerprint theta if incorrect
                ###increase theta towards 1 if correct
                ln_theta_n = np.argmax(belief)
                ln_pred_diff = diff[ln_theta_n]
                self.ln_relationships[rel][ln_theta_n] -= alpha * ln_pred_diff
            elif rel in self.available_ll_relationships:
                var_node_start  = self.factor_graph.nodes[start]['start']
                var_node_end    = self.factor_graph.nodes[start]['end']
                if 'name' not in self.model.nodes[var_node_end]:
                    raise RuntimeError('name not in factor_graph model nodes end')

                start_name  = self.factor_graph.model.nodes[var_node_start]['name']
                end_name    = self.factor_graph.model.nodes[var_node_end]['name']

                start_node_index = self.Exp.to_index('name_vector', start_name)
                end_node_index = self.Exp.to_index('name_vector', end_name)
                """
                ll relationships have a direction
                start -> end

                0R matrix contains thetas for start -rel-> end
                The generic relationship is -rel-> end, ignores start
                """
                self.ll_relationships[rel]['w'][start_node_index, :] -= alpha * diff
                #return start_node_index, end_node_index,  alpha * diff, error
            else:
                raise RuntimeError("Could not determine relationship type for rel: {}".format(rel))
        return error

 
    def train_mp(self, alpha=1e-3):
        """
        train crf using loop belief propagation to infer results
        Then perform SGD against approximate marginals
        """
        MAX_EPOCHS = 10

        last_epoch_error = math.inf
        for epoch in range(MAX_EPOCHS):
            self.logger.info("Starting epoch {}/{}".format(1+epoch, MAX_EPOCHS))
            self.logger.info("Running loopy belief propagation inference")
            self.infer()

            self.logger.info("Performing Stochastic Gradient Descent...")

            epoch_error = 0.0
            ##for each clique in graph, SGD weightings
            #for node in tqdm.tqdm(self.model_unknowns_iter(), desc='Back propagating weights'):


            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(self._train_back_propagation, node, alpha) for node in self.model_unknowns_iter()]
                for future in as_completed(futures):
                    if future.exception():
                        self.config.error("Exception occurred calculating back propagation!")
                        IPython.embed()
                        raise RuntimeError("Exception occurred calculating back prop")
                    #start_node_index, end_node_index, diff, error = future.result()
                    error = future.result()
                    epoch_error += error

            ##renormalise_relationships - similar to regularization
            #self.logger.info("Normalising weight matrices")
            #self.normalise_weight_matrices()
            #self.apply_tikinov_regularization()
            self.logger.info("Absolute error this epoch: {}".format(epoch_error))
            if epoch_error > last_epoch_error and epoch > 1:
                self.logger.info("EPOCH error increased, stopping")
                break
            last_epoch_error = epoch_error
            #self.save_to_db()
            #IPython.embed()

        print("Finish training...")
        #IPython.embed()


    def train(self, alpha=1e-3):
        """
        train crf using loop belief propagation to infer results
        Then perform SGD against approximate marginals
        """
        MAX_EPOCHS = 10
        IDEAL_WEIGHT_EXP = 4.0

        last_epoch_error = math.inf
        for epoch in range(MAX_EPOCHS):
            self.logger.info("Starting epoch {}/{}".format(1+epoch, MAX_EPOCHS))
            self.infer_sp()

            #"""
            self.logger.info("Converting weight matrices to lil format")
            for rel in self.ln_relationships:
                if not isinstance(self.ln_relationships[rel], np.ndarray):
                    self.ln_relationships[rel] = self.ln_relationships[rel].tolil()

            for rel in self.ll_relationships:
                if not isinstance(self.ll_relationships[rel], np.ndarray):
                    self.ll_relationships[rel] = self.ll_relationships[rel].tolil()
            #"""

            self.logger.info("Performing Stochastic Gradient Descent...")
            epoch_error = 0.0
            ##for each clique in graph, SGD weightings
            for node in tqdm.tqdm(self.model_unknowns_iter(), desc='Back propagating weights'):
                #belief = self.compute_bp_marginal(node)
                ##math.e is the range of the "true" marginal. using [0,1] is almost linear for e^x
                epoch_error += self.compute_bp_marginal_error(node)
                if 'name' not in self.model.nodes[node]:
                    IPython.embed()

                true_marginal = self.Exp.to_vec('name_vector', [ self.model.nodes[node]['name'] ])
                #true_marginal = self.Exp.to_vec('name_vector', [ self.model.nodes[node]['name'] ])
                #node_marginal = classes.pmfs.PMF.normalise_numpy_density(self.compute_bp_marginal(node))
                #marginal_diff = node_marginal - true_marginal

                ##for each contibuting factor, apply back prop
                for bp_msg in self.messages.to_node_iter(node):
                    start, end  = bp_msg.direction
                    belief      = bp_msg.value
                    #norm_belief = classes.pmfs.PMF.normalise_numpy_density(np.log(belief))
                    norm_belief = np.log(belief)
                    diff        = norm_belief - (IDEAL_WEIGHT_EXP * true_marginal)
                    #diff        = belief - true_marginal

                    rel = self.factor_graph.model.nodes[start]['relationship']
                    ##update weightings for this relationship
                    if rel in self.available_ln_relationships:
                        if len(self.ln_relationships[rel].shape) > 1:
                            end_node_index = self.factor_graph.model.nodes[start]['known_index']
                            self.ln_relationships[rel][end_node_index,:] -= alpha * diff
                        else:
                            self.ln_relationships[rel] -= alpha * diff
                    elif rel in self.available_ll_relationships:
                        var_node_start  = self.factor_graph.model.nodes[start]['start']
                        var_node_end  = self.factor_graph.model.nodes[start]['end']

                        start_name  = self.factor_graph.model.nodes[var_node_start]['name']
                        end_name    = self.factor_graph.model.nodes[var_node_end]['name']

                        start_node_index = self.Exp.to_index('name_vector', start_name)
                        end_node_index = self.Exp.to_index('name_vector', end_name)
                        """
                        ll relationships have a direction
                        start -> end

                        R matrix contains thetas for start -rel-> end
                        The generic relationship is -rel-> end, ignores start

                        """
                        self.ll_relationships[rel][end_node_index, :] -= alpha * diff
                    else:
                        raise RuntimeError("Could not determine relationship type for rel: {}".format(rel))

            #"""
            self.logger.info("Converting weight matrices to csc format")
            for rel in self.ln_relationships:
                if not isinstance(self.ln_relationships[rel], np.ndarray):
                    self.ln_relationships[rel] = self.ln_relationships[rel].tocsr()

            for rel in self.ll_relationships:
                if not isinstance(self.ll_relationships[rel], np.ndarray):
                    self.ll_relationships[rel] = self.ll_relationships[rel].tocsr()
            #"""

            ##renormalise_relationships - similar to regularization
            #self.logger.info("Normalising weight matrices")
            #self.normalise_weight_matrices()
            #self.logger.info("Applying tikinoz regularization")
            #self.apply_tikinov_regularization()
            self.logger.info("Absolute error this epoch: {}".format(epoch_error))
            if epoch_error > last_epoch_error and epoch > 2:
                self.logger.info("EPOCH error increased, stopping")
                break
            last_epoch_error = epoch_error
            #self.save_to_db()
            #IPython.embed()

        print("Finished training...")
        #IPython.embed()

    def compute_bp_marginal(self, node):
        """
            Multiply all incoming messages
            :param node: Node id in CRF model
            :return: Belief of node
            :rtype: numpy.ndarray
        """
        #print('bp_marginal:')
        #IPython.embed()
        return functools.reduce(lambda x, y: x*y.value, self.messages.to_node_iter(node), self.constraints )

    def compute_bp_marginal_error(self, node):
        """
            Compute the error in the belief at a node
            Old [ 1, 1, 1, e, 1, 1, 1 ] error is wrong.
            true marginal should be [ -inf, -inf, -inf, e, -inf]
            inf error doesn't make any sense, use error between 
        """
        true_marginal = self.Exp.to_vec('name_vector', [ self.model.nodes[node]['name'] ])
        #true_marginal = self.Exp.to_vec('name_vector', [ self.model.nodes[node]['name'] ])
        #ind = self.Exp.to_index('name_vector', self.model.nodes[node]['name'])
        belief = classes.pmfs.PMF.normalise_numpy_density(self.compute_bp_marginal(node))
        #print('bp_marginal error:')
        #IPython.embed()
        mean_squared_error = np.square(true_marginal - belief).sum()
        #mean_squared_error = np.square(true_marginal - belief).mean(axis=None)
        #mean_squared_error = np.square(true_marginal - scipy.log(belief)).mean(axis=None)
        #mean_squared_error = np.square(np.e - belief[ind])
        return mean_squared_error

    def weights_to_shared_memory(self):
        descs = []
        for rel in { **self.ln_relationships, **self.ll_relationships}.keys():
            store       = self.ln_relationships if rel in self.ln_relationships else self.ll_relationships
            rel_type    = 'ln_relationships' if rel in self.ln_relationships else 'll_relationships'
            name        = '{}.{}'.format(rel_type, rel)
            mat         = store[rel]
            desc = {}
            if isinstance(mat, np.ndarray):
                desc.update({ 'type' : 'ndarray' })
                nbytes = mat.nbytes
                try:
                    shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
                except: 
                    shm = shared_memory.SharedMemory(create=False, name=name)
                    shm.unlink()
                    shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)

                b = np.ndarray(mat.shape, dtype=mat.dtype, buffer=shm.buf)
                b[:] = mat[:]
            elif isinstance(mat, scipy.sparse.csc_matrix):
                desc.update({ 'type' : 'csc_matrix' })
                for _backing in ['data', 'indptr', 'indices']:
                    bname = name + '.' + _backing
                    bmat  = getattr(mat, _backing)
                    try: 
                        shm = shared_memory.SharedMemory(create=True, size=bmat.nbytes, name=bname)
                    except:
                        shm = shared_memory.SharedMemory(create=False, name=bname)
                        shm.unlink()
                        shm = shared_memory.SharedMemory(create=True, size=bmat.nbytes, name=bname)

                    b = np.ndarray(bmat.shape, dtype=bmat.dtype, buffer=shm.buf)
                    b[:] = bmat[:]
                    bdesc = { 'name' : bname, 'shape': bmat.shape, 'dtype': bmat.dtype }
                    desc[_backing] = bdesc
            else:
                raise RuntimeError("Error saving weights to shared memory, invalid type")

            desc.update({ 'name' : name, 'rel_type': rel_type, 'shape': mat.shape, 'rel' : rel, 'dtype': mat.dtype })
            descs.append(desc)

        return descs

    @staticmethod
    def weights_from_shared_memory(descs):
        """
            desc = [
                {'name','rel', 'rel_type', 'shape', 'dtype'}
            ]
        """
        ln_rels, ll_rels = {}, {}
        for desc in descs:
            rel_store = ln_rels if desc['rel_type'] == 'ln_relationships' else ll_rels
            if desc['type'] == 'ndarray':
                shm = shared_memory.SharedMemory(name=desc['name'])
                obj = np.ndarray(desc['shape'], dtype=desc['dtype'], buffer=shm.buf)
                rel_store[desc['rel']] = obj
            elif desc['type'] == 'csc_matrix':
                backing = {}
                for _b in ['data', 'indptr', 'indices']:
                    bdesc = desc[_b]
                    shm = shared_memory.SharedMemory(name=bdesc['name'])
                    obj = np.ndarray(bdesc['shape'], dtype=bdesc['dtype'], buffer=shm.buf)
                    backing[_b] = obj
                obj = scipy.sparse.csc_matrix((backing['data'], backing['indices'], backing['indptr']), shape=desc['shape'])
                rel_store[desc['rel']] = obj

        return ll_rels, ln_rels


    def infer(self):
        self.logger.info("Running loopy belief propagation inference")
        ##all messages set to constraints
        self.logger.debug("initing all messages")
        self.__init_bp_msg_store_mp()
        ##load weights into shared memory
        #descs = self.weights_to_shared_memory()
        #self.__init_bp_msg_store_mp(descs)

        self.logger.debug("Precomputing fixed ln messages")

        self.logger.debug("Propagating messages!")
        self.logger.info("Total messages: {}".format(len(list(self.messages.valid_msg_iter(True))) + len(list(self.messages.valid_msg_iter(False)))))
        invalid_msgs = math.inf 
        min_iters, counter = 1, 0
        while True:
            self.propagate_messages_mp()
            self.logger.debug("Done iteration..")
            lbp_it_error = functools.reduce(lambda x, node: x + self.compute_bp_marginal_error(node), self.model_unknowns_iter(), 0.0)
            self.logger.info("Error this iteration: {}".format(lbp_it_error))
            n_invalid = len(list(self.messages.valid_msg_iter(False)))

            if n_invalid == 0:
                self.logger.debug("No invalid messages, stopping propagation")
                break

            if n_invalid >= invalid_msgs:
                self.logger.debug("{} invalid messages this iteration".format(n_invalid))
                if counter+1 > min_iters:
                    break

            invalid_msgs = n_invalid

            self.logger.debug("Invalid messages: {}".format(n_invalid))
            counter += 1
            #IPython.embed()

        self.logger.debug("Calculating marginals from messages")
        ####computes marginals and assigns labels
        ###for node in tqdm.tqdm(self.model_unknowns_iter(),desc="compute node marginals"):
        #for node in self.model_unknowns_iter():
        #    #print("computing marginal for node {}".format(node))
        #    m = self.compute_bp_marginal(node)
        #    r = np.argmax(m)
        #    self.model.nodes[node]['label'] = self.Exp.name_vector[r]
        self.compute_marginals_mp()

        self.logger.debug("Done Inference")

    def infer_sp(self):
        self.logger.debug("initing all messages")
        self.__init_bp_msg_store()

        self.logger.debug("Propagating messages!")
        self.logger.info("Total messages: {}".format(len(list(self.messages.valid_msg_iter(True))) + len(list(self.messages.valid_msg_iter(False)))))

        invalid_msgs = math.inf 
        min_iters, counter = 2, 0
        while True:
            self.propagate_messages()
            self.logger.debug("Done iteration..")
            lbp_it_error = functools.reduce(lambda x, node: x + self.compute_bp_marginal_error(node), self.model_unknowns_iter(), 0.0)
            self.logger.info("Error this iteration: {}".format(lbp_it_error))

            if len(list(self.messages.valid_msg_iter(False))) == 0:
                self.logger.debug("No invalid messages, stopping propagation")
                break

            if len(list(self.messages.valid_msg_iter(False))) >= invalid_msgs:
                self.logger.debug("{} invalid messages this iteration".format(len(list(self.messages.valid_msg_iter(False)))))
                if counter+1 >= min_iters:
                    break

            invalid_msgs = len(list(self.messages.valid_msg_iter(False)))
            self.logger.debug("Invalid messages: {}".format(invalid_msgs))
            counter += 1
            
        self.logger.debug("Calculating marginals from messages")
        ####computes marginals and assigns labels
        #for node in self.model_unknowns_iter():
        for node in tqdm.tqdm(self.model_unknowns_iter(),desc="compute node marginals"):
            #print("computing marginal for node {}".format(node))
            m = self.compute_bp_marginal(node)
            r = np.argmax(m)
            self.model.nodes[node]['label'] = self.Exp.name_vector[r]
            self.model.nodes[node]['marginal'] = m

        self.logger.debug("Done Inference")

    def infer_greedy(self):
        self.logger.debug("initing all messages")
        self.__init_bp_msg_store_mp()
        #fg = self.add_messages_to_factor_graph()

        self.logger.debug("Propagating messages!")
        self.logger.info("Total messages: {}".format(len(list(self.messages.valid_msg_iter(True))) + len(list(self.messages.valid_msg_iter(False)))))
        invalid_msgs = math.inf 
        min_iters, counter = 2, 0
        while True:
            self.propagate_messages_mp()
            self.logger.debug("Done iteration..")
            lbp_it_error = functools.reduce(lambda x, node: x + self.compute_bp_marginal_error(node), self.model_unknowns_iter(), 0.0)
            self.logger.info("Error this iteration: {}".format(lbp_it_error))
            

            if len(list(self.messages.valid_msg_iter(False))) == 0:
                self.logger.debug("No invalid messages, stopping propagation")
                break

            if len(list(self.messages.valid_msg_iter(False))) >= invalid_msgs:
                if counter + 1>= min_iters:
                    break

            invalid_msgs = len(list(self.messages.valid_msg_iter(False)))
            self.logger.debug("Invalid messages: {}".format(invalid_msgs))
            counter += 1
            

        self.logger.debug("Calculating maximum belief from marginals.")
        max_belief      = -math.inf
        max_belief_ind  = None
        max_u           = None

        for node in tqdm.tqdm(self.model_unknowns_iter(),desc="compute node marginals"):
            #print("computing marginal for node {}".format(node))
            m = self.compute_bp_marginal(node)
            r = np.argmax(m)
            v = m[r]
            if v > max_belief:
                max_belief      = v
                max_belief_ind  = r
                max_u           = copy.deepcopy(node)

        #if max_belief < 1e-7:
        #    self.logger.debug("Strongest belief is too small, stopping inference")
        #    break

        self.logger.debug("Strongest belief is with unknown node: {} that we think is {}".format(max_u, self.Exp.name_vector[r]))
        self.model.nodes[node]['label'] = self.Exp.name_vector[r]

        self.logger.debug("Done Inference")

    def _mp_emit_edge_messages(self, u, v):
        msg = self.compute_message(u, v)
        if msg:
            self.messages.add(msg)
        bmsg = self.compute_message(v, u)
        if bmsg:
            self.messages.add(bmsg)

    def __init_bp_msg_store_mp(self):
        """
            Initialise all messages to vectors of ones
            and store in cache
            Edges here are between factor and variable nodes
        """

        with ThreadPoolExecutor(max_workers=64) as executor:
            msg_futures = [executor.submit(self._mp_emit_edge_messages, u, v) for u, v in self.factor_graph.model.edges()]

            for future in tqdm.tqdm(as_completed(msg_futures)):
                if future.exception():
                    IPython.embed()
                    raise RuntimeError("Error, exception occoured initialising message store")

    def __init_bp_msg_store(self):
        """
            Initialise all messages to vectors of ones
            and store in cache

            Edges here are between factor and variable nodes

            O - x
            x - O
            K - x
            x - K

            If factor is only shared between known and unknown - don't send message from unknown to factor
        """
        for u, v in tqdm.tqdm(self.factor_graph.model.edges(), desc="Computing messages"):
            msg = self.compute_message(u, v)
            if msg:
                self.messages.add(msg)
            bmsg = self.compute_message(v, u)
            if bmsg:
                self.messages.add(bmsg)

        return

        ##message are added in reverse order in order to trigger them in becoming invalid
        ##finally emit messages from all factors
        for f in tqdm.tqdm(self.factor_graph.factor_nodes, desc="factor to variable messages"):
            for node in self.factor_graph.model.neighbors(f):
                if node == f:
                    continue
                print("Computing message")
                forward_msg_update = self.compute_message(f, node)
                if not forward_msg_update:
                    print("message was invalid")
                if forward_msg_update:
                    self.messages.add(forward_msg_update)

                IPython.embed()

        ##emit messages from unknowns to label-label factors
        for k in tqdm.tqdm(self.factor_graph.variable_nodes, desc="unknown to factor messages"):
            for node in self.factor_graph.model.neighbors(k):
                if node == k:
                    continue

                ##message may return none if it's N/A
                forward_msg_update = self.compute_message(k, node)
                if forward_msg_update:
                    self.messages.add(forward_msg_update)

    def _mp_compute_marginal(self, node):
        #print("computing marginal for node {}".format(node))
        m = self.compute_bp_marginal(node)
        r = np.argmax(m)
        self.model.nodes[node]['label'] = self.Exp.name_vector[r]
        self.model.nodes[node]['marginal'] = m

    def marginals_from_fps(self):
        """
            Ignore CRF. Compute marginal taking into account fingerprint only
            ***warning: depricated. testing only***
        """
        for node in self.model_unknowns_iter():
            self.model.nodes[node]['marginal'] = self.model.nodes[node]['fingerprint'].reshape(-1)

    def compute_marginals_mp(self):
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(self._mp_compute_marginal, node) for node in self.model_unknowns_iter()]
            for future in as_completed(futures):
                if future.exception():
                    raise RuntimeError("Error, exception occurred computing unknown node marginals")


    def _mp_propagate_msg(self, msg):
        msg_update = self.compute_message(msg.start, msg.end)
        if msg_update:
            self.messages.add(msg_update)

    def propagate_messages_mp(self):
        invalid_now = list(self.messages.valid_msg_iter(False))
        #shuffle message updates
        random.shuffle(invalid_now)

        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(self._mp_propagate_msg, msg) for msg in invalid_now]
            for future in as_completed(futures):
                if future.exception():
                    raise RuntimeError("Error, exception occurred propagating messages")

    def propagate_messages(self):
        invalid_now = list(self.messages.valid_msg_iter(False))
        random.shuffle(invalid_now)
        for msg in tqdm.tqdm(invalid_now):
            msg_update = self.compute_message(msg.start, msg.end)
            if msg_update:
                self.messages.add(msg_update)

    def compute_message(self, start, end):
        return self.compute_message_sum_product(start, end)

    def compute_message_sum_product(self, start, end):
        """
        Uses the Sum-Product algorithm
            Uses a factor graphs nodes/edges!!!
            m_x->t(x_t) = SUM_x_s [ f_st(x_s, x_t) PROD_neigh(s) { \t m_u->s(x_s) }  ]
            
            Message is function of x_t, x_t is a random variable. It does not have an assignment?
        """
        if start in self.factor_graph.factor_nodes:
            """
                Computes a message from factors to variables
                This is given as the sum over variables not {{end}} of the product
                of all of their factors

                NB: Sum over incoming nodes (product over all relationships per incoming node)

                m_as(y_s) = SUM_(y_v_a \y_s) [ PROD_(iota_b \in F_a) [ iota_b(y_b) ] ]
            """
            #Factor node to variable node message
            #PRODUCT of all incoming messages (NB: label-node factors have no incoming messages)

            #create default msg with default value of 1 
            msg = BPMessage(self._dim, start, end, self.constraints)
            rel = self.factor_graph.model.nodes[start]['relationship']

            ##multiply by factor associated with this 
            if rel in self.available_ln_relationships:
                ff = self.factor_graph.model.nodes[start]['ff']
                if len(ff.shape) == 1:
                    belief = ff * self.ln_relationships[rel]
                else:
                    belief = ff @ self.ln_relationships[rel]
            else:
                assert(rel in self.available_ll_relationships)
                ##incoming messages into start
                incoming_messages = self.messages.to_node_iter(start)
                #if(len(incoming_messages) == 0):
                #    self.logger.warning("Emitting placeholder default message")

                #incoming_belief = functools.reduce(lambda x, y: x + y.value, incoming_messages, msg.value*0.0)
                incoming_belief = functools.reduce(lambda x, y: x + y.value, incoming_messages, np.zeros((self._dim,), dtype=np.float64))
                belief = incoming_belief @ self.ll_relationships[rel]

            #belief = classes.pmfs.PMF.normalise_numpy(np.exp(belief))
            #belief = classes.pmfs.PMF.normalise_numpy_density(np.exp(belief))
            belief = np.exp(belief)
            msg.value = belief
            if(len(belief) != self.Exp.name_vector_dims):
                print("Belief wrong shape")
                IPython.embed()
            return msg

        elif start in self.factor_graph.variable_nodes:
            """
                Computes a message from variables to factors.
                This is given as the sum over all factors of the product
                of all messages in that factor

                m_sa(y_s) = SUM_y_v_s [ PROD_(iota_b \in F_s)  iota_b(y_b) ] ]

                i.e. sum of product of feature functions per factor
            """
            #variable node to factor node message
            #PRODUCT over alls incoming factors

            ##don't send a message back to a single factor
            rel = self.factor_graph.model.nodes[end]['relationship']
            if rel in self.available_ln_relationships:
                return None

            #create default msg with default value of 1 
            msg = BPMessage(self._dim, start, end, self.constraints)

            assert(rel in self.available_ll_relationships)
            ##calculate based on incoming messages, we are unknown
            belief = functools.reduce(lambda x, y: x * y.value, self.messages.to_node_iter(start), msg.value)

            #belief is a matrix, trun into an array
            if not isinstance(belief, np.ndarray):
                belief = np.asarray(belief.todense())[0]

            if(len(belief) != self.Exp.name_vector_dims):
                print("Belief wrong shape")
                IPython.embed()

            msg.value = classes.pmfs.PMF.normalise_numpy_density(belief)
            #msg.value = belief
            return msg

    def check_accuracy(self, confidence=1.61, top_n=1):
        tp, fp = 0, 0
        tn, fn = 0, 0
        for node in list(self.model_unknowns_iter()):
            corr_name   = self.model.nodes[node]['name']

            connected_knowns = 0
            connected_unknowns = 0
            for v in self.model[node]:
                #IPython.embed()
                if self.model.nodes[v]['name'] in self.knowns:
                    connected_knowns+=1
                else:
                    connected_unknowns+=1

            if 'marginal' not in self.model.nodes[node]:
                print("node has no marginal")
                IPython.embed()

            if np.max(self.model.nodes[node]['marginal']) < confidence:
                fn += 1
                inf_name = self.Exp.name_vector[ self.model.nodes[node]['marginal'].argmax() ]
                self.logger.debug("{:<40}--{:<3} NOT CONFIDENT {:<3}->{:>40}".format(corr_name, connected_knowns, connected_unknowns, inf_name))
                continue

            self.logger.debug("{:<40}--{:<3} {:<3}-┐".format(corr_name, connected_knowns, connected_unknowns))
            self.logger.debug("┌{}-{}-{}-┘".format("-"*40, "-"*3, "-"*3))

            nsorted = self.model.nodes[node]['marginal'].argsort()
            CORRECT = False
            for i in range(top_n):
                guess = nsorted[-(i+1)]
                bullet = "├"
                if i+1 == top_n:
                    bullet = "└"

                inf_name = self.Exp.name_vector[guess] 
                if corr_name == inf_name:
                    CORRECT = True

                self.logger.debug("{}- {:<40}".format(bullet, inf_name))

            if CORRECT:
                tp += 1
            else:
                fp += 1

        for node in list(self.model_knowns_iter()):
            name = self.model.nodes[node]['name']
            if name in list(self.assumed_knowns):
                self.logger.debug("{:<40}--ASS ASS->{:>40}".format(name, name))
                #tp += 1

        accuracy = 0.0
        p = 0.0
        r = 0.0
        f1 = 0.0
        if (tp+tn+fp+fn) > 0:
            accuracy = (tp+tn) / (tp+tn+fp+fn)
            p = 0
            if tp + fp != 0:
                p = tp / (tp + fp)
            r = 0
            if tp + fn != 0:
                r = tp / (tp + fn)
            f1 = 0
            if p + r != 0:
                f1 = (2 * p * r) / (p + r)
        self.logger.debug("TP : {}, TN: {}, FN: {}, FP: {}, ACC: {}".format(tp, tn, fn, fp, accuracy))
        self.logger.debug("P: {}, R: {}, F1: {}".format(p, r, f1))


