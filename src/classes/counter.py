#!/usr/bin/python3
import json
import pymongo
import random
import logging
import pprint
import copy
import numpy as np
import scipy as sp
import scipy
import functools
import collections



#def to_npvec(counter, dim, name_to_index):
def to_scipy_sparse_vec(counter, dim, name_to_index):
    """
        Convert dCounter into a vector given a hashmap of keys to an index in the vector

        :param dict name_to_index: A hashmap of keys to index in the vector
        :param int dim: dimension of the vector
        :return: a numpy array vector
        :rtype: np.array
    """
    vec = sp.sparse.dok_matrix( (1, dim), dtype=sp.float128 )
    for k, v in counter.items():
        #lookup key index
        ind = name_to_index[k]
        vec[0, ind] = sp.float128(v)
    return vec

def to_scipy_sparse_vec_prob(counter, dim, name_to_index):
    """
        Convert dCounter into a vector given a hashmap of keys to an index in the vector

        :param dict name_to_index: A hashmap of keys to index in the vector
        :param int dim: dimension of the vector
        :return: a numpy array vector
        :rtype: np.array
    """
    total = sum(cunter.values())
    ## lil_matrix and dok_matrix cannot be saved. Soring N rows and 1 column so use column matrix
    vec = sp.sparse.dok_matrix( (1, dim), dtype=sp.float128 )
    for k, v in counter.items():
        #lookup key index
        ind = name_to_index[k]
        vec[ind] = sp.float128(v) / sp.float128(total)
    return vec

def unique_keys(counter):
    """
        Number of unique keys in counter
    """
    return len( counter.items() )






class dCounter:

    #use Database.client as MongoDB reference
    def __init__(self):
        pass
    def to_npvec_prob(self, dim, name_to_index):
        """
            Convert dCounter into a vector given a hashmap of keys to an index in the vector

            :param dict name_to_index: A hashmap of keys to index in the vector
            :param int dim: dimension of the vector
            :return: a numpy array vector
            :rtype: np.array
        """
        vec = np.zeros( dim, dtype=np.float64 )
        #total = float( functools.reduce(lambda x, y: x+y, self.store.values(), 0.0 ) )
        total = self.total
        #self.logger.info("Total: {}, Unique: {}".format(total, len(self.store.keys())))
        for k, v in self.store.items():
            #lookup key index
            ind = name_to_index[k]
            vec[ind] = float( v )/ total 

        return vec

    def to_npvec_unique_prob(self, dim, name_to_index):
        """
            Convert dCounter into a vector given a hashmap of keys to an index in the vector.
            The returned vector is a probability that ignores the counts (>1) in the counter. 

            :param dict name_to_index: A hashmap of keys to index in the vector
            :param int dim: dimension of the vector
            :return: a numpy array vector
            :rtype: np.array
        """
        vec = np.zeros( dim, dtype=np.float64 )
        entries = float( len(self.store.keys() ) )
        #self.logger.info("Total: {}, Unique: {}".format(total, len(self.store.keys())))
        for k, v in self.store.items():
            #lookup key index
            ind = name_to_index[k]
            vec[ind] = 1.0/ entries

        return vec
