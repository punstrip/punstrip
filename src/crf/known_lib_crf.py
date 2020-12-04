#!/usr/bin/python3
import os, sys, gc
import logging, math, resource
import numpy as np
import scipy, dill
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import functools
import itertools
import copy, re, subprocess
import random
import networkx as nx
import glob
from tqdm import tqdm
from networkx.drawing.nx_pydot import write_dot 
#from multiprocessing import Pool
import multiprocess
from multiprocess.pool import ThreadPool
import IPython



import context
from classes.symbol import Symbol
from classes.database import Database
from classes.config import Config
from classes.counter import dCounter
from classes.bin_mod import BinaryModifier
import classes.NLP
import classes.utils
import classes.callgraph
import crf.utils
import classes.binary
#import scripts.perform_analysis

random.seed()

#logging.basicConfig(level=logging.DEBUG , format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')



global GG
global n
global P

global bin_path
global linkage
global compiler
global bin_name
global optimisation

global nth_callees
global nth_callers

def dict_indexes_to_probabilities( d ):
    """
        convert dict of { symbol indexes : occourances or counts } to probabilities normalised to 1
        e.g. { 0: 2, 1:2, 3:4, 4:1, 5:1} -> [ 1:0.2, 2:0.2, 3:0.4, 4:0.1, 5:0.1 ]
    """
    assert( isinstance(d, dict) )
    total = float( functools.reduce( lambda x, y: x + y, list( map(lambda kv: kv[1], d.items()) ) ) ) 
    return dict( map( lambda kv: (kv[0] , float(kv[1])/total), d.items()) )

def occourances_to_probabilities( vec ):
    """
        convert list of occourances or counts to probabilities normalised to 1
        e.g. [ 2, 2, 4, 1, 1] -> [ 0.2, 0.2, 0.4, 0.1, 0.1 ]
    """
    assert( isinstance(vec, list) )
    total = float( sum(vec) )
    return list( map( lambda x: float(x)/vec, vec) )

def build_dynamic_binary_constraints( db, symbol_names, name_to_index ):
    """
    Build constraints which limit symbols to dynamic symbols
    :param db: A Database instance
    :param symbol_names: A list of strings for symbol_names that we wish to check
    :param name_to_index: Hahsmap of symbol name to index in vector
    :return: A numpy matrix with elements either 0.0 or 1.0. Elements that are 0.0 represent symbols that only exist in static binaries
    :rtype: numpy.matrix
    """
    d = len(symbol_names)
    vec = np.ones( (d, 1), dtype=np.float)
    for name in symbol_names:
        if db.is_static_only_symbol(name):
            vec[ name_to_index[name], 0 ] = 0.0
    return vec

def sum_feature_function(G, ff_mat, nodes, name_to_index):
    """
    p(y) = p(y | x) * p(x)
    Returns the probability distribution over the combined nodes
    :param NetworkX.DiGraph G: The callgraph with each node containing node potentials
    :param np.matrix ff_mat: Feature Function matrix conecting symbols -> symbols for a feature function
    :param list[str] nodes: list of nodes this feature function maps to
    :return: Probabilty mass function of the most likly node (vector) that connects his feature function
        to the nodes given the nods potentials
    :rtype np.array:
    """
    n, m = np.shape( ff_mat )
    assert(n==m)
    pmfs = []

    for node in nodes:
        node_potential = G.nodes[ node ]['node_potential']
        #pmf = np.multiply( ff_mat, node_potential )
        #np_t = np.transpose( node_potential )
        # ff_mat * node_potential is the same as x^t = node_pot^t * ff_mat^t
        pmf = ff_mat @ node_potential
        #pmf = np_t @ ff_mat_t
        pmfs.append( pmf )

    pmf = functools.reduce(lambda x, y: np.add(x, y), pmfs, np.zeros((m,1 ), dtype=np.float128))
    #return P.normalise_numpy_density( pmf )
    return pmf

#NB: Not used
def cg_mat_lookup(cg_mat, name_to_index, parents):
    """
    Returns the probability distribution over the combined parents
    :param cg_mat: Callgraph matrix cg_mat[r, c] represents the probability of r transitioning to c
    :param name_to_index: Hashmap to convert symbols names to vector indexes
    :param parents: List of symbol names of parents. 
    :return: A matrix probability mass function that represents the probability of the given symbol
    :rtype: numpy.matrix
    """
    assert(len(parents) > 0)

    n, m = np.shape( cg_mat )
    assert(n==m)
    pmfs = []

    for p_symb in parents:
        p_ind = name_to_index[p_symb]
        #print("mat index lookup")
        pmf = cg_mat[p_ind, :]
        #print("mat index lookup complete")
        #m = np.shape(pmf)
        #logger.info("Shape of cg_mat[{}][:] is: {}".format( p_ind, m))
        pmfs.append(pmf)
        #print("matrix appended")

    if len(parents) == 1:
        return pmfs[0]

    #print("reducing pmfs")
    #cpf = functools.reduce(lambda x, y: rescale( x + np.transpose(y) ), pdfs)
    pmf = functools.reduce(lambda x, y: np.add(x, y), pmfs)
    #print("done")
    return pmf

#NB: Not used
def rescale_prob(vec):
    """
    Rescale a vector so that the elements sum to 1.0
    :param vec: A numpy.matrix vector
    :return: A rescaled vector
    :rtype: numpy.matrix
    """
    m = np.shape(vec)
    s = np.sum(vec)
    ### WTF BUG, np.spare.csr is of type numpy only when sum?!?1?1
    #here we have tye csr mat
    if isinstance(s, csr_matrix):
        npvec = s.toarray()
        fs = float( np.sum( npvec ) )
        if fs == 0:
            return vec
        c = float(1.0)/fs
        return np.multiply(c, npvec)

    fs = float(np.sum(s))
    if fs == 0.0:
        return vec
    c = float(1.0)/fs
    return np.multiply(c, vec)

def rescale( vec ):
    """
    Rescale a vector so all of its values are between 0.0 and 1.0
    :param vec: A numpy.matrix vector
    :return: A rescaled vector
    :rtype: numpy.matrix
    """
    min_ind = np.argmin( vec )
    max_ind = np.argmax( vec )
    min_v = vec[min_ind]
    max_v = vec[max_ind]

    m = np.shape(vec)
    #print(m)
    #print(m[0])

    nvec = np.zeros( m )
    for i in range(m[0]):
        #print(np.shape(nvec))
        #print(i)
        nvec[i] = (vec[i] - min_v) / (max_v - min_v)

    return nvec

def row_sparse_from_dok(sparse):
    """
    Convert a sparse dok matrix into a sparse row matrix
    :param: scipy.sparse.dok_matrix
    :return: A sparse matrix
    :rtype: scipy.sparse.csr_matrix
    """
    r, c = np.shape( sparse )
    logger.info("Building numpy matrix from sparse matrix...")
    print(r, c)
    d = csr_matrix( (r, c), dtype=np.float )
    for r in range(r):
        for c in range(c):
            #print(sparse[r, c])
            d[r, :] = sparse[r, :]

    logger.info("Done!")
    return d

def npmat_from_sparse(sparse):
    """
    Convert a scipy.sparse matrix into a numpy.matrix
    :param: scipy.sparse
    :return: A numpy matrix
    :rtype: numpy.matrix
    """
    r, c = np.shape( sparse )
    logger.info("Building numpy matrix from sparse matrix...")
    #print(r, c)
    d = np.zeros( (r, c), dtype=np.float )
    for r in range(r):
        for c in range(c):
            #print(sparse[r, c])
            d[r, c] = sparse[r, c]

    logger.info("Done!")
    return d

def nth_parents(G, node, n):
    """
    Calculate the nth parents of a node in a callgraph
    :param G: The callgraph
    :param node: The node
    :param: The order n
    :return: list of nodes that correspond to the nth parents
    :rtype: list
    """
    _parents = _rec_nth_parents(G, node, n)
    return _rec_unravel( _parents )

def _rec_nth_parents(G, node, n):
    """
    Recursively compute the nth parents of a node
    :param G: networkx.DiGraph of callgraph
    :param node: Current node
    :param n: Number of parents away
    :return: A node or list of lists nodes
    :rtype: Either a node or list of nodes or list^n(nodes)
    """
    if n == 0:
        return node

    if n > 0:
        return list(map(lambda x: _rec_nth_children(G, x, n-1), G.predecessors( node )))

def nth_children(G, node, n):
    """
    Calculate the nth children of a node in a callgraph
    :param G: The callgraph
    :param node: The node
    :param: The order n
    :return: list of nodes that correspond to the nth children
    :rtype: list
    """
    _children = _rec_nth_children(G, node, n)
    return _rec_unravel( _children )

def _rec_nth_children(G, node, n):
    """
    Recursively compute the nth children of a node
    :param G: networkx.DiGraph of callgraph
    :param node: Current node
    :param n: Number of children away
    :return: A node or list of lists nodes
    :rtype: Either a node or list of nodes or list^n(nodes)
    """
    if n == 0:
        return node

    if n > 0:
        return list(map(lambda x: _rec_nth_children(G, x, n-1), G.successors( node )))

#def __count_nth_symbol_rels(GG, symbol_name, n):
def __count_nth_symbol_rels(symbol_name):
    global GG
    global n
    assert(isinstance(GG, list))
    assert(isinstance(symbol_name, str))
    assert(isinstance(n, int))

    logger.info("Building {}{} cg for {}".format(n, classes.utils.ordinal_follower(n), symbol_name))
    callees = dCounter()
    callers = dCounter()
    #build nth degree callee counts for all callgraphs
    for G in GG:
        if symbol_name not in G.nodes:
            continue

        bin_callees = nth_children(G, symbol_name, n)
        bin_callers = nth_parents(G, symbol_name, n)

        #print(callees)
        assert(isinstance(bin_callees, list))
        assert(isinstance(bin_callers, list))

        for item in bin_callees:
            callees += item

        for item in bin_callers:
            callers += item

    #no loops
    callees.remove_node( symbol_name )
    callers.remove_node( symbol_name )

    #logger.info("Callees:" + str(callees))
    #logger.info("Callers:" + str(callers))
    #logger.info(callers - callees)

    return callees, callers, symbol_name

def _rec_unravel(lst):
    if isinstance(lst, str):
        return [ lst ]
    big_list = []
    for s in lst:
        if isinstance(s, list):
            big_list += _rec_unravel(s)
        else:
            big_list += [ s ]
    return big_list

def build_nth_from_cg(symbol_names, _GG, _n, name_to_index):
    d = len(symbol_names)
    nth_callee_count = lil_matrix( (d, d), dtype=np.uint )
    nth_caller_count = lil_matrix( (d, d), dtype=np.uint )
    nth_callee = lil_matrix( (d, d), dtype=np.float )
    nth_caller = lil_matrix( (d, d), dtype=np.float )
    
    global GG
    global n
    n = _n
    GG = _GG

    mp = Pool(processes=32)
    #args = list(map(lambda x: (GG, x, n), symbol_names))
    #res = mp.starmap(__count_nth_symbol_rels, args )
    res = mp.map(__count_nth_symbol_rels, symbol_names )

    logger.info("Symbol relations calculated!")
    logger.info("Finalising matrix")

    for callees, callers, symbol_name in res:
        s_index = name_to_index[ symbol_name ]
        nth_callee_count[s_index, :] = callees.to_npvec( (d,), name_to_index)
        nth_caller_count[s_index, :] = callers.to_npvec( (d,) , name_to_index)
        nth_callee[s_index, :] = callees.to_npvec_prob( (d,), name_to_index)
        nth_caller[s_index, :] = callers.to_npvec_prob( (d,) , name_to_index)

    mp.close()
    return nth_callee, nth_caller, nth_callee_count, nth_caller_count

def gen_new_symbol_indexes(db):
    distinct_symbols = set( db.distinct_symbol_names() )
    symbs_from_rels = db.flatten_callees_callers()
    distinct_symbols = list( distinct_symbols.union( set(symbs_from_rels) ) )
    dim = len(distinct_symbols)

    symbol_to_index = dict( map( lambda x: [distinct_symbols[x], x], range(dim)) )
    index_to_symbol = dict( map( lambda x: [x, distinct_symbols[x]], range(dim)) )
    return distinct_symbols, symbol_to_index, index_to_symbol

def get_random_sample( l, n ):
    """
        return a random sample of n elements in l
    """
    N = len(l)
    inds = set([])
    while len(inds) < n:
        inds.add( random.randint(0, N-1) )
    return list( map( lambda x: l[x], inds) )

def gradient_descent_pmf(old_pmf, new_pmf):
    #GD_alpha
    diff = np.subtract( old_pmf , new_pmf )
    correction = np.multiply( GD_alpha, diff )
    return np.subtract( old_pmf, correction )

def score_crf(unknowns, G, pmfs):
    H = copy.deepcopy(G)

def update_crf(unknowns, G, cgs, name_to_index, index_to_name, constraints, pmfs, updated_node):
    del pmfs[updated_node]

    first_callers = set( nth_parents(G, updated_node, 1) )
    first_callees = set( nth_children(G, updated_node, 1) )

    second_callers = set( nth_parents(G, updated_node, 2) )
    second_callees = set( nth_children(G, updated_node, 2) )


    #update all nodes constraints
    for node in pmfs.keys():
        pmfs[node] = np.multiply( pmfs[node], constraints )

    neighbours = functools.reduce(lambda x, y: x.union(y), [ first_callers, first_callees, second_callers, second_callees], set([]))
    #delete neigbours pmfs from dict
    for node in neighbours:
        if node in pmfs:
            del pmfs[node]

    #recalculate probabilities for updated nodes only
    for node in neighbours:
        if node not in unknowns:
            continue

        #logger.info("\nNode index is {} and node is {}".format(i, node))
        pmf = np.zeros([N, 1], dtype=np.float)
        n = 1
        for ffi, ffr,  in cgs:
            #find ffn and ffrn
            nth_callers = set( nth_parents(G, node, n) )
            nth_callees = set( nth_children(G, node, n) )

            fi_pmf = sum_feature_function(G, ffi, nth_callees, name_to_index)
            ri_pmf = sum_feature_function(G, ffr, nth_callers, name_to_index)

            #logger.info("Connectivity of node: {}".format( len(nth_callers) + len(nth_callees) ))

            pmf = np.add( pmf, fi_pmf )
            pmf = np.add( pmf, ri_pmf )

            n+=1


        pmf = np.multiply(pmf, constraints)
        node_potential = G.nodes[ node ]['node_potential']
        pmf = np.multiply( pmf, node_potential)
        pmfs[node] = pmf
    return pmfs


def most_probable_symbols(pmfs, n=5):
    top_n = {}
    for node in pmfs.keys():
        pmf = pmfs[node]

        if type(pmf) == type(np.ndarray((2, 1))):
            pass

        #assert(type(pmf) == np.ndarray)
        r, c = np.shape(pmf)
        assert(c == 1)

        """
        #np.argmax fucks up when all elements are 0. Different data type returns, not consistent.
        r, c = np.where(pmf > 0.0)
        assert(len(r) == len(c))
        if len(r) <= n:
            import IPython
            IPython.embed()

            py_list_sorted_pmf = [ 0, 1, 2, 3, 4 ] #take the first 5 elements
        else:
        """
        #highest n 
        sorted_pmf = np.argsort(pmf[:,0], axis=None)
        #print(type(sorted_pmf))
        #print(sorted_pmf)
        if type(pmf) == type(np.ndarray((2, 1))):
            py_list_sorted_indexes = list(map(lambda x: sorted_pmf[x-1], range(r, r-n, -1)))
        else:
            py_list_sorted_indexes = list(map(lambda x: sorted_pmf[0, x-1], range(r, r-n, -1)))
        #print("worked")

        node_probs = []
        for i in range(0, n):
            ith_index = py_list_sorted_indexes[i]
            node_probs.append( [ith_index, pmf[ ith_index, 0] ] )
        top_n[ node ] = node_probs

    return top_n



def emit_callee_message_update_from_mat(G, cg, u, v, dim, index_to_name, theta):
    assert(isinstance(u, str))
    assert(isinstance(v, str))
    """
    Perform 2.6 million relationships update by check non-zero elements of v. 
    We create a new node potntial for u and v based of calls to v and calls from u
    Find non-zero elements of scipy vector(matrix) u and v, them multiply by scipy sparse matrix relationship
    w(m+1) = w(m) - n*err(m)

    Emit message from v (from probability of u calling v)
    """
    vpot = G.nodes[v]['node_potential']
    ## multiplication between cg and theta is element wise
    ## multiplication between vpot and cg is done as matrix 
    msg =  scipy.transpose(vpot) @ cg.multiply(theta)
    return scipy.transpose( msg )

def emit_caller_message_update_from_factor_knowns(G, factor_cgr, knowns, index_to_name):
    kf = frozenset(knowns) 
    names = []
    max_score = 0.0

    pmf = scipy.sparse.lil_matrix( (len(name_to_index.keys()), 1), dtype=scipy.float128)
    if len(knowns) == 0:
        return pmf

    #find most similar factor
    for k, v in factor_cgr.items():
        inter = kf.intersection(k)
        if len(inter) > 0:
            names.append( (v, float( len(inter) ) / float( max(len(kf), len(k) ) ) ) )

    #print("About to sort")
    #import IPython
    #IPython.embed()

    #sort by overlap
    names = sorted(names, key=lambda x: x[1])[::-1]

    #intersection of 1 shoudl be highest in PMF, could be lots of small combinations e.g. main
    used_funcs = set([])
    for counter, overlap in names:
        for name, freq in counter.most_common():
            if name not in used_funcs:
                pmf[ name_to_index[name], 0] += overlap
                used_funcs.add(name) 

    #print("About to return")
    #import IPython
    #IPython.embed()
    return pmf


def emit_caller_message_update_from_mat(G, cg, u, v, dim, index_to_name, theta):
    assert(isinstance(u, str))
    assert(isinstance(v, str))
    """
    Perform 2.6 million relationships update by check non-zero elements of v. 
    Emit message for relationship u calls v
    Find non-zero elements of scipy vector(matrix) u and v, them multiply by scipy sparse matrix relationship
    w(m+1) = w(m) - n*err(m)

    Emit message from u (from probability of v being called by u). i.e. What does u call?
    """
    upot = G.nodes[u]['node_potential']
    msg = scipy.transpose(upot) @ cg.multiply(theta)
    return scipy.transpose( msg )


def emit_callee_message_update_from_rel(G, rels, u, v, dim, index_to_name):
    assert(isinstance(u, str))
    assert(isinstance(v, str))
    """
    Perform 2.6 million relationships update by check non-zero elements of v. 
    We create a new node potntial for u and v based of calls to v and calls from u
    Find non-zero elements of scipy vector(matrix) u and v, them multiply by scipy sparse matrix relationship
    w(m+1) = w(m) - n*err(m)

    Emit message from u from calling v
    """
    upot = G.nodes[u]['node_potential']
    vpot = G.nodes[v]['node_potential']

    #sum callee relationships from u to v
    #sum each relationship, then raise as power of exp
    w = scipy.zeros( (dim, 1), dtype=scipy.float128 )
    r, c = scipy.where(vpot > 0.0)
    #c should all be 0
    for i in tqdm(r):
        name = index_to_name[i]
        pmf = rels[name + "_callee"]
        w += scipy.multiply( upot[i, 0] , pmf )

    return w

def emit_caller_message_update_from_rel(G, rels, u, v, dim, index_to_name):
    assert(isinstance(u, str))
    assert(isinstance(v, str))
    """
    Perform 2.6 million relationships update by check non-zero elements of v. 
    Emit message for relationship u calls v
    Find non-zero elements of scipy vector(matrix) u and v, them multiply by scipy sparse matrix relationship
    w(m+1) = w(m) - n*err(m)

    Emit message from v from being called by u
    """
    upot = G.nodes[u]['node_potential']
    vpot = G.nodes[v]['node_potential']

    #sum caller relationships from v to u
    #sum each relationship, then raise as power of exp
    x = scipy.zeros( (dim, 1), dtype=scipy.float128 )
    r, c = scipy.where(upot > 0.0)
    #c should all be 0
    for i in tqdm(r):
        name = index_to_name[i]
        pmf = rels[name + "_caller"]
        x += scipy.multiply( vpot[i, 0] , pmf )

    return x

def perform_loopy_belief_batch_iteration(G, rels, unknowns, index_to_name, constraints, P):
    """
    Perform batch iteration of loopy belief propagation. 
    Calculate weighting updates for all network before updating them
    """
    N = len(index_to_name)
    H = copy.deepcopy(G)
    randomly_ordered_nodes = random.sample(unknowns, len(unknowns))
    #learning_rate = 1.0 / math.e
    learning_rate = 1.0
    total_error = 0.0

    for node in randomly_ordered_nodes:
        orig_node_pmf = G.nodes[ node ]['node_potential']
        new_node_pmf = scipy.ones( (N, 1), dtype=scipy.float128 )

        print("Using node " + node)

        for u, v in G.edges(nbunch=node):
            sys.stdout.write('.')

            if node == u:
                msg_update_for_node = emit_callee_message_update_from_rel(G, rels, u, v, N, index_to_name)
                new_node_pmf *= msg_update_for_node
            else:
                msg_update_for_node = emit_caller_message_update_from_rel(G, rels, u, v, N, index_to_name)
                new_node_pmf *= msg_update_for_node

        #add constrainst
        new_node_pmf *= constraints

        normalised_node_pmf = P.normalise_numpy_density( new_node_pmf )
        error = orig_node_pmf - normalised_node_pmf
        total_error += scipy.sum(error)

        updated_node_pmf = orig_node_pmf - ( learning_rate * error )
        H.nodes[node]['node_potential'] = updated_node_pmf

    return H, total_error


def update_single_node_max_product(node):

    global _cg
    global _cgr
    global _G
    global _dim
    global _index_to_name
    global _P
    global _constraints
    global _learning_rate

    orig_node_pmf = _G.nodes[ node ]['node_potential']
    #new_node_pmf = scipy.zeros( (_dim, 1), dtype=scipy.float128 )
    new_node_pmf = scipy.ones( (_dim, 1), dtype=scipy.float128 )

    max = 0.0

    for u, v in G.edges(nbunch=node):
        if node == u:
            #emits message from v with the probability of v being called
            #msg_update_for_node = emit_callee_message_update_from_rel(G, cg, cgr, u, v, N, index_to_name)
            msg_update_for_node = emit_callee_message_update_from_mat(_G, _cgr, u, v, _dim, _index_to_name)
            mm = np.max(msg_update_for_node)
            if mm > max:
                max = mm
                new_node_pmf = msg_update_for_node
        else:
            #emits message from u with the probability of what u calls
            #msg_update_for_node = emit_caller_message_update_from_rel(G, cg, cgr, u, v, N, index_to_name)
            msg_update_for_node = emit_caller_message_update_from_mat(_G, _cg, u, v, _dim, _index_to_name)
            mm = np.max(msg_update_for_node)
            if mm > max:
                max = mm
                new_node_pmf = msg_update_for_node

    new_node_pmf = scipy.exp( new_node_pmf )
    #add constrainst
    new_node_pmf *= constraints

    normalised_node_pmf = _P.normalise_numpy_density( new_node_pmf )
    error = orig_node_pmf - normalised_node_pmf
    #total_error += scipy.sum(error)

    updated_node_pmf = orig_node_pmf - ( _learning_rate * error )
    return updated_node_pmf, error


def update_single_node_sum_product(node):
    #nth_root = 2.0

    global _cg
    global _cgr
    global _G
    global _dim
    global _index_to_name
    global _constraints
    global _learning_rate
    global _theta
    global _thetar
    global _factor_cgr

    #product of exponentials is the same as the exponential of the sum

    orig_node_pmf = _G.nodes[ node ]['node_potential']
    new_node_pmf = scipy.zeros( (_dim, 1), dtype=scipy.float128 )
    #new_node_pmf = scipy.ones( (_dim, 1), dtype=scipy.float128 )

    #for u, v in G.edges(nbunch=node):
    for u in G.predecessors(node):
        v = node

        #skip data refrences
        if G[u][v]['data_ref']:
            continue

        #emits message from u with the probability of u calling Y
        #msg_update_for_node = emit_callee_message_update_from_rel(G, cg, cgr, u, v, N, index_to_name)
        #msg_update_for_node = emit_caller_message_update_from_mat(_G, _cgr, u, v, _dim, _index_to_name)
        msg_update_for_node = emit_caller_message_update_from_mat(_G, _cg, u, v, _dim, _index_to_name, _theta)
        #msg_update_for_node = scipy.exp( msg_update_for_node )

        new_node_pmf += msg_update_for_node
        #new_node_pmf *= msg_update_for_node

    for v in G.successors(node):
        u = node

        #skip data refrences
        if G[u][v]['data_ref']:
            continue

        #else:
        #emits message from v with the probability of what u calls
        #msg_update_for_node = emit_caller_message_update_from_rel(G, cg, cgr, u, v, N, index_to_name)
        msg_update_for_node = emit_callee_message_update_from_mat(_G, _cgr, u, v, _dim, _index_to_name, _thetar)
        #msg_update_for_node = scipy.exp( msg_update_for_node )

        new_node_pmf += msg_update_for_node
        #new_node_pmf *= msg_update_for_node

    #factor calling knowns
    known_callees = list(filter(lambda x: _G.nodes[x]['func'] and not _G.nodes[x]['text_func'], _G.successors(node)))
    new_node_pmf += emit_caller_message_update_from_factor_knowns(_G, _factor_cgr, known_callees, _index_to_name)

    if scipy.shape(new_node_pmf) != scipy.shape(_constraints):
        raise Exception("Error, matrix shape missmatch between constraints and node PMF in update single node sum product")

    #add constraints
    summed_pmf = scipy.multiply( new_node_pmf,  _constraints )

    if scipy.shape(_constraints) != np.shape(summed_pmf):
        raise Exception("Error, matrix shape missmatch between constraints and summed PMF in update single node sum product")

    #summed_pmf = _P.normalise_numpy_density( summed_pmf )


    #new_node_pmf = scipy.exp( new_node_pmf / nth_root )
    new_node_pmf = scipy.exp( summed_pmf )

    #normalised_node_pmf = _P.normalise_numpy_density( new_node_pmf )
    normalised_node_pmf = new_node_pmf
    error = orig_node_pmf - normalised_node_pmf

    updated_node_pmf = orig_node_pmf - ( _learning_rate * error )

    return updated_node_pmf, error

def compute_z_x(G, unknowns):
    """
        Compute Z(x) by setting all nodes y to 1 and compute sum of all nodes
        ERROR - This function calculates prod(sum) not sum(prod)
        ERROR - This function should not be used
    """
    raise Exception("Error, this function is incorrect, please do not use it")
    #perform inference with all nodes in y set to all possible values at the same time and sum
    global _cg
    global _cgr
    global _G
    global _dim
    global _index_to_name
    global _P
    global _constraints
    global _learning_rate
    global _theta
    global _thetar

    H = copy.deepcopy(G)
    randomly_ordered_nodes = random.sample(unknowns, len(unknowns))

    #set all unknown pmfs to all 1's minus constraints
    #This is to sum over all possible combinations of y at once
    everything = np.ones( (_dim, 1), dtype=np.float128 ) * _constraints
    everything /= _dim 
    for node in G.nodes():
        if G.node[ node ]['text_func']:
            H.node[ node ]['node_potential'] = copy.deepcopy( everything )

    _G = H

    pool = ThreadPool(64)
    res = pool.map(update_single_node_sum_product, randomly_ordered_nodes)

    #partition_func = functools.reduce(lambda x, y: np.multiply( x[0] , y[0] ), res)
    partition_func = functools.reduce(lambda x, y: x[0] + y[0], res)
    return partition_func

def perform_loopy_belief_batch_iteration_mats(G, cg, cgr, unknowns, index_to_name, constraints, P, theta, thetar, factor_cgr):
    """
    Perform batch iteration of loopy belief propagation. 
    Calculate weighting updates for all network before updating them
    """
    N = len(index_to_name)
    H = copy.deepcopy(G)
    randomly_ordered_nodes = random.sample(unknowns, len(unknowns))
    #learning_rate = 0.05
    learning_rate = 1.0 / math.e
    total_error = 0.0

    global _cg
    global _cgr
    global _G
    global _dim
    global _index_to_name
    global _P
    global _constraints
    global _learning_rate
    global _theta 
    global _thetar
    global _factor_cgr

    _cg = cg
    _cgr = cgr
    _G = copy.deepcopy(G)
    _dim = dim
    _index_to_name = index_to_name
    _P = P
    _constraints = constraints
    _learning_rate = learning_rate
    _theta = theta
    _thetar = thetar
    _factor_cgr = factor_cgr

    #NB: must be called after setting global vars above
    #normalising_constant = compute_z_x(G, unknowns)

    #for node in randomly_ordered_nodes:
    #    print(node)
    #    update_single_node_sum_product(node)
    #sys.exit()

    pool = ThreadPool(64)
    #res = pool.map(update_single_node_max_product, randomly_ordered_nodes)
    res = pool.map(update_single_node_sum_product, randomly_ordered_nodes)

    P.logger.info("Parallel compute complete...")

    #normalising_pmf = normalising_constant
    normalising_pmf = scipy.zeros( (_dim, 1), dtype=scipy.float128 )
    for i, e in tqdm(res):
        normalising_pmf += i
    normalising_pmf /= float(len(randomly_ordered_nodes))

    #normalising_pmf = classes.utils.load_py_obj(P.config, "z_x")
    #IPython.embed()

    for i in range(len(res)):
        node = randomly_ordered_nodes[i]
        updated_node_pmf, error = res[i]

        total_error += scipy.sum( scipy.absolute( error ) )

        updated_node_pmf = scipy.multiply( updated_node_pmf,  (1.0/normalising_pmf) )
        H.nodes[node]['node_potential'] = _P.normalise_numpy_density( updated_node_pmf )

    P.logger.info("Total error: {}".format(total_error))

    pool.close()
    del res
    gc.collect()

    return H, total_error




def estimate_theta_stochastic_gradient_descent(G, cg, cgr, theta, thetar):
    """
        Train model to learn theta using SGD

    """
    pass



















def confidence(node_prob):
    #node name, probs array
    #probs arr = [ node ind, prob ] * 5
    highest = node_prob[0][1]
    second  = node_prob[1][1]

    if float(highest) < 0.01:
        return 0.0

    #print(node)
    #print(probs)
    if float(highest) == 0.0:
        return float('-inf')
    if float(second) == 0.0:
        return float('inf')
    ratio = highest / second
    return ratio

def most_confident_nodes(pmfs,min_ratio=2.0):
    mpmfs = copy.deepcopy(pmfs)    
    c_nodes = []
    while True:
        node, new_node_index = find_most_confident_node(mpmfs, min_ratio=100.0)
        if not node:
            break
        c_nodes.append(node)
        del mpmfs[node]
    return c_nodes

def find_most_confident_node(pmfs, min_ratio=1.20):
    mp = most_probable_symbols(pmfs, n=2)
    #print(mp)
    rd = {}

    for name, prob in mp.items():
        rd[name] = confidence( prob )
    #print(rd)
    max_r = max(rd.values())
    #print(max_r)
    #print(ratios)
    logger.info("Highest ratio: {}".format(max_r))

    #min ratio cut off
    if max_r < min_ratio:
        return False, False

    #import IPython
    #IPython.embed()

    highest_ratio_nodes = list( filter( lambda x: rd[x] == max_r, rd.keys() ) ) 
    #print(highest_ratio_nodes)

    hrn = dict( map( lambda x: [ x , mp[x][0][1] ], highest_ratio_nodes ) )
    max_ratio_max_value = max(hrn)
    #print(hrn)
    #print(max_ratio_max_value)

    return max_ratio_max_value, mp[ max_ratio_max_value ][0][0]

def node_to_inferred_name(G, node, index_to_name):
    node_potential = G.nodes[ node ]['node_potential']
    n_ind = np.argmax(node_potential)
    #r, c = np.where(node_potential == 1.0)
    #assert(le(r) <= 1)
    #assert(len(c) <= 1)
    #if len(r) > 0:
    #   return index_to_name[ r[0] ]
    #return "unknown"
    return index_to_name[ n_ind ]

def save_new_binary(orig_path, new_path, symbols):
    logger.info("Saving binary to {}".format( new_path ) )
    b = BinaryModifier(orig_path)
    b.add_symbols( new_symbols )
    b.save( new_path )

def save_inferred_binary(orig_path, new_path, G, symbols, index_to_name):
    new_symbols = []
    for node in G.nodes:
        symbs = list( filter( lambda x: x.name == node, symbols) )
        print(symbs)
        assert(len(symbs) == 1)
        symbol = symbs[0]

        new_name = node_to_inferred_name(G, node, index_to_name)
        symbol.name = new_name
        new_symbols.append( symbols )

    b = BinaryModifier(orig_path)
    b.add_symbols( new_symbols )
    b.save( new_path )

def save_inferred_graph(G, fname, index_to_name):
    #H = copy.deepcopy(G)
    unknowns = list(filter(lambda x: G.nodes[x]['func'], G.nodes()))
    guesses = list(map(lambda x: node_to_inferred_name(G, x, index_to_name), unknowns))
    mapping = dict(zip(unknowns, guesses))
    H = nx.relabel_nodes(G, mapping)

    """
    H = nx.DiGraph()
    for es, ee in G.edges:
        H.add_edge( node_to_inferred_name(G, es, index_to_name), node_to_inferred_name(G, ee, index_to_name) )
    """

    write_dot(H, fname + ".dot")
    return H


def calculate_node_pmf(G, cgs, name_to_index, node):
    assert(False)
    pmf = np.zeros([N, 1], dtype=np.float)
    n = 1
    for ffi, ffr,  in cgs:
        #find ffn and ffrn
        nth_callers = set( nth_parents(G, node, n) )
        nth_callees = set( nth_children(G, node, n) )

        fi_pmf = sum_feature_function(G, ffi, nth_callees, name_to_index)
        ri_pmf = sum_feature_function(G, ffr, nth_callers, name_to_index)

        #logger.info("Connectivity of node: {}".format( len(nth_callers) + len(nth_callees) ))
        pmf = np.add( pmf, fi_pmf )
        pmf = np.add( pmf, ri_pmf )
        n+=1

    pmf = np.multiply(pmf, constraints)
    return pmf


def update_crf_node_pmfs(G, pmfs, unknowns):
    for node in unknowns:
        old_node_potential = G.nodes[ node ]['node_potential']
        new_pmf = gradient_descent_pmf(old_node_potential, pmfs[node])

        #np.sum( np.subtract( old_node_potential, new_pmf ) )
        attr = { node : { 'node_potential' : new_pmf } }
        nx.set_node_attributes(G, attr)
    return G

def total_diff_pmfs(pmfs, m_pmfs, unknowns):
    if not pmfs:
        return float('inf')

    difference = 0.0
    for node in unknowns:
        diff = np.subtract( np.absolute( pmfs[node] ), np.absolute(m_pmfs[node]) )
        difference += np.sum( np.absolute(diff) )

    return difference

def infer_crf_loopy_belief_propagation(G, unknowns, cg, cgr, name_to_index, index_to_name, constraints, db, theta, thetar, factor_cgr):
    """
        Infer CRF with loopy belief propagation
        Use a greedy algorithm and start with N most confident nodes. 
        Then iterate passing messages until stable
    """
    assert(len(unknowns) >= 0)
    if len(unknowns) == 0:
        return G

    old_error = float('inf')
    local_maxima_it = 0
    nlp = classes.NLP.NLP(db.config)
    P = classes.pmfs.PMF(db.config)


    #capped at 50 iterations
    #for i in range(25):
    for i in range(25):  #3 , 2 + final loop
        logger.info("infer_crf_loopy_belief_propagation :: Starting epoch {}...".format(i))
        #H, error = perform_loopy_belief_batch_iteration(G, rels, unknowns, index_to_name, constraints, P)
        H, error = perform_loopy_belief_batch_iteration_mats(G, cg, cgr, unknowns, index_to_name, constraints, P, theta, thetar, factor_cgr)

        #"""
        n = 5
        correct, incorrect, avg_correct_index = check_graph_unknowns_top_n(nlp, H, index_to_name, unknowns, n)
        #correct, incorrect = check_graph_unknowns(nlp, G, index_to_name, unknowns)
        logger.info("Correct nodes: {}, Incorrect nodes: {}, Total unknown: {}, Total known: {}".format(correct, incorrect, len(unknowns), len(H.nodes) - len(unknowns)))
        logger.info("Average correct index: {}".format( avg_correct_index ))
        #"""

        if error >= old_error and i >= 2:
            break

        old_error = error
        G = H

    return G

def check_graph_unknowns(nlp, G, index_to_name, unknowns):
    correct = 0
    for node in unknowns:
        node_potential = G.nodes[ node ]['node_potential']
        max_ind = np.argmax( node_potential )

        if nlp.check_word_similarity(node, index_to_name[ max_ind ]) > 0.0:
            #print("{}   ==  {}".format(node, index_to_name[max_ind]))
            correct += 1
        else:
            #print("{}   !=  {}".format(node, index_to_name[max_ind]))
            pass

    return correct, len(unknowns) - correct

def check_graph_unknowns_top_n(nlp, G, index_to_name, unknowns, n):
    dim = len(name_to_index)
    correct = 0
    correct_name_indexes = []
    for node in unknowns:
        node_potential = G.nodes[ node ]['node_potential']
        #max_ind = np.argmax( node_potential )
        sorted_inds = np.argsort(node_potential, axis=0)
        top_inds = [a for y in sorted_inds[-n:].tolist() for a in y][::-1]
        #top_inds = [a for y in scipy.argsort(node_potential, axis=0).tolist() for a in y][::-1]


        nlp.logger.debug("============================")
        nlp.logger.debug("UNKNOWN: {}".format(node))
        c = 0
        for i in top_inds:
            nlp.logger.debug("\t{}{} guess: {}, p(y|X): {}".format(c, classes.utils.ordinal_follower(c), index_to_name[ i ] , node_potential[i, 0]))

            if nlp.check_word_similarity(node, index_to_name[ i ]) > 0.0:
                #print("{}   ==  {}".format(node, index_to_name[max_ind]))
                correct += 1
                break
            else:
                #print("{}   !=  {}".format(node, index_to_name[max_ind]))
                pass

            c += 1

        #correct_index = top_inds.index( name_to_index[ node ] )
        if node in name_to_index:
            correct_index = (dim-1) - np.where( sorted_inds == name_to_index[ node ] )[0][0]
            correct_name_indexes.append( correct_index )

            nlp.logger.debug("\tThe correct name was at rank - {} with p(y|X): {}".format(correct_index, node_potential[ name_to_index[node], 0 ]))
        else:
            nlp.logger.debug("\tThe correct name was not previously seen")
            nlp.logger.debug("============================")

    if len(correct_name_indexes) == 0:
        avg_correct_index = -1.0
    else:
        avg_correct_index = sum(correct_name_indexes) / float( len( correct_name_indexes ) )
    return correct, len(unknowns) - correct, avg_correct_index



def check_graph_unknowns_stripped(db, G, index_to_name, unknowns):
    correct = 0
    for node in unknowns:
        node_potential = G.nodes[ node ]['node_potential']
        max_ind = np.argmax( node_potential )
        #print(node)
        #print(max_ind)
        #print(index_to_name[ max_ind ] )
        global bin_path
        global optimisation
        global linkage
        global bin_name
        global compiler
        vaddr = int(node[4:], 16)
        known_bin_path = bin_path.replace("bin-stripped", "bin")

        #print("{}:{}".format(node, vaddr))
        #print("I think name:\"{}\", vaddr:{}, path:\"{}\", optimisation:\"{}\", linkage:\"{}\", compiler:\"{}\", bin_name:\"{}\" ".format( index_to_name[ max_ind ], vaddr, known_bin_path, optimisation, linkage, compiler, bin_name))
        s = classes.symbol.Symbol(path=known_bin_path, linkage=linkage, optimisation=linkage, name=node, bin_name=bin_name, vaddr=vaddr, size=1, compiler=compiler)
        #print(s)
        #import IPython
        #IPython.embed()

        if scripts.perform_analysis.check_inferred_symbol_name(db, s, index_to_name[ max_ind ]):
            correct += 1

    return correct, len(unknowns) - correct




def check_graph_correct_stripped(db, G, index_to_name):
    #check node labels == name index of pmfs for each label
    #check with nlp function from analysis script
    correct, incorrect = 0, 0

    for node in G.nodes:
        node_potential = G.nodes[ node ]['node_potential']
        r, c = np.where(node_potential == 1.0)
        assert(len(r) <= 1)
        assert(len(c) <= 1)
        if len(r) > 0:
            print(node)
            if node[:4] != "fcn.":
                continue
            print(node)
            vaddr = int(node[4:], 16)
            print(vaddr)

            global bin_path
            global linkage
            global compiler
            global bin_name
            global optimisation




            s = classes.symbol.Symbol(path=bin_path, linkage=linkage, optimisation=optimisation, name=node, bin_name=bin_name, vaddr=vaddr)
            if scripts.perform_analysis.check_inferred_symbol_name(db, s, index_to_name[ r[0] ]):
                print("correct")
                correct += 1
            else:
                incorrect += 1
                print("incorrect")

    return correct, incorrect

def expand_lil_sparse_reachablility(cgs, knowns_in_bin, name_to_index, index_to_name):
    """
        Return a dense matrix with relationships between knowns in bin and others when given an aray of cg relationships as input
    """
    #calculate size of "new" matrix
    reachable = set(knowns_in_bin)
    for known in knowns_in_bin:
        for cg in cgs:
            ind = name_to_index[ known ]
            lol = cg[ind, :]
            for r, v in zip(lol.rows[0], lol.data[0]):
                t = index_to_name[ r ] 
                reachable.add( index_to_name[ r ] )

    print("Total reachable {}".format( len(reachable) ))


    uname_to_index = {}
    uindex_to_name = {}
    n = 0
    for s in reachable:
        uname_to_index[ s ] = n
        uindex_to_name[ n ] = s
        n += 1

    ucgs = []
    for cg in cgs:
        N = len(reachable)
        ucg = np.zeros((N, N), dtype=np.float128)

        for known in knowns_in_bin:
            ind = name_to_index[ known ]
            lol = cg[ind, :]
            uknown_ind = uname_to_index[ index_to_name[ r ] ]
            for r, v in zip(lol.rows[0], lol.data[0]):
                 uind = uname_to_index[ index_to_name[ r ] ]
                 ucg[ uknown_ind, uind ] = v

        ucgs.append(ucg)

    return ucgs, uname_to_index, uindex_to_name


def check_graph_correct(nlp, G, index_to_name):
    #check node labels == name index of pmfs for each label
    #check with nlp function from analysis script
    correct, incorrect, unknown = 0, 0, 0
    for node in G.nodes:
        node_potential = G.nodes[ node ]['node_potential']
        r, c = np.where(node_potential == 1.0)
        assert(len(r) <= 1)
        assert(len(c) <= 1)
        if len(r) > 0:
            #if scripts.perform_analysis.check_similarity_of_symbol_name(node, index_to_name[ r[0] ]):
            if nlp.check_word_similarity(node, index_to_name[ r[0] ]) > 0.0:
                correct += 1
            else:
                incorrect += 1
        else:
            unknown += 1

    return correct, incorrect, unknown

def load_knowns(path):
    """
    Load knowns from a line separated text file
    :param path: text file path
    :return: set of knowns
    :rtype: set
    """
    knowns = set([])
    with open(path, 'r') as f:
        for line in f:
            knowns.add(line.strip())
    return knowns

def load_relationships(db, G):
    """
    Load relationships for nodes in G g
    :param db: Database instance
    :param G: Graph of nodes and relationships
    :return: A set of relationships
    :rtype: set
    """
    rels = {}
    for node in G.nodes():
        for t in [ "caller", "callee" ]:
            r = db.client['xrefs_pmfs'].find_one({ "name" : node, "type": t })
            if not r:
                continue
            pmf = classes.pmfs.PMF.bytes_to_scipy_sparse(r['pmf'])
            rels[node + '_' + t] = pmf
    db.config.logger.info("Relationships loaded in memory - {} B".format( sys.getsizeof(rels) ))
    return rels

def load_all_relationships(db):
    """
    Load all relationships between training set symbols into memory and unpacked scipy sparse matrices
    :param: Database instance
    :return: relationships dictionary
    :type: dict
    """
    rels = {}
    res = db.client['xrefs_pmfs'].find()

    packed = []
    meta = []
    for r in tqdm(res):
        packed.append( r['pmf'] )
        meta.append( ( r['name'], r['type'] ) )

    pool = multiprocess.Pool(processes=int(64*1.5))
    #pool = ThreadPool(int(64*1.5))
    db.config.logger.info("Starting multithreaded unpacking of XREFS")
    r = pool.map( classes.pmfs.PMF.bytes_to_scipy_sparse, packed)
    db.config.logger.info("Done! Building memory index...")
    for i in range(len(r)):
        rels[meta[i][0] + '_' + meta[i][1] ] = r[i]

    db.config.logger.info("Relationships loaded in memory - {} B".format( sys.getsizeof(rels) ))
    return rels


def get_dynamic_libs(db, bin_path, resolved=set([])):
    """
        Calculate missing shared libraries binary needs to link at run time.
        Recursivly looks for missing shared libraries.
        :param db: Database instance
        :param bin_path: The path of binary to resolve
        :param resolved: NOT USED. Pre-resolved libraries
        :return: The missing library names
        :rtype: set
    """

    libs_missing = set([])
    text_libs = subprocess.check_output('readelf --dynamic {} 2>/dev/null | grep "Shared library:" | cut -f 1 -d "]" | cut -f 2 -d "["'.format(bin_path), shell=True).decode('ascii')
    libs = text_libs.split('\n')
    for lib in libs:
        if len(lib) == 0:
            continue

        m = re.match(r'^.*\.so', lib)
        if not m:
            raise Exception("Error matching dynamic librarys to regex")

        if m.group(0) in resolved:
            continue

        pattern = "^{}.*\.so.*".format( m.group(0)[:-3] )
        s = db.find_binary_like( pattern )
        if not s:
            libs_missing.add( m.group(0) )
        else:
            resolved.add( m.group(0) )
            libs_missing = libs_missing.union( get_dynamic_libs(db, s['path'], resolved=resolved ) )
    return libs_missing.union(resolved)




def missing_libs(db, bin_path, resolved=set([])):
    """
        Calculate missing shared libraries binary needs to link at run time.
        Recursivly looks for missing shared libraries.
        :param db: Database instance
        :param bin_path: The path of binary to resolve
        :param resolved: NOT USED. Pre-resolved libraries
        :return: The missing library names
        :rtype: set
    """

    libs_missing = set([])
    text_libs = subprocess.check_output('readelf --dynamic {} 2>/dev/null | grep "Shared library:" | cut -f 1 -d "]" | cut -f 2 -d "["'.format(bin_path), shell=True).decode('ascii')
    libs = text_libs.split('\n')
    for lib in libs:
        if len(lib) == 0:
            continue

        m = re.match(r'^.*\.so', lib)
        if not m:
            raise Exception("Error matching dynamic librarys to regex")

        if m.group(0) in resolved:
            continue

        pattern = "^{}.*\.so.*".format( m.group(0)[:-3] )
        s = db.find_binary_like( pattern )
        if not s:
            libs_missing.add( m.group(0) )
        else:
            resolved.add( m.group(0) )
            libs_missing = libs_missing.union( missing_libs(db, s['path'], resolved=resolved ) )
    return libs_missing



def build_experiment_knowns_constraints(GLOBAL_KNOWNS, GLOBAL_UNKNOWNS, name_to_index, knowns_in_bin, unknowns_in_bin):
    """
        Build constraint vector of possible output symbol names!
        :param db: Database instance
        :param name_to_index: mapping of symbol names to indexes
        :param knowns_in_bin: List of symbol names that are knowns (dyn imports)
        :return: Vector of [{1,0}] for eligable symbols
        :rtype: numpy.matrix
        :warn: Assumes a known function in the binary is not repeated!
    """
    N = len(name_to_index)

    knowns = GLOBAL_KNOWNS
    unknowns = GLOBAL_UNKNOWNS

    #a known might also be an unknown, so take knowns as those only in knowns set
    only_knowns = knowns.difference( unknowns )

    #add knowns in the current binary
    experiment_knowns = only_knowns.union( knowns_in_bin )

    #import IPython
    #IPython.embed()

    for s in unknowns_in_bin:
        if s in experiment_knowns:
            raise Exception("Error unknown symbol {} is in experiment_knowns constraints".format(s))
        #if s not in unknowns:
        #    raise Exception("Error unknown symbol {} is not in unknowns".format(s))

    for s in knowns_in_bin:
        if s not in experiment_knowns:
            raise Exception("Error known symbol {} is not in experiment_knowns constraints".format(s))
        #if s in unknowns:
        #    raise Exception("Error known symbol {} is in unknowns".format(s))


    #any unknown is definitely not in experiment_knowns
    #all oness -> any symbols are possible
    constraints = np.ones((N,1), dtype=np.float128)
    for symb in list( experiment_knowns ): 
        if symb not in name_to_index:
            #db.config.logger.warn("{} is a known that doesn't have an index".format(symb))
            continue
        g_index = name_to_index[ symb ]
        constraints[ g_index ] = 0.0

    return constraints, experiment_knowns

def build_knowns_constraints(db, name_to_index, knowns_in_bin, unknowns_in_bin):
    """
        Build constraint vector of possible output symbol names!
        :param db: Database instance
        :param name_to_index: mapping of symbol names to indexes
        :param knowns_in_bin: List of symbol names that are knowns (dyn imports)
        :return: Vector of [{1,0}] for eligable symbols
        :rtype: numpy.matrix
        :warn: Assumes a known function in the binary is not repeated!
    """
    N = len(name_to_index)

    USED_CACHED_KNOWNS_UNKNOWNS = True
    if not USED_CACHED_KNOWNS_UNKNOWNS:
        #get al knowns and unknowns in database
        knowns = db.get_known_symbol_names()
        unknowns = db.get_unknown_symbol_names()
        classes.utils.save_py_obj(db.config, knowns, "knowns")
        classes.utils.save_py_obj(db.config, unknowns, "unknowns")
    else:
        knowns = classes.utils.load_py_obj(db.config, "knowns")
        unknowns = classes.utils.load_py_obj(db.config, "unknowns")

    #a known might also be an unknown, so take knowns as those only in knowns set
    only_knowns = knowns.difference( unknowns )

    #add knowns in the current binary
    experiment_knowns = only_knowns.union( knowns_in_bin )

    #import IPython
    #IPython.embed()

    for s in unknowns_in_bin:
        if s in experiment_knowns:
            raise Exception("Error unknown symbol {} is in experiment_knowns constraints".format(s))
        #if s not in unknowns:
        #    raise Exception("Error unknown symbol {} is not in unknowns".format(s))

    for s in knowns_in_bin:
        if s not in experiment_knowns:
            raise Exception("Error known symbol {} is not in experiment_knowns constraints".format(s))
        #if s in unknowns:
        #    raise Exception("Error known symbol {} is in unknowns".format(s))


    #any unknown is definitely not in experiment_knowns
    #all oness -> any symbols are possible
    constraints = np.ones((N,1), dtype=np.float128)
    for symb in list( experiment_knowns ): 
        if symb not in name_to_index:
            #db.config.logger.warn("{} is a known that doesn't have an index".format(symb))
            continue
        g_index = name_to_index[ symb ]
        constraints[ g_index ] = 0.0

    return constraints, experiment_knowns


def filter_disconnected_nodes(G, logger):
    """
        Look for nodes with no edges and remove them
        :param G: Graph
        :return: New graph and array of nodes removed
    """
    disconnected_nodes = []
    #calculate number of disconnected nodes
    for node in G.nodes():
        #remove node with null name
        if node == '':
            disconnected_nodes.append(node)

        deg = G.degree(node)
        if deg == 0:
            disconnected_nodes.append(node)

    logger.info("{} disconnected nodes in callgraph out of {} total nodes".format( len(disconnected_nodes), len(G.nodes()) ))
    for node in set(disconnected_nodes):
        G.remove_node(node)

    if len(G.nodes()) == 0:
        logger.error("No connected nodes in callgraph!")
        sys.exit()

    return G, disconnected_nodes


def load_mat_rels(db, index_to_name):
    """
        Load relationships from database and convert into scipy.matix
        :return: cg - matrix where the i'th row represents the probability of what i calls
        :return: cgr - matrix where the i'th row represents the probability of what i is called by 
    """
    USED_CACHED_RELATIONSHIPS = True

    logger = db.config.logger
    if not USED_CACHED_RELATIONSHIPS:
        #rels = load_relationships(db, G)
        rels = load_all_relationships(db)
        #classes.utils.save_py_obj(config, rels, "all_db_relationships")
        #rels = classes.utils.load_py_obj(config, "all_db_relationships")

        logger.info("Creating callgraph relation matrices")
        pmfs_f, pmfs_r = [], []
        for i in tqdm( range(len(index_to_name.keys())) ):
            #rels are stored as sparse csr matrices
            #cg operates as a per row, need to transpose
            pmfs_f.append(  rels[ index_to_name[i] + "_callees" ] )
            pmfs_r.append( rels[ index_to_name[i] + "_callers" ] )

        logger.info("Stacking pmfs for cg")
        cg = scipy.sparse.vstack( pmfs_f )
        logger.info("Stacking pmfs for cgr")
        cgr = scipy.sparse.vstack( pmfs_r )

        classes.utils.save_py_obj(config, cg, "cg")
        classes.utils.save_py_obj(config, cgr, "cgr")
    else:
        cg  = classes.utils.load_py_obj(config, "cg")
        cgr = classes.utils.load_py_obj(config, "cgr")

    return cg, cgr

def calculate_unseen_relationships(G, cg, cgr, logger):
    """
        Calculate the number of relationships that we have/have not seen before
        :return: seen edges of cg, unseen edges of cg and then nodes with no known rels in cg
    """
    seen_rels = set([])
    unseen_rels = set([])

    for u, v in G.edges():
        #ignore data relationships
        if G[u][v]['data_ref']:
            continue

        if u not in name_to_index or v not in name_to_index:
            unseen_rels.add( (u, v) )
            continue

        u_ind, v_ind = name_to_index[u], name_to_index[v]
        if cg[u_ind, v_ind] > 0.0 or cgr[v_ind, u_ind] > 0.0:
            seen_rels.add( (u, v) )
        else:
            unseen_rels.add( (u, v) )

    logger.info("{} relationship{} seen in training set. {} relationship{} never seen before.".format( len(seen_rels), classes.utils.plural_follower(len(seen_rels)),  len(unseen_rels), classes.utils.plural_follower( len(unseen_rels) ) ))

    nodes_seen, nodes_all = set([]), set(filter(lambda x: G.node[x]['func'], G.nodes))
    for u, v in seen_rels:
        nodes_seen.add(u)
        nodes_seen.add(v)

    nodes_with_no_known_relationships = nodes_all.difference(nodes_seen)
    logger.info("{} node{} with no known relationships seen in training set i.e. they have relationships but we havent seen any of them before.".format( len(nodes_with_no_known_relationships), classes.utils.plural_follower(len(nodes_with_no_known_relationships)) ))

    return seen_rels, unseen_rels, nodes_with_no_known_relationships


if __name__ == '__main__':
    import IPython
    np.set_printoptions(edgeitems=15)
    global P
    config = Config()
    logger = config.logger
    logger.setLevel(logging.DEBUG)
    nlp = classes.NLP.NLP(config)
    db = classes.database.Database(config)
    P = classes.pmfs.PMF(config)

    logger.info("Connecting to mongod...")
    db = Database(config)

    logger.info("Loading resources...")
    ############
    #### Load res
    ############

    symbol_names = classes.utils.load_py_obj( config, "symbol_names")
    name_to_index = classes.utils.load_py_obj( config, "name_to_index")
    index_to_name = classes.utils.load_py_obj( config, "index_to_name")

    #unknown_bins = db.get_unknown_symbol_binaries() 
    #classes.utils.save_py_obj( config, unknown_bins, "unknown_bins")
    #unknown_bins =  classes.utils.load_py_obj( config, "unknown_bins_with_xrefs")
    unknown_bins =  classes.utils.load_py_obj( config, "testing_bins")
    training_bins = classes.utils.load_py_obj( config, "training_bins")

    logger.info("Loading relationships")
    #cg, cgr = load_mat_rels(db, index_to_name)

    ## cgr does not contain calls from knowns!!
    ## ignore cgr, use cg trasposed!

    USE_DENORMALISED_RELS = True
    USE_CACHED_DENORMALISED_RELS = True

    if USE_DENORMALISED_RELS:
        logger.info("Denormalising cg relationships")
        if not USE_CACHED_DENORMALISED_RELS:
            #call_matrix[ X, Y ] now counts how many times X calls Y
            call_matrix = cg + scipy.transpose(cgr)
            denormalised_cg  = classes.utils.denormalise_scipy_sparse(call_matrix)
            logger.info("Saving denormalised matrices")
            classes.utils.save_py_obj( config, denormalised_cg, "denormalised_cg" )
        else:
            denormalised_cg = classes.utils.load_py_obj( config, "denormalised_cg" )

        cg = denormalised_cg

        cgr = scipy.transpose( cg )
        cgr = cgr.tocsr(copy=True)

    USED_CACHED_THETAS = True
    if not USED_CACHED_THETAS:

        assert(False)
        logger.info("Creating thetas")
        theta = cg.tolil(copy=True)

        for r, c in tqdm(list(zip(theta.nonzero()[0], theta.nonzero()[1]))):
            theta[r, c] = 1.0

        ##too low to transpose lareg matrix every use
        theta = theta.tocsr(copy=True)
        classes.utils.save_py_obj( config, theta, "theta" )
    else:
        theta = classes.utils.load_py_obj( config, "theta_opt" )

    cg = classes.utils.load_py_obj( config, "denormalised_cg")
    cgr = scipy.transpose(cg).tocsr(copy=True)

    theta = theta.tocsr(copy=True)
    thetar = scipy.transpose( theta )
    thetar = thetar.tocsr(copy=True)


    BUILD_PROBS_THETAS = False
    if BUILD_PROBS_THETAS:
        for r, c in tqdm(list(zip(theta.nonzero()[0], theta.nonzero()[1]))):
            uniqueness = len(thetar[c, :].nonzero()[0])
            if uniqueness == 0:
                theta[r, c] = 0.0
                continue
            theta[r, c] = theta[r, c] / float(uniqueness)
            
        thetar = scipy.transpose(theta).tocsr(copy=True)
    else:
        #theta = classes.utils.load_py_obj( config, "theta_trained" )
        #thetar = classes.utils.load_py_obj( config, "thetar_trained" )
        pass

    USED_CACHED_KNOWNS_UNKNOWNS = True
    if not USED_CACHED_KNOWNS_UNKNOWNS:
        #get al knowns and unknowns in database
        GLOBAL_KNOWNS = db.get_known_symbol_names()
        GLOBAL_UNKNOWNS = db.get_unknown_symbol_names()
        classes.utils.save_py_obj(db.config, GLOBAL_KNOWNS, "knowns")
        classes.utils.save_py_obj(db.config, GLOBAL_UNKNOWNS, "unknowns")
    else:
        GLOBAL_KNOWNS = classes.utils.load_py_obj(db.config, "knowns")
        GLOBAL_UNKNOWNS = classes.utils.load_py_obj(db.config, "unknowns")


    factor_cgr = classes.utils.load_py_obj(db.config, "ff_rfactors")




    print("Total testing binaries:", str(len(unknown_bins)))
    #for path in unknown_bins:
    for ZZZZZ in [ 'a' ]:
        #path = sys.argv[1]
        path = "/root/friendly-corpus/debian/heartbeat/usr/lib/heartbeat/heartbeat"
        #path = "/root/friendly-corpus/debian/ion/usr/bin/tcpcli"

        #path = list(unknown_bins)[35]
        #path = "/root/friendly-corpus/debian/numactl/usr/bin/numactl"
        #path ="/root/friendly-corpus/debian/xmms2-client-cli/usr/bin/xmms2"
        #path = "/root/friendly-corpus/debian/coop-computing-tools/usr/bin/catalog_query"
        #path = "/root/friendly-corpus/debian/util-vserver/usr/sbin/naddress"
        #path = "/root/friendly-corpus/debian/mate-power-manager/usr/lib/mate-power-manager/mate-brightness-applet"

        logger.info("Using binary {}".format(path))
        if classes.binary.Binary.linkage_type(path) != 'dynamic':
            logger.error("{} is a static binary, skipping...")
            continue

        bin_name = os.path.basename(path)

        missing = missing_libs(db, path)
        if len(missing) == 0:
            logger.info("All dynamically linked libraries are in the database!")
        else:
            logger.warn("Missing linked libraries - {}".format(missing))


        logger.info("Dynamic libs - {}".format(get_dynamic_libs(db, path)))

        orig_G = classes.callgraph.build_crf_for_binary(db, path, name_to_index)
        G, disconnected_nodes = filter_disconnected_nodes(orig_G, logger)


        #IPython.embed()
        logger.info("Getting knowns")

        #raw_known_symbols = load_knowns(config.desyl + "/libs/libcoreutils.symbs").union( load_knowns(config.desyl + "/libs/libc.symbs") )
        #raw_known_symbols =  load_knowns(config.desyl + "/libs/libc.symbs").union( db.get_known_symbol_names() )
        #known_symbols = list(map(lambda x: nlp.strip_library_decorations( nlp.strip_ida_decorations( x ) ), raw_known_symbols ) )

        #classes.utils.save_py_obj( config, known_symbols, "known_symbols")
        #known_symbols = classes.utils.load_py_obj( config, "known_symbols")

        #logger.info("Expanding feature function matracies")
        #cg1 = classes.utils.load_py_obj(config, "1_degree_cg")
        #cgr1 = classes.utils.load_py_obj(config, "1_degree_cgr")

        #knowns_in_bin = set(G.nodes()).intersection(known_symbols)

        #knowns are all xrefs that are not in .text section of code
        #knowns_in_bin = db.get_set_all_xrefs({ "$match" : { "path" : path } }).difference( set(G.nodes()) ) 
        knowns_in_bin = set(filter(lambda x: G.nodes[x]['func'] and not G.nodes[x]['text_func'], G.nodes()))
        unknowns_in_bin = set(filter(lambda x: G.nodes[x]['func'] and G.nodes[x]['text_func'], G.nodes()))
        calculated_knowns = crf.utils.get_calculated_knowns(nlp, path)

        MAX_UNKNOWNS = 1000
        if len(unknowns_in_bin) > MAX_UNKNOWNS:
            logger.info("{} has more than {} unknowns, skipping".format(path, MAX_UNKNOWNS))
            continue

        print(calculated_knowns)

        ##restrict calculated knows to dynsym entires that correspond to entires in the .text section of binary
        for name in list(calculated_knowns):
            if name not in unknowns_in_bin:
                calculated_knowns.remove(name)

        for name in calculated_knowns:
            knowns_in_bin.add(name)
            unknowns_in_bin.remove(name)

        print("Found {} unknowns in the dynamic symbol table.".format(calculated_knowns))

        #ucgs, uname_to_index, uindex_to_name = expand_lil_sparse_reachablility([cg1, cgr1], knowns_in_bin, name_to_index, index_to_name)
        #name_to_index = uname_to_index
        #index_to_name = uindex_to_name
        #cg1 = ucgs[0]
        #cgr1 = ucgs[1]

        dim = len(name_to_index)

        logger.info("Resources loaded!")

        logger.info("Building constraints")
        #build index numpy constraints vector 
        #constraints, known_symbols = build_knowns_constraints(db, name_to_index, knowns_in_bin, unknowns_in_bin)
        constraints, known_symbols = build_experiment_knowns_constraints(GLOBAL_KNOWNS, GLOBAL_UNKNOWNS, name_to_index, knowns_in_bin, unknowns_in_bin)

        #add knowns/unknowns and pmfs to graph
        unknowns = []
        knowns = []
        remove = []
        for node in G.nodes:
            #skip non functions
            if not G.nodes[ node ]['func']:
                continue

            if node in known_symbols:
                knowns.append( node )
                pmf = np.zeros( (dim, 1), dtype=np.float128)
                if node not in name_to_index:
                    logger.error("Known symbol {} is not in the symbol name hashmap".format( node ) )
                    remove.append(node)
                    continue
                pmf[ name_to_index[ node ] ] = 1.0
                attr = { node : { 'node_potential' : pmf, 'known': True } }
                nx.set_node_attributes(G, attr)
            else:
                unknowns.append( node )
                ###assign initial probability from stage1
                pmf = np.ones( (dim, 1), dtype=np.float)
                pmf = np.multiply( pmf, constraints )
                #assign equal 1/possible symbs
                r, c = np.where( constraints > 0.0 ) 
                pmf = np.multiply( float(1) / len(r) , pmf )
                #pmf[ name_to_index[ node ] ] = 1.0
                attr = { node : { 'node_potential' : pmf, 'known': False } }
                nx.set_node_attributes(G, attr)

        for n in remove:
            G.remove_node(n)


        logger.info("Calculating maximum theoretically inferrable for this binary")
        knowns_set = set(knowns)

        #write_dot(G, '/tmp/graph.dot')

        theoretically_impossible = set(filter(lambda x: G.nodes[x]['func'] and x not in symbol_names, G.nodes()))
        #calculate maximum theoretical
        seen_rels, unseen_rels, impossible_to_infer = calculate_unseen_relationships(G, cg, cgr, logger)

        logger.info("Original binary has {} symbols, {} knowns, {} unknowns, and {} symbols we have never seen before".format( len(knowns+unknowns), len(knowns) - len(calculated_knowns), len(unknowns), len(theoretically_impossible) ))
        logger.info("{} knowns which we have never seen before".format( len(remove) ))

        logger.info("Maximum theoretically inferrable symbols in unknowns: {}".format( len(knowns+unknowns) - len(impossible_to_infer.difference(knowns_in_bin)) ))


        logger.info("Building CRF with {} knowns and {} unknowns. CRF has {} total nodes (inc data).".format(len(knowns), len(unknowns), len(G.nodes)))

        #crf.utils.calculate_unknown_permutations(G, [[cg, cgr]], name_to_index, index_to_name)
        #IPython.embed()
        try:
            #rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            #H = infer_crf_loopy_belief_propagation(G, unknowns, rels, name_to_index, index_to_name, constraints, db)
            H = infer_crf_loopy_belief_propagation(G, unknowns, cg, cgr, name_to_index, index_to_name, constraints, db, theta, thetar, factor_cgr)
            correct, incorrect, avg_correct_index = check_graph_unknowns_top_n(nlp, H, index_to_name, unknowns, 5)
            #correct, incorrect = check_graph_unknowns(nlp, H, index_to_name, unknowns)
            logger.info("Correct nodes: {}, incorrect nodes: {}, Total unknown: {}, Total known: {}".format(correct, incorrect, len(unknowns), len(knowns)))
            save_inferred_graph(H, config.res + '/inferred_graphs/' + path.replace("/","_"), index_to_name)
            logger.info("Finished loopy belief propagation")


            """
            print("\t{} known.".format( len( knowns ) ))
            print("\t{} unknown.".format( len( unknowns ) ))
            print("\t{} correct out of unknowns.".format( correct  ))
            print("\t{} total symbols from symtab.".format( len(knowns) + len(unknowns) ))
            """

            #tp = sum( x == True for x in conf_correct )
            tp = correct + len(calculated_knowns)
            tn = 0
            fp = incorrect
            fn = 0

            if tp == 0:
                logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
                logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( 0.0, 0.0, 0.0))
            else:
                precision   = tp / float(tp + fp)
                #recall      = tp / float(tp + fn)
                recall = precision
                f1 = 2.0 * (precision * recall)/(precision + recall)

                logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
                logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( precision, recall, f1))

            #save_new_binary(bin_path, config.res + '/unstripped_bins/' + os.path.basename( bin_name ), new_symbols)

        except KeyboardInterrupt:
            logger.info("Caught keyboard press")
            pass
        finally:
            logger.info("Finished loopy belief propagation")

        logger.info("Freeing memory")
        del G
        del H
        del orig_G
        del unknowns
        del knowns
        del remove
        del seen_rels
        del unseen_rels
        del impossible_to_infer
        del theoretically_impossible
        del known_symbols
        del constraints
        del knowns_in_bin
        del unknowns_in_bin
        del disconnected_nodes
        del missing

        gc.collect()
        continue

        logger.info("Infering CRF")
        pmfs = infer_crf(unknowns,G,rels, name_to_index, index_to_name, constraints)

        new_symbols = list(known_symbols)
        conf_symbols = 0
        while(len(unknowns) > 0):
            """
            if not stripped:
                correct, incorrect = check_graph_correct(G, index_to_name)
            else:
                #correct, incorrect = check_graph_correct_stripped(db, G, index_to_name, unknowns)
                correct, incorrect = check_graph_unknowns_stripped(db, G, index_to_name, unknowns)
            logger.info("Correct nodes: {}, incorrect nodes: {}".format(correct, incorrect))
            """
            #import IPython
            #IPython.embed()

            node, new_node_index = find_most_confident_node(pmfs, min_ratio=config.analysis.crf.MIN_CONF_RATIO)

            if node == False:
                #no more confident nodes, stop
                logger.info("No more confident nodes. Stopping")
                break

            conf_symbols += 1
            print("OLD NODE: {}".format(node))
            print("NEW NODE: {}".format(index_to_name[new_node_index]) )
            new_node_name = index_to_name[ new_node_index ]

            """
            #old_s = db.get_symbol(bin_path, node, collection_name=collection)
            old_s = db.get_symbol(unstripped_bin_path, node)
            new_s = db.get_symbol(unstripped_bin_path, index_to_name[new_node_index])
            assert(isinstance(old_s, classes.symbol.Symbol))
            if new_s:
                print(new_s)
                corr = scripts.perform_analysis.check_inferred_symbol( db, old_s, new_s )
                if corr:
                    corr_symbs += 1
                logger.debug("Is symbol correct: {} (in binary)".format(corr))
            else:
                logger.debug("New symbol is incorrect (not in binary)")

            old_s.name = new_node_name
            new_symbols.append(old_s)
            """


            """
            print("Old constraints for new node: {}".format( constraints[new_node, 0]))

            if constraints[ new_node, 0] != 1.0:
                assert(False)
            """

            #update constraints
            constraints[new_node_index, 0] = 0

            #remove from unknowns
            del unknowns[ unknowns.index( node ) ]

            #set new node potentials as node
            pmf = np.zeros( (dim, 1), dtype=np.float128)
            pmf[ new_node_index, 0 ] = 1.0
            attr = { node : { 'node_potential' : pmf } }
            nx.set_node_attributes(G, attr)

            del pmfs[node]
            #pmfs = update_crf(unknowns, G, cgs, name_to_index, index_to_name, constraints, pmfs, node)


        #correct, incorrect, unknowns = check_graph_correct(G, index_to_name)
        #correct, incorrect, unknowns = check_graph_correct(G, list(map(lambda x: x[0].name, unknown_symbs)))
        #correct, incorrect, unknowns = check_graph_correct(G, index_to_name, known_symbol_names)
        correct, incorrect, unknown = check_graph_correct_top_n(nlp, G, index_to_name, 5)
        """
        if not stripped:
            correct, incorrect = check_graph_correct(G, index_to_name)
        else:
            #correct, incorrect = check_graph_correct_stripped(db, G, index_to_name, unknowns)
            correct, incorrect = check_graph_unknowns_stripped(db, G, index_to_name, unknowns)
        """


        logger.info("///////////////Previous")
        print("\t{} known.".format( len( knowns ) ))
        print("\t{} unknown.".format( len( unknowns ) ))

        print("\t{} correct out of knowns.".format( len(knowns) ) )
        print("\t{} correct out of unknowns.".format( correct - len(knowns)  ))
        print("\t{} total symbols from symtab.".format( len(knowns) + len(unknowns) ))

        #tp = sum( x == True for x in conf_correct )
        tp = correct - len(knowns)
        tn = 0
        fp = incorrect
        fn = unknown

        if tp == 0:
            logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
            logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( 0.0, 0.0, 0.0))
        else:
            precision   = tp / float(tp + fp)
            recall      = tp / float(tp + fn)
            f1 = 2.0 * (precision * recall)/(precision + recall)

            logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
            logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( precision, recall, f1))

        #sys.exit()
        continue
        save_inferred_graph(G, "graph_inferred", index_to_name)
        save_new_binary(bin_path, "/tmp/desyl/" + os.path.basename( bin_name ), new_symbols)

