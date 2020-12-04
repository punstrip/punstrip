#!/usr/bin/python3
import os, sys
import logging, math, math, math
import numpy as np
import scipy, time
import collections
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import functools, itertools
import copy, re
import random
import networkx as nx
import glob
from networkx.drawing.nx_pydot import write_dot 
#from multiprocessing import Pool
from multiprocess import Pool
from multiprocess.pool import ThreadPool
import multiprocess
from tqdm import tqdm



import context
from classes.symbol import Symbol
from classes.database import Database
from classes.config import Config
from classes.counter import dCounter
from classes.bin_mod import BinaryModifier

import classes.utils
import scripts.perform_analysis

random.seed()

global GG
global n
global P

global bin_path

global linkage
global compiler
global bin_name
global optimisation


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


    #logger.info("orig ff_mat shape: {}".format(np.shape(ff_mat)))
    for node in nodes:
        #print("Node name: {}".format( node ) )
        node_potentials = G.nodes[ node ]['node_potentials']
        #print("Node potentials: {}".format( np.where( node_potentials > 0.0 ) ) )
        #logger.info("node_potentials shape: {}".format(np.shape(node_potentials)))
        #ind = name_to_index[node]

        #debug_np_obj( node_potentials, "node_potential")
        #debug_np_obj( ff_mat, "ff_mat")
        pmf = np.dot( ff_mat, node_potentials )
        #debug_np_obj( pmf, "pmf")
        #sys.exit()
        #print( "pmf ff: {}".format(np.where( pmf > 0.0 ) ) )


        #print( "symbol {}: {}".format( name_to_index[ node ], node ))

        #column = ff_mat[name_to_index[node], : ]
        #column_indexes = np.where( column > 0.0 )
        #print( "column_indexes: {}".format( column_indexes ))

        #rev_callees = ffi[:, 1959 ]


        #logger.info("pmf shape: {}".format(np.shape(pmf)))
        #m = np.shape(pmf)
        #logger.info("Shape of cg_mat[{}][:] is: {}".format( p_ind, m))
        pmfs.append( pmf )

    #cpf = functools.reduce(lambda x, y: rescale( x + np.transpose(y) ), pdfs)
    pmf = functools.reduce(lambda x, y: np.add(x, y), pmfs, np.zeros((m,1 ), dtype=np.float))
    #pmf = np.transpose(pmf)
    #print("sum feature functions final pmf:")
    #debug_np_obj( pmf, "pmf")
    #logger.info("reduced pmf shape: {}".format(np.shape(pmf)))
    #assert( np.shape(pmf) == m )
    return P.normalise_numpy_density( pmf )

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
    d = csr_matrix( (r, c), dtype=np.float64 )
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
    d = np.zeros( (r, c), dtype=np.float64 )
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

    ADD_SELF_LOOPS = False
    MAX_UNIQUE_KEYS = 1000

    #logger.info("Building {}{} cg for {}".format(n, ordinal_follower(n), symbol_name))
    callees = dCounter()
    callers = dCounter()
    #build nth degree callee counts for all callgraphs
    for G in GG:
        if symbol_name not in G.nodes:
            continue

        bin_callees = nth_children(G, symbol_name, n)

        #print(callees)
        assert(isinstance(bin_callees, list))

        for item in bin_callees:
            callees += item

        if callees.unique_keys() > MAX_UNIQUE_KEYS:
            #zero out relationships
            callees = dCounter()
            break

    for G in GG:
        if symbol_name not in G.nodes:
            continue

        bin_callers = nth_parents(G, symbol_name, n)
        assert(isinstance(bin_callers, list))

        for item in bin_callers:
            callers += item

        if callers.unique_keys() > MAX_UNIQUE_KEYS:
            callers = dCounter()
            break

    if not ADD_SELF_LOOPS: 
        #no loops
        callees.remove_node( symbol_name )
        callers.remove_node( symbol_name )

    #logger.info("Callees:" + str(callees))
    #logger.info("Callers:" + str(callers))
    #logger.info(callers - callees)

    print("Finished building cg relations for symbol name {}".format(symbol_name))

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

def build_nth_from_cg(config, symbol_names, _GG, _n, name_to_index):
    d = len(symbol_names)
    nth_callee_count = lil_matrix( (d, d), dtype=np.uint )
    nth_caller_count = lil_matrix( (d, d), dtype=np.uint )
    nth_callee = lil_matrix( (d, d), dtype=np.float64 )
    nth_caller = lil_matrix( (d, d), dtype=np.float64 )
    
    global GG
    global n
    n = _n
    GG = _GG

    mp = Pool(processes=64)
    #args = list(map(lambda x: (GG, x, n), symbol_names))
    #res = mp.starmap(__count_nth_symbol_rels, args )
    res = mp.map(__count_nth_symbol_rels, symbol_names )

    logger.info("Symbol rleations found! Summing relationships!")
    #logger.info("Finalising matrix")

    for callees, callers, symbol_name in res:
        config.logger.debug("Summing relations for {}".format(symbol_name))
        #config.logger.debug("\t->{}".format(callees))
        #config.logger.debug("\t<-{}".format(callers))
        s_index = name_to_index[ symbol_name ]
        nth_callee_count[s_index, :] = callees.to_npvec( (d,), name_to_index)
        nth_caller_count[s_index, :] = callers.to_npvec( (d,) , name_to_index)
        nth_callee[s_index, :] = callees.to_npvec_prob( (d,), name_to_index)
        nth_caller[s_index, :] = callers.to_npvec_prob( (d,) , name_to_index)

    mp.close()
    return nth_callee, nth_caller, nth_callee_count, nth_caller_count

def tqdm_wrapper(names): 
    #load resources
    conf = classes.config.Config(no_logging=True)
    db = classes.database.Database(conf)

    name_to_index = classes.utils.load_py_obj(conf, 'name_to_index')
    #training_set = classes.utils.load_py_obj(conf, 'training_bins')
    d = len(name_to_index)

    #compute pmf for symbol name
    for name in names:
        #query = [ { "$match" : { "name" : name, "path": { "$in" : list(training_set) } } }, { "$project" : { "callees": 1, "callers": 1 } } ]
        #query = [ { "$match" : { "name" : name } }, { "$project" : { "callees": 1, "callers": 1 } } ]

        for rel in [ 'callers', 'callees' ]:

            query = [{ "$match" : { "name" : name } }, { "$project" : { rel: 1 } }, { "$unwind": "$"+rel}, { "$group": { "_id" : "$"+rel, "count": { "$sum" : 1} } } ]

            res = db.run_mongo_aggregate( query , coll_name="tmp_training_set_out" )
            xref_counter = collections.Counter()
            for r in res:
                xref_counter[r['_id']] = r['count']

            scipy_vec = classes.counter.to_scipy_sparse_vec(xref_counter, d, name_to_index)
            u = db.client['xrefs_pmfs'].insert({ "type" : rel, "name": name, "pmf": classes.pmfs.PMF.scipy_sparse_to_bytes(scipy_vec.tocsr())})
            if not u:
                raise Exception("Error adding {} relations to database".format(name))

def __rels_db_optimise_query_out(config):
    config.logger.info("Creating temporary collection with training set")
    db = classes.database.Database(config)
    out_coll_name = 'tmp_training_set_out'
    training_set = classes.utils.load_py_obj(config, 'training_bins')
    match   = { "$match" : { "path": { "$in" : list(training_set) } } }
    out     = { "$out" : out_coll_name } 
    res = db.run_mongo_aggregate([ match, out ])
    config.logger.info("Creating index on name!")
    db.client[out_coll_name].create_index('name')
    config.logger.info("Done!")

def build_rels_from_db(config, symbol_names, name_to_index):

    """
    db = classes.database.Database(config)
    binaries = db.get_number_of_xrefs()
    binaries_with_xrefs = set( {k: v for k, v in binaries.items() if v > 0}.keys() )

    ## > 16MB single response
    #symbols_in_xrefs = db.get_set_all_xrefs( { "$match": { "path" : { "$in" : list(binaries_with_xrefs) } } } )

    config.logger.info("Generating set of unknown symbols and symbols in xrefs")

    res = db.run_mongo_aggregate([ { "$match": { "path" : { "$in" : list(binaries_with_xrefs) } } }, { "$project" : { "callees" : 1, "callers": 1, "name": 1} } ] )


    symbols = set([])
    for r in tqdm(res):
        for s in r['callers'] + r['callees'] + [ r['name'] ]:
            if s not in symbols:
                symbols.add(s)

    config.logger.info("Building XREFS for {} symbol names! Starting after shell exit...".format( len(symbols) ))

    import IPython
    IPython.embed()
    """

    __rels_db_optimise_query_out(config)

    ###to optimize processing, split into N chunks
    #NB: database server is maxed out
    N_PROC = int(32*2)
    CHUNKSIZE = 256
    #data = classes.utils.n_chunks(list(symbol_names), N_PROC)
    data = list( classes.utils.chunks_of_size(list(symbol_names), CHUNKSIZE) )

    pool = multiprocess.Pool(processes=N_PROC)
    #pool = ThreadPool(N_THREADS)

    #for _ in tqdm(pool.imap_unordered(tqdm_wrapper, symbol_names, chunksize), total=len(symbol_names)):
    for _ in tqdm(pool.imap_unordered(tqdm_wrapper, data), total=len(data)):
            pass
    pool.close()
    return

def gradient_descent_pmf(old_pmf, new_pmf):
    GD_alpha = 0.1
    diff = np.subtract( old_pmf , new_pmf )
    correction = np.multiply( GD_alpha, diff )
    return np.subtract( old_pmf, correction )

def update_crf(unknowns, G, cgs, name_to_index, index_to_name, constraints, pmfs, updated_node):
    """
    return an array with teh top 5 probabilities
    """
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
        node_potentials = G.nodes[ node ]['node_potentials']
        pmf = np.multiply( pmf, node_potentials)
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

def infer_crf(unknowns, G, cgs, name_to_index, index_to_name, constraints):
    """
    For all unknowns, calculate the nodes PMF
    :return: Probability Mass Functions as a dict of { node: pmf }
    :rtype: dict
    """
    pmfs = {}
    correct_count = 0
    total = len(unknowns)
    logger.info("{} nodes in the callgraph. {} unknowns. {} knowns.".format( len(G.nodes), total, len(G.nodes) - total ))

    for node in unknowns:
        pmf = np.zeros([N, 1], dtype=np.float)
        n = 1
        for ffi, ffr in cgs:
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
        node_potentials = G.nodes[ node ]['node_potentials']
        #pmf = np.multiply( pmf, node_potentials)
        pmf = pmf +  node_potentials

        pmfs[node] = P.normalise_numpy_density( pmf )

    return pmfs


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
    if max_r <= min_ratio:
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
    node_potentials = G.nodes[ node ]['node_potentials']
    n_ind = np.argmax(node_potentials)
    #r, c = np.where(node_potentials == 1.0)
    #assert(len(r) <= 1)
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
    H = nx.DiGraph()
    for es, ee in G.edges:
        H.add_edge( node_to_inferred_name(G, es, index_to_name), node_to_inferred_name(G, ee, index_to_name) )

    write_dot(H, fname)
    return H


def calculate_node_pmf(G, cgs, name_to_index, node):
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
        old_node_potentials = G.nodes[ node ]['node_potentials']
        new_pmf = gradient_descent_pmf(old_node_potentials, pmfs[node])

        #np.sum( np.subtract( old_node_potentials, new_pmf ) )

        attr = { node : { 'node_potentials' : new_pmf } }
        nx.set_node_attributes(G, attr)
    return G

def total_diff_pmfs(pmfs, m_pmfs, unknowns):
    if not pmfs:
        return float('inf')

    difference = 0.0
    for node in unknowns:
        diff = np.subtract( np.absolute( pmfs[node] ), np.absolute(m_pmfs[node]) )
        #print(diff)
        difference += np.sum( np.absolute(diff) )
        #print(difference)
        #import IPython
        #IPython.embed()

    return difference

def infer_crf_loopy_belief_propagation(G, unknowns, cgs, name_to_index, index_to_name, constraints, db):
    """
        Infer CRF with loopy belief propagation
        Use a greedy algorithm and start with N most confident nodes. 
        Then iterate passing messages until stable
    """

    pmfs = None
    old_diff = float('inf')
    local_maxima_it = 0
    #capped at 50 iterations
    #for i in range(25):
    for i in range(25):  #3 , 2 + final loop
        logger.info("infer_crf_loopy_belief_propagation :: Starting Iteration {}...".format(i))
        new_pmfs = infer_crf(unknowns, G, cgs, name_to_index, index_to_name, constraints)
        G = update_crf_node_pmfs(G, new_pmfs, unknowns)
        #logger.info("Finished iteration. Checking correctness!")
        logger.info("Finished iteration.")

        #if i % 3 == 0:
            #correct, incorrect = check_graph_unknowns_stripped(db, G, index_to_name, unknowns)
            #logger.info("Correct nodes: {}, incorrect nodes: {}, Total unknown: {}, Total known: {}".format(correct, incorrect, len(unknowns), len(G.nodes) - len(unknowns)))

        diff = total_diff_pmfs(pmfs, new_pmfs, unknowns)
        logger.info("Diff: {}".format(diff))

        if diff < old_diff:
            local_maxima_it = 0
        else: 
            #local_maxima_it += 1
            break

        #if local_maxima_it > 1 and i > 3:
        #    break

        if diff < old_diff:
            old_diff = diff

        old_diff = diff
        pmfs = new_pmfs

    return G

def check_graph_unknowns(G, index_to_name, unknowns):
    correct = 0
    for node in unknowns:
        node_potentials = G.nodes[ node ]['node_potentials']
        max_ind = np.argmax( node_potentials )

        if scripts.perform_analysis.check_similarity_of_symbol_name(node, index_to_name[ max_ind ]):
            correct += 1

    return correct, len(unknowns) - correct

def check_graph_unknowns_stripped(db, G, index_to_name, unknowns):
    correct = 0
    for node in unknowns:
        node_potentials = G.nodes[ node ]['node_potentials']
        max_ind = np.argmax( node_potentials )
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
        node_potentials = G.nodes[ node ]['node_potentials']
        r, c = np.where(node_potentials == 1.0)
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


def check_graph_correct_unknown(G, index_to_name,known):
    #check node labels == name index of pmfs for each label
    #check with nlp function from analysis script
    correct, incorrect, unknown = 0, 0, 0
    for node in G.nodes:
        if node in known:
            continue

        node_potentials = G.nodes[ node ]['node_potentials']
        r, c = np.where(node_potentials == 1.0)
        assert(len(r) <= 1)
        assert(len(c) <= 1)
        if len(r) > 0:
            if scripts.perform_analysis.check_similarity_of_symbol_name(node, index_to_name[ r[0] ]):
                correct += 1
            else:
                incorrect += 1
        else:
            unknown += 1

    return correct, incorrect, unknown




def check_graph_correct(G, index_to_name):
    #check node labels == name index of pmfs for each label
    #check with nlp function from analysis script
    correct, incorrect, unknown = 0, 0, 0
    for node in G.nodes:
        node_potentials = G.nodes[ node ]['node_potentials']
        r, c = np.where(node_potentials == 1.0)
        assert(len(r) <= 1)
        assert(len(c) <= 1)
        if len(r) > 0:
            if scripts.perform_analysis.check_similarity_of_symbol_name(node, index_to_name[ r[0] ]):
                correct += 1
            else:
                incorrect += 1
        else:
            unknown += 1

    return correct, incorrect, unknown

if __name__ == '__main__':

    logger.setLevel(logging.INFO)
    global P
    P = classes.pmfs.PMF()

    global bin_path
    global linkage
    global compiler
    global bin_name
    global optimisation


    NO_KNOWNS = False
    ASSIGN_FROM_RES = False

    logger.info("Connecting to mongod...")
    db = Database()

    """
    logger.info("Building all Callgraphs...")
    build_all_cgs(db)
    sys.exit(-1)
    """

    """
    ## build all cgs for ARMv7 binaries
    res = db.distinct_binaries()
    for path in res:
        #if "ppc64" not in path or "dynamic" not in path:
        if "ppc64" not in path:
            continue

        logger.info("\tBuilding cg for {}".format(path))
        build_and_save_cg(path)

    sys.exit()
    """
    """
    #logger.info("Generating distinct symbols and indexes...")
    symbol_names, name_to_index, index_to_name = gen_new_symbol_indexes(db)

    classes.utils.save_py_obj( symbol_names, "symbol_names")
    classes.utils.save_py_obj( name_to_index, "name_to_index")
    classes.utils.save_py_obj( index_to_name, "index_to_name")
    sys.exit()

    """
    #GG = load_all_cgs()

    """
    logger.info("Loading all CGs...")
    GG = mp_load_all_cgs()
    logger.info("Loaded all CGs")

    symbol_names = classes.utils.load_py_obj( "symbol_names")
    name_to_index = classes.utils.load_py_obj( "name_to_index")
    logger.info(len(symbol_names))
    #0th callgraph is the node itself!
    for i in [1, 2]:
        logger.info("Builing n={} degree callgraph".format(i))
        callees, callers, callees_count, callers_count = build_nth_from_cg(symbol_names, GG, i, name_to_index)
        classes.utils.save_py_obj( callees, "{}_degree_cg".format(i))
        classes.utils.save_py_obj( callers, "{}_degree_cgr".format(i))
        classes.utils.save_py_obj( callees_count, "{}_degree_cg_count".format(i))
        classes.utils.save_py_obj( callers_count, "{}_degree_cgr_count".format(i))
        
    sys.exit(-1)
    """

    logger.info("Loading resources...")

    symbol_names = classes.utils.load_py_obj( "symbol_names")
    name_to_index = classes.utils.load_py_obj( "name_to_index")
    index_to_name = classes.utils.load_py_obj( "index_to_name")

    #test_symbols = classes.utils.load_py_obj("test_symbols")
    #unknown_pmfs = classes.utils.load_py_obj("unknown_pmfs")

    #print("#test symbols: {}".format( len(test_symbols) ))
    #assert(len(test_symbols) > 0)

    #[ test_symbol, symbol_name, inferred_pmf ]
    confident_inferred_symbols = classes.utils.load_py_obj("confident_inferred_symbs")
    unknown_inferred_symbols = classes.utils.load_py_obj("unknown_inferred_symbols")

    logger.info("Expanding feature function matracies")
    cg1 = classes.utils.load_py_obj("1_degree_cg").todense()
    cgr1 = classes.utils.load_py_obj("1_degree_cgr").todense()
    cg2 = classes.utils.load_py_obj("2_degree_cg").todense()
    cgr2 = classes.utils.load_py_obj("2_degree_cgr").todense()
    #cg3 = classes.utils.load_py_obj("3_degree_cg").todense()
    #cgr3 = classes.utils.load_py_obj("3_degree_cgr").todense()

    #cgs = [ [cg1, cgr1], [cg2, cgr2], [cg3, cgr3] ]
    cgs = [ [cg1, cgr1], [cg2, cgr2] ]

    logger.info("Resources loaded!")

    print("Confient nodes: {}".format(len(confident_inferred_symbols)))
    print("Unknown nodes: {}".format(len(unknown_inferred_symbols)))

    ##get a list of unique binaries to perform CRF for...
    binaries = set(map(lambda x: x[0].path, confident_inferred_symbols + unknown_inferred_symbols))
    #test_symbol_names = list(map(lambda x: x[0].name, confident_inferred_symbols + unknown_inferred_symbols))

    for bin_path in binaries:
        print(bin_path)
        unstripped_bin_path = bin_path.replace("/bin-stripped/", "/bin/")

        conf_symbs = list(filter(lambda x: x[0].path == bin_path, confident_inferred_symbols))
        unknown_symbs = list(filter(lambda x: x[0].path == bin_path, unknown_inferred_symbols))
        print("\t{} known.".format( len( conf_symbs ) ))
        print("\t{} unknown.".format( len( unknown_symbs ) ))

        conf_correct = list(map(lambda x: scripts.perform_analysis.check_inferred_symbol_name(db, x[0], x[1]),
            conf_symbs))
        unknown_correct = list(map(lambda x: scripts.perform_analysis.check_inferred_symbol_name(db, x[0], x[1]),
            unknown_symbs))

        config = ({ 'paths' : [ unstripped_bin_path ] }, {}) 
        total = scripts.perform_analysis.get_number_of_symbols(db, config)

        print("\t{} correct out of knowns.".format( sum( x == True for x in conf_correct ) ))
        print("\t{} correct out of unknowns.".format( sum( x == True for x in unknown_correct ) ))
        print("\t{} total symbols from symtab.".format( total ))


        linkage = "dynamic" if bin_path.find("dynamic") > 0 else "static"
        stripped = True if bin_path.find("stripped") > 0 else False
        compiler = "gcc" if bin_path.find("gcc") > 0 else "clang"
        optimisation = str( re.findall(r'/o([\dgs])/', bin_path)[0] )
        bin_name = os.path.basename( bin_path )
        collection = "symbols" if not stripped else "symbols_stripped"

        stripped = False
        G = build_binary_cg(db, unstripped_bin_path)

        #logger.info("Building dynamic constraints")
        #dynamic_constraints = build_dynamic_binary_constraints( db, symbol_names, name_to_index )
        #classes.utils.save_py_obj( dynamic_constraints, "dynamic_constraints")
        dynamic_constraints = classes.utils.load_py_obj( "dynamic_constraints")


        ### assign node potentials and save as attribute to each node
        #each node is known
        test_symbol_names = list(map(lambda x: x[0].name, conf_symbs + unknown_symbs))
        known_symbol_names = list(map(lambda x: x[1],           conf_symbs))
        old_known_symbol_names = list(map(lambda x: x[0].name,  conf_symbs))

        unknown_pmfs = list(map(lambda x: x[2], conf_symbs + unknown_symbs))

        #logger.info("Test symbol names: {}".format( test_symbol_names ))
        #logger.info("Known symbol names: {}".format( known_symbol_names ))
        #logger.info("Old known symbol names: {}".format( old_known_symbol_names ))



        logger.info("Building constraints")
        #build index numpy constraints vector 
        N = len(name_to_index)
        #all oness -> any symbols is possible
        constraints = np.ones([N,1])
        for n in G.nodes:
            if n not in old_known_symbol_names and n in name_to_index:
                g_index = name_to_index[ n ]
                constraints[ g_index ] = 0

        
        if linkage == "dynamic":
            logger.info("Adding dynamic constraints")
            constraints = np.multiply( constraints, dynamic_constraints )
            #print(np.shape(constraints))

        #mereg old symbol with symbol properties and new names
        known_symbols = []
        for i in range(len(conf_symbs)):
            s = conf_symbs[i][0]
            o = copy.deepcopy(s)
            o.name = conf_symbs[i][1]
            known_symbols.append(o)

        #add knowns/unknowns and pmfs to graph
        dim = len(symbol_names)
        imported = nx.get_node_attributes(G, 'imported_function')
        unknowns = []
        knowns = []
        remove = []
        for node in G.nodes:

            #print(node)
            """
            Check to see if node is imported dynamic function
            """
            if node in imported.keys():
                pmf = np.zeros( (dim, 1), dtype=np.float)
                if node not in name_to_index:
                    logger.error("SYMBOL {} is not in the symbol name hashmap".format( node ) )
                    remove.append(node)
                    continue
                pmf[ name_to_index[ node ] ] = 1.0
                attr = { node : { 'node_potentials' : pmf } }
                nx.set_node_attributes(G, attr)

                knowns.append(node)
                continue

            """
            Check to see if node has been confidently inferred
            """

            if not NO_KNOWNS: 
                __multiple_known_from_one_name = False
                res = list(filter( lambda x: x[1] == node, enumerate(old_known_symbol_names) ))
                for kn in res:
                    __multiple_known_from_one_name = True
                    #logger.info("Adding confident inferred known symbol")
                    ind, old_name = kn
                    known_name = known_symbol_names[ ind ]
                    knowns.append( known_name )

                    #build new pmf
                    pmf = np.zeros( (dim, 1), dtype=np.float)
                    if known_name not in name_to_index:
                        logger.debug("WARNING! SYMBOL {} is not in the symbol name hashmap".format( known_name ) )
                        remove.append( known_name )
                        continue

                    pmf[ name_to_index[ known_name ] ] = 1.0
                    attr = { node : { 'node_potentials' : pmf } }
                    nx.set_node_attributes(G, attr)

                if __multiple_known_from_one_name:
                    continue

            """
            Add unknown knows with their pmfs
            """
            res = list(filter( lambda x: x[1] == node, enumerate(test_symbol_names) ))
            #print(test_symbol_names)
            #print(node)
            #print(res)
            assert(len(res) > 0)
            #ind, symb = res[0]

            if len(res) > 1:
                logger.error("STAGE 1 resulted in multiple symbols being called the same name!")

            for instance in res:
                ind, symb = instance
                #print(ind)
                #print(symb)
                #assign from import

                if ASSIGN_FROM_RES:
                    ###assign equal probability
                    pmf = unknown_pmfs[ind]
                    attr = { node : { 'node_potentials' : pmf } }
                    nx.set_node_attributes(G, attr)
                    unknowns.append( node )
                else:
                    ###assign initial probability from stage1
                    pmf = np.ones( (dim, 1), dtype=np.float)
                    pmf = np.multiply( pmf, constraints )
                    #assign equal 1/possible symbs
                    r, c = np.where( constraints > 0.0 ) 
                    pmf = np.multiply( float(1) / len(r) , pmf )
                    #pmf[ name_to_index[ node ] ] = 1.0
                    attr = { node : { 'node_potentials' : pmf } }
                    nx.set_node_attributes(G, attr)
                    unknowns.append( node )

        for node in remove:
            #logger.debug("Removing node: {}".format(node))
            G.remove_node( node )


        """
        Hide some nodes
        """
        """
        ratio = 2.0 / 3.0
        logger.info("Hiding {}% of nodes...".format( ratio * 100))
        nodes_to_hide = int( ratio * len(G.nodes) )
        logger.info("Hiding {} nodes!".format(nodes_to_hide))
        unknowns = get_random_sample(list(G.nodes), nodes_to_hide )
        classes.utils.save_py_obj( unknowns, "unknowns")
        #unknowns = classes.utils.load_py_obj("unknowns")

        #print(sample)

        #remove from constraints
        for node in unknowns:
            g_index = name_to_index[node]
            constraints[ g_index ] = 1

        non_zero_nodes = np.where(constraints > 0.0)
        equal_prob = 1.0 / len(non_zero_nodes[0])
        print("equal prob = {}".format(equal_prob))
        #build node potentials
        for node in unknowns:
            pmf = np.ones( (dim, 1), dtype=np.float)    #ones
            pmf = np.multiply( pmf, constraints )       #0 constraints
            pmf = np.multiply( pmf, equal_prob )        #sets ones to equals probs

            attr = { node : { 'node_potentials' : pmf } }
            nx.set_node_attributes(G, attr)
        """

        #logger.info("KNOWNS: {}".format(knowns))
        #logger.info("UNKNOWNS: {}".format(unknowns))



        logger.info("Building CRF with {} knowns and {} unknowns. CRF has {} total nodes.".format(len(knowns), len(unknowns), len(G.nodes)))
        try:
            H = infer_crf_loopy_belief_propagation(G, unknowns, cgs, name_to_index, index_to_name, constraints, db)
            if stripped:
                correct, incorrect = check_graph_unknowns_stripped(db, H, index_to_name, unknowns)
            else:
                correct, incorrect = check_graph_unknowns(H, index_to_name, unknowns)
            logger.info("Correct nodes: {}, incorrect nodes: {}, Total unknown: {}, Total known: {}".format(correct, incorrect, len(unknowns), len(G.nodes) - len(unknowns)))
            save_inferred_graph(H, "graph_inferred", index_to_name)
            logger.info("Finished loopy belief propagation")
            logger.info("Applying |c| == 1 for c in S")
            G = H
            #sys.exit()
        except KeyboardInterrupt:
            logger.info("Caught keyboard press")
            pass
        finally:
            logger.info("Finished loopy belief propagation")

        logger.info("Performing final pass")
        pmfs = infer_crf(unknowns,G,cgs, name_to_index, index_to_name, constraints)

        new_symbols = [] + known_symbols
        corr_symbs = 0
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

            node, new_node_index = find_most_confident_node(pmfs, min_ratio=1.05)

            if node == False:
                #no more confident nodes, stop
                logger.info("No more confident nodes. Stopping")
                break
            print("OLD NODE: {}".format(node))
            print("NEW NODE: {}".format(index_to_name[new_node_index]) )
            new_node_name = index_to_name[ new_node_index ]

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
            print("Old constraints for new node: {}".format( constraints[new_node, 0]))

            if constraints[ new_node, 0] != 1.0:
                assert(False)
            """

            #update constraints
            constraints[new_node_index, 0] = 0

            #remove from unknowns
            del unknowns[ unknowns.index( node ) ]

            #set new node potentials as node
            pmf = np.zeros( (dim, 1), dtype=np.float)
            pmf[ new_node_index, 0 ] = 1.0
            attr = { node : { 'node_potentials' : pmf } }
            nx.set_node_attributes(G, attr)

            pmfs = update_crf(unknowns, G, cgs, name_to_index, index_to_name, constraints, pmfs, node)


        #correct, incorrect, unknowns = check_graph_correct(G, index_to_name)
        #correct, incorrect, unknowns = check_graph_correct(G, list(map(lambda x: x[0].name, unknown_symbs)))
        correct, incorrect, unknowns = check_graph_correct(G, index_to_name, known_symbol_names)
        """
        if not stripped:
            correct, incorrect = check_graph_correct(G, index_to_name)
        else:
            #correct, incorrect = check_graph_correct_stripped(db, G, index_to_name, unknowns)
            correct, incorrect = check_graph_unknowns_stripped(db, G, index_to_name, unknowns)
        """


        logger.info("///////////////Previous")
        print("\t{} known.".format( len( conf_symbs ) ))
        print("\t{} unknown.".format( len( unknown_symbs ) ))

        print("\t{} correct out of knowns.".format( sum( x == True for x in conf_correct ) ))
        print("\t{} correct out of unknowns.".format( sum( x == True for x in unknown_correct ) ))
        print("\t{} total symbols from symtab.".format( total ))

        tp = sum( x == True for x in conf_correct )
        tn = 0
        fp = len(conf_symbs) - tp
        fn = len(unknown_symbs)

        precision   = tp / float(tp + fp)
        recall      = tp / float(tp + fn)
        f1 = 2.0 * (precision * recall)/(precision + recall)

        logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
        logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( precision, recall, f1))

        logger.info("///////////END Previous")


        logger.info("\tStarted with {} known.".format( len( conf_symbs ) ))
        logger.info("\tStarted with {} unknown.".format( len( unknown_symbs ) ))
        logger.info("\tCorrect nodes: {}, incorrect nodes: {}, unknown nodes: {}".format(correct, incorrect, unknowns))

        tp = correct + len(conf_correct)
        tn = 0
        fp = incorrect
        fn = unknowns

        precision   = tp / float(tp + fp)
        recall      = tp / float(tp + fn)
        f1 = 2.0 * (precision * recall)/(precision + recall)

        logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
        logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( precision, recall, f1))





        sys.exit()
        save_inferred_graph(G, "graph_inferred", index_to_name)
        save_new_binary(bin_path, "/tmp/desyl/" + os.path.basename( bin_name ), new_symbols)
        sys.exit(0)

