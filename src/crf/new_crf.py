#!/usr/bin/python3
import os, sys
import logging, math, math, math
import numpy as np
import scipy
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import functools
import itertools
import random
import copy, re
import random
import networkx as nx
import glob
from networkx.drawing.nx_pydot import write_dot 
from multiprocessing import Pool



import context
from classes.symbol import Symbol
from classes.database import Database
from classes.config import Config
from classes.counter import dCounter
from classes.bin_mod import BinaryModifier
import classes.utils
import scripts.perform_analysis

random.seed()

cfg = Config()
logger = logging.getLogger(cfg.logger)
#logging.basicConfig(level=logging.DEBUG , format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


global GG
global n
global P

global bin_path

global linkage
global compiler
global bin_name
global optimisation


"""
    Gradient descent parameters
"""

# 0.5 is too agressive, 0.01 is too subtle
GD_alpha = 1.0 / math.e #h


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

       
def prob_yi( node ):
    """
    Compute the probability of node over the set of Symbols
    p(y_i) = s in pi(y_i) |^| p(y_i | y_s )   * s in psi(y_i) |^| p( yi | y_s)

    pi(y_i) is the nodes parents
    psi(y_i) is the nodes children
    """

    pi_y_i = node.parents
    psi_u_i = node.children

    prod = 0.0
    for n in pi_y_i:
        cond_prob = prob_y_i_given_xi(node, n)
        prod *= cond_prob

    for n in psi_y_i:
        cond_prob = prob_y_i_given_x_i(n, node)
        prod *= cond_prob

    return prod


def z_of_x(x):
    """
        return Z(x)
    """
    return 1.0

def prob_y_i_given_x_i(y, x):
    """
        p(y | x) = 1/z(x) * exp( sum ( w_k . f_k(y, x) ) )
    """
    Z = z_of_x(x)
    exponent_sumation = 0.0

    #1 feature
    for i in range(len(feature_functions)):
        w = feature_function_weightings[i]
        r = perform_feature_function( feature_functions[i], y, x)
        exponent_sumation += (r * w)
    
    p = ( 1 / Z ) * math.exp( exponent_sumation )
    return p

def map_inference(O, K, A):
    """
        For each unknown symbol 
    """
    for i in range(O):
        for o in A:
            for k in K:
                cp = compute_conditional_probability(o, k)

#P(O=o | K=k)
def compute_conditional_probability(o, k):
    Z = 1.0
    exponent_sumation = 0.0

    assert(len(o.callees) > 0 or len(o.callers) > 0)
    assert(len(k.callees) > 0 or len(k.callers) > 0)

    #1 feature
    for i in range(1):
        r = feature_function_a_calls_b(o, k )
        w = 1.0
        exponent_sumation += (r * w)
    
    p = ( 1 / Z ) * math.exp( exponent_sumation )
    return p







def mp_load_all_cgs(pattern=r'.*'):
    """
    Load all callgraphs using multiprocessing.

    :return: list of networkx.DiGraph() for each cg
    :rtype: list
    """
    #files = list(glob.iglob(cfg.desyl + '/res/cfgs/' + pattern, recursive=False))
    files = [f for f in os.listdir(cfg.desyl + '/res/cfgs/') if re.search(pattern, f)]
    mf = list(map(lambda x: x.replace(".pickle", ""), files))
    models = list(map(lambda x: 'cfgs/' + os.path.basename(x) , mf))

    p = Pool(processes=32)
    return p.map(classes.utils.load_py_obj, models)

def load_all_cgs(pattern=r'.*'):
    """
    Load all callgraphs caches in /res/cfgs
    :return: list of networkx.DiGraph() for each cg
    :rtype: list
    """
    GG = []

    res = [f for f in os.listdir(cfg.desyl + '/res/cfgs/') if re.search(pattern, f)]
    #for f in glob.iglob(cfg.desyl + '/res/cfgs/'+pattern, recursive=False):
    for f in res:
        f = f.replace(".pickle", "")
        model_name = 'cfgs/' + os.path.basename(f)
        logger.info("Loading object {}".format( model_name ))
        G = classes.utils.load_py_obj( model_name )
        GG.append(G)
    return GG

def build_all_cgs(db):
    """
    Build all callgraphs for each distinct binary in symbols collection of the database
    :param db: An instance of classes.database.Database
    :return: None
    :rtype: None
    """
    bins = db.distinct_binaries()
    p = Pool(processes=20)
    p.map( build_and_save_cg, bins)

def build_and_save_cg(bin_path):
    """
    Build the callgraph for a binary and save it under /res/cfgs/
    :param bin_path: The file path of the binary to build a callgraph for
    :return: None
    :rtype: None
    """
    db = classes.database.Database()
    G = build_binary_cfg(db, bin_path)
    fname = bin_path.replace("/", "_")
    classes.utils.save_py_obj( G, "cfgs/" + fname + ".cfg")


def build_binary_cfg(db, path, collection='symbols'):
    """
    Build a callgraph for a binary by loading relations from the database
    :param db: An instance of classes.database.Database
    :path path: Full path of the binary to build the callgraph for.
    :param collection: Collection name to use for looking up symbols in the database
    :return: The callgraph
    :rtype: networkx.DiGraph
    """

    logger.info("Fetching symbols in {}".format(path))
    symbols = db.get_symbols_from_binary(path, collection)
    logger.debug(symbols)

    logger.info("Buidling symbol hash maps for binary...")

    #build symbol name to vaddr hash map
    symb_to_vaddr = dict(map(lambda x: [x.name, x.vaddr], symbols))
    vaddr_to_symb = dict(map(lambda x: [x.vaddr, x.name], symbols))

    logger.info("{} symbols in binary '{}'".format( len(symbols), path))
    G = nx.DiGraph()
    for s in symbols:
        ## mod for using CFG from real binary!!
        #symb_name = __mod_name_to_stripped(symb_to_vaddr, s.name)
        symb_name = s.name

        for c in s.callees:
            G.add_edge( symb_name, c )

        for c in s.callers:
            #G.add_edge( __mod_name_to_stripped(symb_to_vaddr, c), symb_name )
            G.add_edge( c, symb_name )


        if len(s.callers + s.callees) == 0:
            G.add_node( symb_name )
            logger.debug("WARNING! Unconnected function: '{}'".format(s.name))

    for n in G.nodes:
        #imported symbol!!
        if n not in symb_to_vaddr:
            symb_to_vaddr[n] = 0
            attr = { n : { 'imported_function' : n } }
            nx.set_node_attributes(G, attr)

        #logger.debug("Labelling node {} with label vaddr={}".format(n, symb_to_vaddr[n]))
        G.nodes[n]['vaddr'] = symb_to_vaddr[n]

    logger.info("{} nodes (symbols) in CFG for binary '{}'".format( len(G.nodes), path))

    return G#, symbols

def new_loopy_belief_prop(G, W):
    ##get cliques that do not have label-word feature functions
    ## label-label edges only

    set_counter = 1
    cliques = list( filter( lambda x: '::' not in ''.join(x) ,nx.find_cliques(G) ) )
    for i in range(1000):
        #clique_counter = 0
        epoch_diff = 0.0
        for clique in cliques:
            for node in clique:
                G, diff = update_node_potentials(G, W, node, clique)
                epoch_diff += diff
            #print("FINSIHED CLIQUE {}".format(clique_counter))
            #clique_counter += 1
        print("FINSHED EPOCH {}".format(set_counter))
        print("EPOCH DIFF {}".format(epoch_diff))
        set_counter += 1
        if not set_counter % 1:
            check_correctness(G, W)

    return G


def node_confidence(npmat):
    """
        Highest difference between max and max_-1
    """
    #print(np.max(npmat))
    arr = np.squeeze( np.asarray( npmat ) )
    maxinds = np.argsort(arr)[-2:]
    #print( maxinds )
    assert(np.shape(maxinds)[0] == 2)
    conf = arr[ maxinds[1] ] - arr[ maxinds[0] ]
    assert(conf >= 0.0)
    return conf

def new_loopy_belief_prop_conf_node_technique(G, W):
    ##get cliques that do not have label-word feature functions
    ## label-label edges only

    set_counter = 1
    unknown_nodes = set( filter( lambda x: not G.node[x]['known'] , G.nodes ) )
    conf = list(map(lambda x: [ x, np.max(G.node[x]['node_potential']) ], unknown_nodes))
    #conf = list(map(lambda x: [ x, node_confidence(G.node[x]['node_potential']) ], unknown_nodes))
    conf.sort(key = lambda x: x[1])
    conf = conf[::-1]

    rec_list = []
    avg = P.uniform_pmf()[0,0]
    for i in range(len(conf)):
        #if conf[i][1] < 3 * avg:
        #    print(conf[i][1])
        #    break

        rec_list.append( conf[i][0] )


        

        epoch_diff = 0.0
        for node in rec_list:
            neigh = set([])
            for u, v in G.edges(nbunch=node):
                if u in unknown_nodes:
                    neigh.add(u)
                if v in unknown_nodes:
                    neigh.add(v)
            neigh -= set([node])
            for other in neigh:
                #print( G.edges(nbunch=node) )
                #print( neigh )
                G, diff = emit_node_potentials(G, W, other, neigh)
                epoch_diff += diff
        check_correctness(G, W)
        print("FINSHED EPOCH {}".format(set_counter))
        print("EPOCH DIFF {}".format(epoch_diff))
        set_counter += 1
    return G


def emit_node_potentials(G, W, emitter, clique):
    """
        Emit msg update from node to all other nodes
"""
    #beta = 1.0 / float( len( G.edges(nbunch=emitter) ) )
    #beta = 1.0 / len(clique)
    #beta = 10.0
    others = set(clique) - set([emitter])

    logger.debug("emitting from {} that we think is {}".format(emitter, index_to_name[ np.argmax( G.node[emitter]['node_potential'] ) ]))
    #import IPython
    #IPython.embed()

    if len(others) == 0:
        logger.error("No has no connection, skipping msg update")
        return G, 0.0

    assert(len(others) > 0)
    assert(len(others) + 1 == len(clique))
    assert(not G.node[emitter]['known'])



    for node in others:
        node_pot = G.node[node]['node_potential']
        #label_node_pot = __calculate_label_node_potentials(G, W, node)

        edge_sum = P.zero_pmf()

        avg = P.uniform_pmf()[0,0]

        for other_node_id in [ emitter ]:
            if not G.node[node]['known']:
                #other_node_pot = __calculate_label_node_potentials(G, W, other_node_id)
                other_node_pot = G.node[other_node_id]['node_potential']


                #assert(np.max(other_node_pot) > avg)
                #assert(np.max(node_pot) > avg)

                edge_pot = (np.transpose( other_node_pot ) @ np.transpose( cg ) ) #* node_pot #* node_pot
                #edge_sum += W['label-label'] * P.normalise_numpy_density(np.transpose(edge_pot)) * np.exp( 1 + np.max(other_node_pot) + np.max( node_pot) - (2*avg)  )
                edge_sum += W['label-label'] * P.normalise_numpy_density(np.transpose(edge_pot)) 
                #edge_pot_r = (np.transpose( other_node_pot ) @ cgr ) #* node_pot
                #edge_sum += W['label-label'] * P.normalise_numpy_density(np.transpose(edge_pot_r)) * np.exp(1 + np.max(node_pot) + np.max(other_node_pot - (2*avg) )) 
                #edge_sum += W['label-label'] * P.normalise_numpy_density(np.transpose(edge_pot_r))

                #edge_pot2 = (np.transpose( other_node_pot ) @ cg2 ) #* node_pot #* node_pot
                #edge_sum += W['label-label2'] * P.normalise_numpy_density(np.transpose(edge_pot2)) * ( 1 + np.max(other_node_pot) + np.max( node_pot) - (2*avg) )
                #edge_pot_r2 = (np.transpose( other_node_pot ) @ cgr2 ) #* node_pot
                #edge_sum += W['label-label2'] * P.normalise_numpy_density(np.transpose(edge_pot_r2)) * ( 1 + np.max(node_pot) + np.max(other_node_pot ) - (2*avg) ) 




                #edge_sum *= W['label-label'] * P.normalise_numpy_density(  np.transpose(edge_pot_r)) * np.linalg.norm(other_node_pot, ord=2) * np.linalg.norm( node_pot, ord=2) 
                

        #new_label_pot = P.normalise_numpy_density( np.exp( edge_sum ) )
        #new_label_pot = P.normalise_numpy_density( label_node_pot + edge_sum )
        new_label_pot = P.normalise_numpy_density( edge_sum )
        diff_pot = node_pot - new_label_pot

        total_diff = np.sum( np.absolute(diff_pot))
        #import IPython
        ##IPython.embed()
        beta = float(len(W)) / float( len( G.edges(nbunch=node) ) )
        #print("beta == {}".format(beta))
        #G.node[node]['node_potential'] = P.normalise_numpy_density( node_pot - ( beta * diff_pot ) )
        G.node[node]['node_potential'] = node_pot - ( beta * diff_pot )
        #G.node[node]['node_potential'] = P.normalise_numpy_density( new_label_pot )

    return G, total_diff




def update_node_potentials(G, W, node, clique):
    """
        Update node_potentials based on messages passed
"""
    #beta = 1.0 / float( len( G.edges(nbunch=node) ) )
    beta = 0.5
    others = set(clique) - set([node])

    if len(others) == 0:
        logger.error("No has no connection, skipping msg update")
        return G, 0.0

    assert(len(others) > 0)
    assert(len(others) + 1 == len(clique))
    assert(not G.node[node]['known'])

    node_pot = G.node[node]['node_potential']
    label_node_pot = __calculate_label_node_potentials(G, W, node)
    #ff_sum = label_node_pot
    #ff_sum = copy.deepcopy( node_pot )

    edge_sum = P.ones_pmf()

    for other_node_id in clique:
        if not G.node[node]['known']:
            #other_node_pot = __calculate_label_node_potentials(G, W, other_node_id)
            other_node_pot = G.node[other_node_id]['node_potential']

            #benefit confident messages more th aunconfident ones

            #edge_pot = (np.transpose( other_node_pot ) @ cg ) * label_node_pot #* node_pot
            edge_pot = (np.transpose( node_pot ) @ cg ) * other_node_pot #* node_pot
            #ff_sum += W['label-label'] * P.normalise_numpy_density( np.exp( np.linalg.norm(node_pot, ord=2) * np.linalg.norm(other_node_pot, ord=2)) * np.transpose(edge_pot))

            #ff_sum += W['label-label'] * P.normalise_numpy_density(  np.transpose(edge_pot)) * np.exp( np.linalg.norm(node_pot, ord=2) * np.linalg.norm(other_node_pot, ord=2))

            #print(edge_pot)
            edge_sum += W['label-label'] * P.normalise_numpy_density(np.transpose(edge_pot)) * np.exp( 1 + np.max(other_node_pot) + np.max( node_pot ) )
            #import IPython
            #IPython.embed()

            #edge_pot_r = (np.transpose( other_node_pot ) @ cgr ) * label_node_pot
            #edge_pot_r = (np.transpose( other_node_pot ) @ cgr ) * node_pot
            #ff_sum += W['label-label'] * P.normalise_numpy_density(np.transpose(edge_pot_r) )


            #ff_sum += W['label-label'] * P.normalise_numpy_density(np.transpose(edge_pot_r)) * np.exp( np.linalg.norm(node_pot, ord=2) * np.linalg.norm(other_node_pot, ord=2)) 

            #edge_sum *= W['label-label'] * P.normalise_numpy_density(  np.transpose(edge_pot_r)) * np.linalg.norm(other_node_pot, ord=2) * np.linalg.norm( node_pot, ord=2) 
            

    #new_label_pot = P.normalise_numpy_density( np.exp( edge_sum ) )
    #new_label_pot = P.normalise_numpy_density( label_node_pot * edge_sum )
    new_label_pot = P.normalise_numpy_density( edge_sum )
    diff_pot = node_pot - new_label_pot

    total_diff = np.sum( np.absolute(diff_pot))

    G.node[node]['node_potential'] = P.normalise_numpy_density( node_pot - ( beta * diff_pot ) )
    return G, total_diff

def run_init_node_pots(G, W):
    for node in G.nodes():
        if not G.node[node]['known']:
            label_node_pot = __calculate_label_node_potentials(G, W, node)
            new_pot = np.exp( label_node_pot )
            G.node[node]['node_potential'] = new_pot
    return G


def check_correctness(G, W):
    correct, incorrect = 0, 0
    for gid in G.nodes():
        if not G.node[gid]['known']:
            #node_pot = __calculate_label_node_potentials(G, W, gid)
            #node_pot = calculate_yt(G, W, gid)
            node_pot = G.node[gid]['node_potential']
            most_prob = np.argmax( node_pot )
            #print(node_pot)
            if gid == index_to_name[ most_prob ]:
            #if scripts.perform_analysis.check_similarity_of_symbol_name(gid, index_to_name[ most_prob ]):
                correct += 1
                print("NODE: {} ==> {}".format( gid, index_to_name[ most_prob ] ) )
            else:
                incorrect += 1

    tp = correct
    tn = 0
    fp = incorrect
    fn = 0

    precision   = 0.0 if tp+fp == 0 else tp / float(tp + fp)
    recall      = 0.0 if tp+fn == 0 else tp / float(tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * (precision * recall)/(precision + recall)

    logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
    logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( precision, recall, f1))




def infer_new_crf(G, W):
    G = run_init_node_pots(G, W)
    #G = new_loopy_belief_prop(G, W)
    G = new_loopy_belief_prop_conf_node_technique(G, W)

    correct, incorrect = 0, 0
    for gid in G.nodes():
        if not G.node[gid]['known']:
            #node_pot = __calculate_label_node_potentials(G, W, gid)
            #node_pot = calculate_yt(G, W, gid)
            node_pot = G.node[gid]['node_potential']
            most_prob = np.argmax( node_pot )
            #print(node_pot)
            #print("NODE: {} ==> {}".format( gid, index_to_name[ most_prob ] ) )
            if gid == index_to_name[ most_prob ]:
                correct += 1
            else:
                incorrect += 1

    tp = correct
    tn = 0
    fp = incorrect
    fn = 0

    precision   = 0.0 if tp+fp == 0 else tp / float(tp + fp)
    recall      = 0.0 if tp+fn == 0 else tp / float(tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * (precision * recall)/(precision + recall)

    logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
    logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( precision, recall, f1))


def __set_node_property(G, node_id, key, value):
    """
        Add property to a node in a graph
    """
    attr = { node_id : { key : value } }
    nx.set_node_attributes(G, attr)

def __set_edge_property(G, edge_tuple, key, value):
    """
        Add property to an edge between a tuple of nodes 
    """
    attr = { edge_tuple : { key : value } }
    nx.set_edge_attributes(G, attr)

def __calculate_label_node_potentials(G, W, node):
    """
        Calculate node potential for unknown node given know nodes and weights
    """
    ff_sum = P.zero_pmf()
    for u, v in G.edges(nbunch=node):
        if not G.node[v]['known']:
            continue

        pmf = G[u][v]['potential']
        ff = G[u][v]['feature_function_name']
        #t = W[ff] * P.add_uniform_error_and_normalise_density( pmf )
        #inf norm
        #t = W[ff] * pmf * np.exp( np.max( pmf ) )
        t = W[ff] * pmf
        ff_sum += t
    #import IPython
    #IPython.embed()

    #print(ff_sum)
    #print(G.node[node]['node_potential'])
    ##element wise multiplication of vectors
    #return P.normalise_numpy_density( np.multiply( ff_sum, G.node[node]['node_potential'] ) )
    #return P.add_uniform_error_and_normalise_density( np.multiply( ff_sum, G.node[node]['node_potential'] ) )
    ###NB TODO: Uncomment multiply by node potential when i have looy belief prop
    #return np.multiply( ff_sum, G.node[node]['node_potential'] ) 
    return ff_sum
    #return P.add_uniform_error_and_normalise_density( ff_sum )
    #return P.normalise_numpy_density(ff_sum )

def calculate_psi(G, W, clique):
    """
        Calculate psi( y_a, x_a ) = exp ( sum_K( W F_k(y, a) ) )
        :param clique: Tuple of node ids that represent clique
        :param G: CRF
        :param W: Theata feature function weightings
        :return: psi(y_a, x_a ; W )
        :rtype: numpy matrix
    """

    unknown = []
    for gid in clique:
        assert( gid in G.nodes())
        if not G.node[gid]['known']:
            node_pot = __calculate_label_node_potentials(G, W, gid)
            unknown.append( [gid, node_pot] )

    ff_sum = P.zero_pmf()
    ##all gids connect to each other
    for gid, node_pot in unknown:
        ff_sum += node_pot
        ff_sum += G.node[ gid ]['node_potential']

        node_pot = G.node[ gid ]['node_potential']

        for ugid, unode_pot in unknown:
            if gid == ugid:
                continue

            unode_pot = G.node[ugid]['node_potential']

            #1st degree cg forward and back
            edge_pot = (np.transpose( unode_pot ) @ cg ) * node_pot #* node_pot
            ff_sum += W['label-label'] * P.normalise_numpy_density( np.transpose(edge_pot)) * ( 1 + np.max( node_pot) + np.max(unode_pot)) 
            edge_pot_r = (np.transpose( unode_pot ) @ cgr ) * node_pot
            ff_sum += W['label-label'] * P.normalise_numpy_density(np.transpose(edge_pot_r) ) * ( 1 + np.max( node_pot) + np.max(unode_pot))

            #2nd degree cg
            edge_pot2 = (np.transpose( unode_pot ) @ cg2 ) * node_pot #* node_pot
            ff_sum += W['label-label2'] * P.normalise_numpy_density( np.transpose(edge_pot2)) * ( 1 + np.max( node_pot) + np.max(unode_pot)) 
            edge_pot_r2 = (np.transpose( unode_pot ) @ cgr2 ) * node_pot
            ff_sum += W['label-label2'] * P.normalise_numpy_density(np.transpose(edge_pot_r2) ) * ( 1 + np.max( node_pot) + np.max(unode_pot))



    #return P.add_uniform_error_and_normalise_density( ff_sum )
    #return np.exp( ff_sum )
    return ff_sum
    #return np.linalg.norm( np.exp( ff_sum ) )
    #return np.max( np.exp( ff_sum ) )

def calculate_z_x(G, W):
    z_x = []
    ##get cliques that do not have label-word feature functions
    ## label-label edges only
    cliques = list( filter( lambda x: '::' not in ''.join(x) ,nx.find_cliques(G) ) )
    for clique in cliques:
        psi_c = calculate_psi(G, W, clique)
        z_x.append( psi_c )
        #z_x *= P.normalise_numpy_density( psi_c )
        #z_x *= P.add_uniform_error_and_normalise_density( psi_c )

    return functools.reduce(lambda x, y: x * y, z_x)
    #return P.normalise_numpy_density( functools.reduce(lambda x, y: x * y, z_x) )
    #return P.normalise_numpy_density( z_x )

def __likelihood_z_x(G, W):
    z_x = 0.0
    ##get cliques that do not have label-word feature functions
    ## label-label edges only
    cliques = list( filter( lambda x: '::' not in ''.join(x) ,nx.find_cliques(G) ) )
    for clique in cliques:
        #print("Clique:")
        #print(clique)
        psi_c = calculate_psi(G, W, clique)
        
        true_pmf = P.zero_pmf()
        for node in clique:
            true_pmf += G.node[node]['node_potential']

        diff = P.normalise_numpy_density( true_pmf )  - P.normalise_numpy_density( psi_c )
        z_x += np.sum( np.absolute(diff) )
        #z_x += np.dot( np.transpose(true_pmf),  psi_c)[0,0]
        #z_x += psi_c

    #return functools.reduce(lambda x, y: x * y, psis)
    #return P.normalise_numpy_density( z_x )
    #return np.log( z_x )
    return z_x



def calculate_likelihood(G, W ):
    """
        Calculate:
        l(w) = | sum[ wf(y,x) ] | - | log Z(x) |
        :rtype: real number
    """
    #print("CALCULATE Z_X")
    #z_x = calculate_z_x(G, W)
    #print("DONE CALCULATE Z_X")
    #assert(np.shape(z_x) == np.shape(P.base_pmf()))
    ##print("Z(x):")
    ##print(z_x)
    #log_z_x = np.log( z_x )
    ##norm_log_z_x = np.linalg.norm( log_z_x, ord=2 )
    #norm_log_z_x = np.max( log_z_x )


    #print("CALCULATE LIKELIHOOD Z_X")
    sum_l = __likelihood_z_x(G, W)
    #print("DONE CALCULATE LIKELIHOOD Z_X")
    #norm_l = np.linalg.norm( sum_l, ord=2 )
    #norm_l = np.linalg.norm( sum_l )

    ##regularization parameter
    w = np.zeros( ( len(W.items()), 1 ), dtype=np.float64 )
    i=0
    for k,v in W.items():
        w[i,0] = v
        i+=1

    #return norm_l - norm_log_z_x - np.linalg.norm(w)
    return sum_l #- np.linalg.norm(w)

def __gen_random_weightings(W, min_bound, max_bound):
    """
        Gen random weightings for each weighting in dict W bound between min_bound and
        max_bound
    """
    V = {}
    for k in W.keys():
        #V[k] = min_bound + ( (max_bound - min_bound) * np.random.random_sample() )
        V[k] = min_bound + ( (max_bound - min_bound) * random.random() )
    return V

def __stochastic_grad_descent_logliklihood(G, W, min_bound, max_bound):
    num_features = len(W.keys())
    num_iterations = 10 * num_features
    beta = 0.5
    diff = 0.0
    last = calculate_likelihood(G, W)
    first = last
    print("/=============")
    print("Original W:")
    print("\t{} - {}".format(last, W))
    #want to maximize liklihood

    #print(W)
    #print(num_features)
    for it in range(num_iterations):
        #rand = np.random.random_integers(num_features) - 1
        #rand = int( math.floor(random.random() * num_features) )
        rand = random.randint(0, num_features-1)
        ##NB: rand is of type numpy.int64 not int!!!!!!!
        feature, weighting = list(itertools.islice(W.items(), int(rand), int(rand+1)))[0]
        mod_weight = beta * ( 2 * (random.random()  - 0.5))

        ##don't go out of bounds
        if W[feature] + mod_weight > max_bound or W[feature] + mod_weight < min_bound:
            continue

        W[feature] += mod_weight
        current = calculate_likelihood(G, W)
        diff = current - last

        ## if worse results, revert Weight change
        #here our aim is to minimise the loglikhood function
        if diff > 0.0:
            W[feature] -= mod_weight
        else:
            last = current

    print("Best W:")
    print("\t{} - {}".format(last, W))
    print("\\=============")
    assert(last <= first)
    return W, last

def maximize_likelihood(G, W):
    num_weights = len(W.keys())
    min_bound = 0.0
    max_bound = 1.0
    num_iterations = int( math.sqrt( (3 * (max_bound-min_bound)) ** num_weights ) )
    print("Peforming {} iterations!".format( num_iterations ) )
    max_weightings = None
    max_liklihood = math.inf
    for it in range(num_iterations):
        W_mod = __gen_random_weightings(W, min_bound, max_bound)
        W_mod, liklihood = __stochastic_grad_descent_logliklihood(G, W_mod, min_bound, max_bound)

        if liklihood < max_liklihood:
            max_liklihood = liklihood
            max_weightings = W_mod

    return max_weightings, max_liklihood


def build_binary_crf(db, path, collection='symbols'):
    """
    Build a conditional random field from a binary
    :param db: An instance of classes.database.Database
    :path path: Full path of the binary to build the callgraph for.
    :param collection: Collection name to use for looking up symbols in the database
    :return: The CRF
    :rtype: networkx.DiGraph
    """

    G = nx.Graph()
    logger.info("Fetching symbols in {}".format(path))
    symbols = db.get_symbols_from_binary(path, collection)

    #query = Database.gen_query((cfg.train, {}))
    #symbols = db.get_symbols('symbols', query)


    #logger.debug(symbols)


    #feature function values are on the edges between x and y nodes. 
    #cannot have feature that is only of x. x needs to connect to y
    for s in symbols:
        node_id = str( s.name )
        G.add_node( node_id, style='filled', fillcolor='grey' )
        __set_node_property(G, node_id, 'known', False)


        #unknown node, starts with uniformly random
        #node_pmf =  P.uniform_pmf()

        #known node
        node_pmf = P.zero_pmf()
        node_pmf[ name_to_index[ s.name ], 0 ] = 1.0

        __set_node_property(G, node_id, 'node_potential', node_pmf )


        G.add_node( node_id + '::hash' )
        __set_node_property(G, node_id + '::hash', 'known', True)
        __set_node_property(G, node_id + '::hash', 'value', s.hash)
        G.add_edge(node_id, node_id + '::hash')
        #hash_pmf = P.normalise_numpy_density( P.load_pmf(db, "hash", s.hash) )
        hash_pmf = P.load_pmf(db, "hash", s.hash)
        #__set_edge_property(G, (node_id, node_id + '_hash'), 'potential', P.punish_pmf( hash_pmf ))
        __set_edge_property(G, (node_id, node_id + '::hash'), 'potential', hash_pmf )
        __set_edge_property(G, (node_id, node_id + '::hash'), 'feature_function_name', 'hash' )



        G.add_node( node_id + '::opcode_hash' )
        __set_node_property(G, node_id + '::opcode_hash', 'known', True)
        __set_node_property(G, node_id + '::opcode_hash', 'value', s.opcode_hash)
        G.add_edge(node_id, node_id + '::opcode_hash')
        #opcode_hash_pmf = P.normalise_numpy_density( P.load_pmf(db, "opcode_hash", s.opcode_hash) )
        opcode_hash_pmf = P.load_pmf(db, "opcode_hash", s.opcode_hash)
        #__set_edge_property(G, (node_id, node_id + '_opcode_hash'), 'potential', P.punish_pmf( opcode_hash_pmf ))
        __set_edge_property(G, (node_id, node_id + '::opcode_hash'), 'potential', opcode_hash_pmf )
        __set_edge_property(G, (node_id, node_id + '::opcode_hash'), 'feature_function_name', 'opcode_hash' )



        G.add_node( node_id + '::vex' )
        __set_node_property(G, node_id + '::vex', 'known', True)
        #__set_node_property(G, node_id + '::vex', 'value', s.vex)
        G.add_edge(node_id, node_id + '::vex')

        vex = { 
            'operations'    : s.vex['operations'], 
            'expressions'   : s.vex['expressions'], 
            'statements'    : s.vex['statements'], 
        }
        vex_pmf = P.pmf_from_vex( db, vex, 3)
        #__set_edge_property(G, (node_id, node_id + '_vex'), 'potential', P.punish_pmf( vex_pmf ))
        __set_edge_property(G, (node_id, node_id + '::vex'), 'potential', vex_pmf )
        __set_edge_property(G, (node_id, node_id + '::vex'), 'feature_function_name', 'vex' )




        G.add_node( node_id + '::cfg' )
        __set_node_property(G, node_id + '::cfg', 'known', True)
        #__set_node_property(G, node_id + '::cfg', 'value', s.cfg)
        G.add_edge(node_id, node_id + '::cfg')
        cfg_pmf = P.pmf_from_cfg(db, s._id, 3)
        #__set_edge_property(G, (node_id, node_id + '_cfg'), 'potential', P.punish_pmf( cfg_pmf ))
        __set_edge_property(G, (node_id, node_id + '::cfg'), 'potential', cfg_pmf )
        __set_edge_property(G, (node_id, node_id + '::cfg'), 'feature_function_name', 'cfg' )



        G.add_node( node_id + '::size' )
        __set_node_property(G, node_id + '::size', 'known', True)
        __set_node_property(G, node_id + '::size', 'value', s.size)
        G.add_edge(node_id, node_id + '::size')

        sel_hyper_r = P.exp_neg_x(10, 10, s.size)
        r = 0.15 * (1.0 + sel_hyper_r)
        size_pmf = P._compute_gaussian_avg_pmf(db, "size", s.size, sigma_ratio=r, selection_ratio=r )
        size_pmf = P.normalise_numpy_density( size_pmf )
        #__set_edge_property(G, (node_id, node_id + '_size'), 'potential', P.punish_pmf( size_pmf ))
        __set_edge_property(G, (node_id, node_id + '::size'), 'potential',  size_pmf )
        __set_edge_property(G, (node_id, node_id + '::size'), 'feature_function_name', 'size' )


    for s in symbols:
        node_id = str( s.name )
        for conn in s.callees + s.callers:
            #if not in all symbosl we want to infer, ignore. The nth degree callgraph feature shoudl should take
            #care. We don't give a shit about learning relationships between known nodes, just hwo this relates
            #to unknown nodes p(y|x).
            if conn in G.nodes:
                G.add_edge(node_id, str( conn ))


        logger.info( node_id )
        logger.info(G.node[node_id])

    write_dot(G, "crf.dot")
    #sys.exit()
    return G




    logger.info("Buidling symbol hash maps for binary...")

    #build symbol name to vaddr hash map
    symb_to_vaddr = dict(map(lambda x: [x.name, x.vaddr], symbols))
    vaddr_to_symb = dict(map(lambda x: [x.vaddr, x.name], symbols))

    logger.info("{} symbols in binary '{}'".format( len(symbols), path))
    for s in symbols:
        ## mod for using CFG from real binary!!
        #symb_name = __mod_name_to_stripped(symb_to_vaddr, s.name)
        symb_name = s.name

        for c in s.callees:
            G.add_edge( symb_name, c )

        for c in s.callers:
            #G.add_edge( __mod_name_to_stripped(symb_to_vaddr, c), symb_name )
            G.add_edge( c, symb_name )


        if len(s.callers + s.callees) == 0:
            G.add_node( symb_name )
            logger.debug("WARNING! Unconnected function: '{}'".format(s.name))

    for n in G.nodes:
        #imported symbol!!
        if n not in symb_to_vaddr:
            symb_to_vaddr[n] = 0
            attr = { n : { 'imported_function' : n } }
            nx.set_node_attributes(G, attr)

        #logger.debug("Labelling node {} with label vaddr={}".format(n, symb_to_vaddr[n]))
        G.nodes[n]['vaddr'] = symb_to_vaddr[n]

    logger.info("{} nodes (symbols) in CFG for binary '{}'".format( len(G.nodes), path))

    return G#, symbols



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

    logger.info("Building {}{} cg for {}".format(n, ordinal_follower(n), symbol_name))
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

    return nth_callee, nth_caller, nth_callee_count, nth_caller_count

def flatten_callees_callers(db):
    proj    = { '$project' : { 'name' : { '$concatArrays' : [ '$callers', '$callees'] } } }
    unwind  = { '$unwind' : '$name' }
    group   = { '$group' : { '_id' : '$name' } } 
    return list(map(lambda x: x['_id'], db.run_mongo_aggregate( [ proj, unwind, group ] ) ))


def gen_new_symbol_indexes(db):
    distinct_symbols = set( db.distinct_symbol_names() )
    symbs_from_rels = flatten_callees_callers(db)
    distinct_symbols = list( distinct_symbols.union( set(symbs_from_rels) ) )
    dim = len(distinct_symbols)

    symbol_to_index = dict( map( lambda x: [distinct_symbols[x], x], range(dim)) )
    index_to_symbol = dict( map( lambda x: [x, distinct_symbols[x]], range(dim)) )
    return distinct_symbols, symbol_to_index, index_to_symbol

def ordinal_follower(n):
    last_digit = str(n)[-1]
    if last_digit == "1":
        return "st"
    elif last_digit == "2":
        return "nd"
    elif last_digit == "3":
        return "rd"
    else:
        return "th"


def get_random_sample( l, n ):
    """
        return a random sample of n elements in l
    """
    N = len(l)
    inds = set([])
    while len(inds) < n:
        inds.add( random.randint(0, N-1) )
    return list( map( lambda x: l[x], inds) )


def debug_np_obj(obj, txt=""):
    if len(txt):
        print(txt)
    print("Shape: {}".format( np.shape(obj) ) )
    print( obj )

def gradient_descent_pmf(old_pmf, new_pmf):
    #GD_alpha
    diff = np.subtract( old_pmf , new_pmf )
    correction = np.multiply( GD_alpha, diff )
    return np.subtract( old_pmf, correction )

def score_crf(unknowns, G, pmfs):
    H = copy.deepcopy(G)

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
        pmf = np.multiply( pmf, node_potentials)

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
    #for i in 25):
    for i in range(0):  #3 , 2 + final loop
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
            local_maxima_it += 1

        if local_maxima_it > 3 and i > 10:
            break

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


def export_crf_to_function_cg(G, fname):
    """
        Export CRF maximum node potentials to function callgraph
    """
    H = nx.Graph()
    for node in list( filter( lambda x: '::' not in ''.join(x), G.nodes() ) ):

        aname = index_to_name[ np.argmax( G.node[node]['node_potential'] ) ]
        if aname == node:
            H.add_node( aname, style='filled', fillcolor='grey' )

        H.add_node( "real_" + node, style='filled', fillcolor='grey' )

        for u, v in G.edges(nbunch=node):
            if G.node[u]['known'] or G.node[v]['known']:
                continue



            uname = index_to_name[ np.argmax( G.node[u]['node_potential'] ) ]
            vname = index_to_name[ np.argmax( G.node[v]['node_potential'] ) ]

            if uname == u:
                H.add_node( uname, style='filled', fillcolor='grey' )
            if vname == v:
                H.add_node( vname, style='filled', fillcolor='grey' )

            H.add_node( "real_" + u, style='filled', fillcolor='grey' )
            H.add_node( "real_" + v, style='filled', fillcolor='grey' )

            H.add_edge(uname, vname)
            H.add_edge("real_" + u, "real_" + v)
    write_dot(H, fname)


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

    global cg
    global cgr
    global cg2
    global cgr2
    global index_to_name
    global name_to_index

    cg = classes.utils.load_py_obj("1_degree_cg").todense()
    cgr = classes.utils.load_py_obj("1_degree_cgr").todense()
    cg2 = classes.utils.load_py_obj("2_degree_cg").todense()
    cgr2 = classes.utils.load_py_obj("2_degree_cgr").todense()

    index_to_name = classes.utils.load_py_obj( "index_to_name")
    name_to_index = classes.utils.load_py_obj( "name_to_index")


    NO_KNOWNS = False
    ASSIGN_FROM_RES = True

    logger.info("Connecting to mongod...")
    db = Database()


    #G = build_binary_crf(db, "/root/friendly-corpus/bin/dynamic/clang/og/who")
    #classes.utils.save_py_obj( G, "crf")
    G = classes.utils.load_py_obj( "crf")
    #W = {'size': 0.19112890216667489, 'hash': 0.1821384028077787, 'opcode_hash': 0.07805943437490115, 'vex': 0.22635400362124425, 'cfg': 0.003079851360380137}
    #W = {'size': 0.08390240867994895, 'hash': 20.28889670562214165, 'opcode_hash': 20.025373976052040503, 'vex': 0.30064337665164391, 'cfg': 0.94063561906220556, 'label-label': 1.32958798116940891} 
    #W = {'size': 0.2915302522297891, 'hash': 0.99, 'opcode_hash': 0.8712701352303624, 'vex': 0.6203195348508783, 'cfg': 0.5829651832309078, 'label-label': 0.9631837299527738, 'label-label2': 0.9} 
    W = {'size': 0.5922272871240989, 'hash': 0.12512052327258202, 'opcode_hash': 0.24845973325973517, 'vex': 0.4474067138971722, 'cfg': 0.6728487876584411, 'label-label': 0.9924369426387137, 'label-label2': 0.9948691242667378}
    W = {'size': 0.004078936407323486, 'hash': 0.1036469222547125, 'opcode_hash': 0.1892807092847274, 'vex': 0.19459984233724448, 'cfg': 0.33180620356064716, 'label-label': 0.1257732449755663, 'label-label2': 0.07775237398619639}


    """
    W_max, likelihood_max = maximize_likelihood(G, W)
    print("Inferred weightings and likelihood:")
    print("\t{}".format(W_max))
    print("\t{}".format(likelihood_max))
    classes.utils.save_py_obj( W, "W_max")
    sys.exit()
    """

    W = {'size': 0.0922272871240989, 'hash': 100.12512052327258202, 'opcode_hash': 10.24845973325973517, 'vex': 1.474067138971722, 'cfg': 0.1728487876584411, 'label-label': 0.9924369426387137, 'label-label2': 0.00948691242667378}   

    #W = {'size': 0.2863922334020193, 'hash': 0.6938467514413893, 'opcode_hash': 0.042104350376966415, 'vex': 0.3066747195126544, 'cfg': 0.724706893227107, 'label-label': 0.8892011229117429, 'label-label2': 0.9144706671006126} 

    #W = classes.utils.load_py_obj( "W_max")
    print(W)
    infer_new_crf(G, W)
    export_crf_to_function_cg(G, "crf.callgraph.dot")
    import IPython
    IPython.embed()
    sys.exit()
    """
    logger.info("Building all Callgraphs...")
    build_all_cgs(db)
    sys.exit(-1)
    """

    #logger.info("Generating distinct symbols and indexes...")
    #symbol_names, name_to_index, index_to_name = gen_new_symbol_indexes(db)

    #classes.utils.save_py_obj( symbol_names, "symbol_names")
    #classes.utils.save_py_obj( name_to_index, "name_to_index")
    #classes.utils.save_py_obj( index_to_name, "index_to_name")
    #sys.exit()

    #GG = load_all_cgs()

    """
    logger.info("Loading all CGs...")
    GG = mp_load_all_cgs()
    logger.info("Loaded all CGs")

    symbol_names = classes.utils.load_py_obj( "symbol_names")
    name_to_index = classes.utils.load_py_obj( "name_to_index")
    logger.info(len(symbol_names))
    #0th callgraph is the node itself!
    for i in [2]:
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
    cg3 = classes.utils.load_py_obj("3_degree_cg").todense()
    cgr3 = classes.utils.load_py_obj("3_degree_cgr").todense()

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
        G = build_binary_cfg(db, unstripped_bin_path, collection="symbols")

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

