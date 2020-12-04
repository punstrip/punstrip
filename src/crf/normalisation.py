import os, sys, random
import math, gc
import tqdm, multiprocessing
import context, logging
from classes.config import Config
from classes.database import Database
import classes.utils
import classes.callgraph
import crf.known_lib_crf
import numpy as np
import scipy
import itertools
from multiprocess import Pool
from multiprocess.pool import ThreadPool


def _compute_partition_function_check_graph(G):
    for node in G.nodes():
        if G.nodes[node]['func']:
            r, c = scipy.where(G.nodes[node]['potential_function'] > 0) 
            if len(r) != 1:
                raise Exception('Error, normalisation constant needs to be computed over known CRF')
    return True

def compute_partition_func_over_training_set():
    config = Config()
    config.logger.setLevel(logging.INFO)
    db = Database(config)

    config.logger.info("Loading resources...")
    training_bins = classes.utils.load_py_obj(config, "training_bins")
    #GG = classes.callgraph.mp_load_cgs(config, training_bins)
    GG = classes.callgraph.mp_load_all_cgs(config)
    #check all graphs are full of knowns
    list(map(lambda x: _compute_partition_function_check_graph(x), GG))

    #test single path
    #path = "/root/friendly-corpus/debian/coop-computing-tools/usr/bin/catalog_update"
    #GG = [ classes.callgraph.load_cg(config, path) ]

    #set these gobal variables for use by threads
    global _cg
    global _cgr
    global _G
    global _dim
    global _index_to_name
    global _constraints
    global _learning_rate
    global _theta
    global _thetar
    ##############

    _index_to_name = classes.utils.load_py_obj(config, "index_to_name")
    _dim = len(_index_to_name)
    _learning_rate = 1.0
    _constraints = np.ones( (_dim, 1), dtype=np.float128 )

    _cg = classes.utils.load_py_obj(config, "denormalised_cg")
    _cgr = scipy.transpose(_cg).tocsr(copy=True)

    _theta = classes.utils.load_py_obj(config, "theta")
    _thetar = classes.utils.load_py_obj(config, "thetar")

    config.logger.info("Starting calculation of Z(x)")

    z_x = np.zeros( (_dim, 1), dtype=np.float128 )
    for G in tqdm.tqdm(GG):
        global _G
        _G = G
        pool = ThreadPool(64)
        #get messages from unknown nodes
        res = pool.map(update_single_node_sum_product, list(filter(lambda x: _G.nodes[x]['func'] and _G.nodes[x]['text_func'], _G.nodes)))
        for r in res:
            pmf, error = r
            z_x += pmf

        classes.utils.save_py_obj(config, z_x, "z_x")


"""
WARNING THE FOLLOWING FUNC USES GLOBAL VARS AND THREAD POOLS FOR SPEED AND EFFICIENCY
GLOBALS DO NOT SHARE BETWEEN FILES. IT IS THEREFORE COPIED HERE
"""
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

    #product of exponentials is the same as the exponential of the sum

    orig_node_pmf = _G.nodes[ node ]['node_potential']
    new_node_pmf = scipy.zeros( (_dim, 1), dtype=scipy.float128 )
    #new_node_pmf = scipy.ones( (_dim, 1), dtype=scipy.float128 )

    #for u, v in G.edges(nbunch=node):
    for u in _G.predecessors(node):
        v = node

        #skip data refrences
        if _G[u][v]['data_ref']:
            continue

        #emits message from u with the probability of u calling Y
        #msg_update_for_node = emit_callee_message_update_from_rel(G, cg, cgr, u, v, N, index_to_name)
        #msg_update_for_node = emit_caller_message_update_from_mat(_G, _cgr, u, v, _dim, _index_to_name)
        msg_update_for_node = crf.known_lib_crf.emit_caller_message_update_from_mat(_G, _cg, u, v, _dim, _index_to_name, _theta)
        #msg_update_for_node = scipy.exp( msg_update_for_node )

        new_node_pmf += msg_update_for_node
        #new_node_pmf *= msg_update_for_node

    for v in _G.successors(node):
        u = node

        #skip data refrences
        if _G[u][v]['data_ref']:
            continue

        #else:
        #emits message from v with the probability of what u calls
        #msg_update_for_node = emit_caller_message_update_from_rel(G, cg, cgr, u, v, N, index_to_name)
        msg_update_for_node = crf.known_lib_crf.emit_callee_message_update_from_mat(_G, _cgr, u, v, _dim, _index_to_name, _thetar)
        #msg_update_for_node = scipy.exp( msg_update_for_node )

        new_node_pmf += msg_update_for_node
        #new_node_pmf *= msg_update_for_node

    #add constrainst
    #new_node_pmf = new_node_pmf.multiply( _constraints )

    #new_node_pmf = scipy.exp( new_node_pmf / nth_root )
    new_node_pmf = scipy.exp( new_node_pmf )

    #normalised_node_pmf = _P.normalise_numpy_density( new_node_pmf )
    normalised_node_pmf = new_node_pmf
    error = orig_node_pmf - normalised_node_pmf

    updated_node_pmf = orig_node_pmf - ( _learning_rate * error )

    return updated_node_pmf, error





def compute_feature_values_for_graph(G, cg, x, index_hash_map):
    score = 0.0
    #for node in tqdm.tqdm(G.nodes()):
    for node in G.nodes():
        if not G.nodes[node]['func']:
            continue

        for u in G.predecessors(node):
            v = node

            #skip data refrences
            if G[u][v]['data_ref']:
                continue

            if ( name_to_index[u], name_to_index[v] ) not in index_hash_map:
                print("Warning - MISSING LINK IN FEATURE FUNCTION - <{}, {}>".format(u, v))
                assert(False)
                continue

            i = index_hash_map[ ( name_to_index[u], name_to_index[v] ) ]
            w = x[i]

            score += scipy.linalg.norm( w * cg[ name_to_index[u], name_to_index[v] ] )

        for v in G.successors(node):
            u = node

            #skip data refrences
            if G[u][v]['data_ref']:
                continue

            if ( name_to_index[u], name_to_index[v] ) not in index_hash_map:
                print("Warning - MISSING LINK IN FEATURE FUNCTION - <{}, {}>".format(u, v))
                assert(False)
                continue

            i = index_hash_map[ ( name_to_index[u], name_to_index[v] ) ]
            w = x[i]

            score += scipy.linalg.norm( w * cg[ name_to_index[u], name_to_index[v] ] )

    return score

def __argmax_theta(x, r, c, theta, cg, GG, index_hash_map):
    #r, c, theta, cg, GG = args

    #print("Mapping X -> theta")
    #new_theta = theta
    #for i in tqdm.tqdm(range(len(r))):
    #    new_theta[r[i], c[i]] = x[i]

    #print("Computing cost function over graphs")
    score = 0.0
    #for G in tqdm.tqdm(GG):
    for G in GG:
        score += compute_feature_values_for_graph(G, cg, x, index_hash_map)
        score -= scipy.linalg.norm( x )
    return -score

def tqdm_callback(cb_res):
    global bar
    bar.update()

#def compute_theta(GG, theta, cg):
def compute_theta(GG):
    global theta_dok
    global cg_dok

    theta = theta_dok
    cg = cg_dok

    r, c = cg.nonzero()
    items = list(range(len(r)))

    symbol_names = set([])
    for G in GG:
        for node in list(filter(lambda x: G.nodes[x]['func'], G.nodes())):
            if node not in symbol_names:
                symbol_names.add( node )

    symbol_indexes = set(map(lambda x: name_to_index[x], symbol_names))

    batch_r, batch_c = [], []
    for a, b in zip(r, c):
        if a in symbol_indexes and b in symbol_indexes:
            batch_r.append(a)
            batch_c.append(b)

    MAX_VARIABLES = 10000
    if len(batch_r) > MAX_VARIABLES:
        print("refusing to optimize a problem with more than ", str(MAX_VARIABLES), "variables")
        return None, None

    #subsample = random.sample(items, 10000)
    #print("Mapping theta -> X")
    #X = list(map(lambda x: theta[x[0], x[1]], zip(r, c)))
    X = []
    bounds = []
    index_hash_map = {}

    #for count, i  in enumerate( tqdm.tqdm(subsample) ):
    #for count in tqdm.tqdm(range(len(batch_r)) ):
    for count in range(len(batch_r)):
        #X.append( theta[r[i], c[i]] )
        #index_hash_map[ ( r[i], c[i] ) ] = count
        X.append( theta[batch_r[count], batch_c[count]] )
        index_hash_map[ ( batch_r[count], batch_c[count] ) ] = count
        bounds.append( (0.0, 1.0) )

    x = np.array(X)
    args = (r, c, theta, cg, GG, index_hash_map)

    solution = scipy.optimize.minimize(__argmax_theta, x, args, method='L-BFGS-B', options={ 'maxiter':5, 'disp': False }, bounds=bounds)
    #solution = scipy.optimize.minimize(__argmax_theta, x, args, method='COBYLA', options={ 'maxiter':500, 'disp': False })
    #theta_opt = scipy.optimize.minimize(__argmax_theta, x, args, method='Powell', callback=tqdm_callback, options={ 'maxiter':10, 'disp': True })

    #invert theta
    theta_opt, err = [], []
    #theta_opt = theta.todok(copy=True)
    for v, gr, gc in zip(solution.x, batch_r, batch_c):
        #theta_opt[gr, gc] = v
        theta_opt.append( ((gr, gc), v) )
        err.append( scipy.power(theta[gr, gc] - v, 2 ) )

    return theta_opt, scipy.sqrt(sum(err))

    #diff = theta - theta_opt
    #err = scipy.sqrt(diff.power(2).sum())
    #err = scipy.sqrt(diff.power(2).sum()) / float(len(list(filter(lambda x: G[x[0]][x[1]]['call_ref'], G.edges() ))))
    #return theta_opt, diff, err

if __name__ == '__main__':
    #compute_partition_func_over_training_set()

    ##used in threading, needed
    global theta_dok
    global cg_dok

    config = Config()
    logger = config.logger
    db = classes.database.Database(config)

    symbol_names = classes.utils.load_py_obj( config, "symbol_names")
    name_to_index = classes.utils.load_py_obj( config, "name_to_index")
    index_to_name = classes.utils.load_py_obj( config, "index_to_name")

    training_bins = classes.utils.load_py_obj( config, "training_bins" )
    testing_bins = classes.utils.load_py_obj( config, "testing_bins" )

    cg = classes.utils.load_py_obj( config, "denormalised_cg" )
    #theta = classes.utils.load_py_obj( config, "theta" )
    theta = classes.utils.load_py_obj( config, "theta_opt" )

    logger.info("Converting theta and cg to DOK format")
    theta_dok = theta.todok()
    cg_dok = cg.todok()
    logger.info("Loading CRFs")

    #zero theta
    #for r, c in zip(*theta_dok.nonzero()):
    #    theta_dok[r, c] = 0.0
    #theta_dok = scipy.sparse.dok_matrix( scipy.shape(theta_dok), dtype=scipy.float128)


    import IPython
    IPython.embed()
    GG = classes.callgraph.mp_load_cgs(config, training_bins)
    FF = classes.callgraph.mp_load_cgs(config, testing_bins)
    sys.exit()
    """
    for path in tqdm.tqdm(training_bins):
        G = classes.callgraph.build_crf_for_binary(db, path, name_to_index)
        GG.append((G, path))
    """

    #sort by number of variables in optimization so that all results in batch take a similar time
    FF = sorted(GG, key=lambda x: len(list(filter(lambda y, G=x: G.nodes[y]['func'], x.nodes()) )) )

    #classes.utils.save_py_obj(config, GG, "GG_CRFS_named")
    #learning_rate = 1.0 / math.e
    learning_rate = 0.1
    err = float('-inf')
    #mod_theta = compute_theta(GG, [theta_dok, thetar], [cg_dok, cgr])

    logger.info("Chunking...")
    #GG_chunked = classes.utils.chunks_of_size(FF, 10)

    GG_chunked = classes.utils.chunks_of_func(FF, lambda x: len(list(filter(lambda n: x.nodes[n]['func'], x.nodes()))), 10000)

    saver_counter = 0
    logger.info("Starting compute...")
    for chunk in tqdm.tqdm(GG_chunked):
        print("{} elemnts in chunk".format(len(chunk)))

        saver_counter += 1
        if saver_counter % 10 == 0:
            classes.utils.save_py_obj(config, theta_dok.tocsr(copy=True), "theta_opt")

        pool = Pool(64)

        #rnd_crfs = random.sample(GG, 2)
        #rnd_crfs.append(G)

        arrayed_chunk = list(map(lambda x: [ x ], chunk))

        #res = pool.starmap(compute_theta, (chunk, itertools.repeat(theta_dok), itertools.repeat(cg_dok)))
        res = pool.map(compute_theta, arrayed_chunk)
        for mod_theta, mod_err in res:
            if isinstance(mod_theta, type(None)):
                continue

            for pos, opt_v in mod_theta:
                i, j = pos
                diff = theta_dok[i, j] - opt_v
                theta_dok[i, j] = theta_dok[i, j] - (learning_rate * diff)

        #import IPython
        #IPython.embed()

        del res
        pool.close()
        del pool
        gc.collect()

        #mod_theta, diff, mod_err = compute_theta([G], [theta_dok], [cg_dok])

        

    classes.utils.save_py_obj(config, theta_dok.tocsr(copy=True), "theta_opt")

    import IPython
    IPython.embed()
