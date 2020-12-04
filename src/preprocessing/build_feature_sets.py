#!/usr/bin/python3
import logging
import re
import numpy as np
import json
import functools, itertools
from io import BytesIO
import binascii
import sys
from multiprocessing import Pool
from annoy import AnnoyIndex
import scipy
from tqdm import tqdm

import context
from classes.config import Config
from classes.database import Database
from classes.symbol import Symbol
import classes.counter
import classes.utils
import crf.crfs

global dim
global name_to_index
global index_to_name

global train_query
global test_query
global P

def numpy_to_bytes( npobj ):
    with BytesIO() as b:
        np.save(b, npobj)
        return b.getvalue()

def bytes_to_numpy( npbytes ):
    return np.load(BytesIO( npbytes ))

def load_pmf(db, feature, value):
    feature_name = feature.replace(".", "_")
    res = db.client[feature_name + '_pmfs'].find_one({ feature_name : value })
    return bytes_to_numpy( res['pmf'] )

def pmf_from_features(features, values):
    db = Database()
    map(lambda feature, value: logger.debug("{}: {}".format(feature, value)), zip(features, values))
    matches = {}
    for i in range(len(features)):
        matches.update({ features[i] : values[i] })

    match = { '$match' : matches }
    groupby = { '$group' : { '_id' : '$name' } }
    res = db.run_mongo_aggregate( train_query + [ match, groupby ] )

    counter = classes.counter.dCounter()

    #pmf = np.zeros( (dim, 1), dtype=np.float)
    #matched = []
    for r in res:
        name = r['_id']
        counter += name
        #matched.append( name )

    pmf = counter.to_npvec_prob((dim, 1), name_to_index)
    pmf = P.normalise_numpy_density(pmf)
    """
    prob = 0.0
    if len(matched) > 0:
        prob = 1.0 / float(len(matched))
    for name in matched:
        pmf[ name_to_index[name] ] = prob
    """

    features_name = "_".join(features)
    features_name = features_name.replace(".", "_")

    values_name = ""
    for y in values:
        assert(isinstance(y, list))
        values_name += classes.utils.list_to_string( y )

    #values_val = functools.reduce(lambda x, y: str(x) + "".join(y), values, "" )
    values_name = values_name[:-1]
    print(values_name)
    db.client[features_name + '_pmfs'].insert( { features_name : values_name, 'pmf' : numpy_to_bytes( pmf ) } )
    logger.info("{} :: Inserted pmf for {}::{}".format( sys._getframe().f_code.co_name, features_name, values_name ))

def pmfs_from_feature(feature, values):
    db = classes.database.Database()
    for val in values:
        if feature in [ "callees", "callers" ]:
            pmf_from_feature_len(db, feature, val)
        else:
            pmf_from_feature(db, feature, val)


 
def pmf_from_feature_len(db, feature, value):
    logger.debug("{}: {}".format(feature, value))
    match = { '$match' : { feature : { '$size' : value } } }
    groupby = { '$group' : { '_id' : '$name' } }

    res = db.run_mongo_aggregate( train_query + [ match, groupby ] )

    matched = []
    for r in res:
        name = r['_id']
        matched.append(name)

    prob = 0.0
    assert(len(matched) > 0)
    if len(matched) > 0:
        prob = 1.0 / float(len(matched))

    pmf = np.zeros( (dim, 1), dtype=np.float)
    for name in matched:
        pmf[ name_to_index[name] ] = prob

    pmf = P.normalise_numpy_density(pmf)

    feature_name = feature.replace(".", "_")
    db.client[feature_name + '_pmfs'].insert( { feature_name : value, 'pmf' : numpy_to_bytes( pmf ) } )
    logger.info("{} :: Inserted pmf for {}::{}".format( sys._getframe().f_code.co_name, feature, value ))

       

def pmf_from_feature(db, feature, value):
    logger.debug("{}: {}".format(feature, value))
    match = { '$match' : { feature : value } }
    groupby = { '$group' : { '_id' : '$name' } }

    res = db.run_mongo_aggregate( train_query + [ match, groupby ] )

    matched = []
    counter = classes.counter.dCounter()
    for r in res:
        name = r['_id']
        counter += name
        #matched.append(name)

    pmf = counter.to_npvec_prob((dim, 1), name_to_index)
    pmf = P.normalise_numpy_density(pmf)
    """
    prob = 0.0
    assert(len(matched) > 0)
    if len(matched) > 0:
        prob = 1.0 / float(len(matched))

    pmf = np.zeros( (dim, 1), dtype=np.float)
    for name in matched:
        pmf[ name_to_index[name] ] = prob
    """

    if "hash" in feature:
        value = binascii.unhexlify( value )

    feature_name = feature.replace(".", "_")
    db.client[feature_name + '_pmfs'].insert( { feature_name : value, 'pmf' : numpy_to_bytes( pmf ) } )
    logger.info("{} :: Inserted pmf for {}::{}".format( sys._getframe().f_code.co_name, feature, value ))

def extract_all_vex_vectors():
    db = Database()
    #groupby = { '$group' : { '_id' : { '$concatArrays' : [ '$vex.operations', '$vex.expressions', '$vex.statements' ] } } }
    groupby = { '$group' : { '_id' : { 'operations' :'$vex.operations', 'expressions' : '$vex.expressions', 'statements' : '$vex.statements' }, 'names' : { '$push' : '$name' } } }
    unique_vex = db.run_mongo_aggregate( train_query + [ groupby ] )
    return unique_vex

def create_vex_pmf( vex ):
    opers_dim = 17
    exprs_dim = 10
    stmts_dim = 5

    opers = vex['_id']['operations']
    exprs = vex['_id']['expressions']
    stmts = vex['_id']['statements']

    #pmf_from_features([ 'vex.operations', 'vex.expressions', 'vex.statements' ], [ opers, exprs, stmts ])
    #build a pmf for this
    _o = np.matrix(opers).reshape( (opers_dim, 1) )
    _e = np.matrix(exprs).reshape( (exprs_dim, 1) )
    _s = np.matrix(stmts).reshape( (stmts_dim, 1) )

    vec = functools.reduce(lambda x, y: np.vstack( (x, y) ), [ _o, _e, _s  ] )

    counts = classes.counter.dCounter()
    for name in vex['names']:
        counts += name 

    pmf = counts.to_npvec_prob((dim, 1), name_to_index)
    pmf = P.normalise_numpy_density(pmf)
    return vec, pmf

def build_vex_vectors():
    opers_dim = 17
    exprs_dim = 10
    stmts_dim = 5

    annoy_db_fname = cfg.desyl + "/res/" + "vex_annoy.tree"

    vec_dim = opers_dim + exprs_dim + stmts_dim

    index_to_vector = {}
    vector_to_index = {}
    
    count = 0
    t = AnnoyIndex( vec_dim )

    unique_vex = extract_all_vex_vectors()
    mp = Pool(32)
    res = mp.map( create_vex_pmf, unique_vex)

    for vec, pmf in res:
        vec_name = classes.utils.list_to_string( vec.tolist() )
        
        r, c = np.shape(pmf)
        assert(c == 1)
        assert(r == dim)
        r = np.where( pmf > 0.0 )
        assert(len(r) > 0 )
        db.client['vex_pmfs'].insert({'vex': vec_name, 'pmf': numpy_to_bytes(pmf) })

        if vec_name in vector_to_index:
            assert(False)
        else:
            logger.info("Adding vector {}".format(vec_name) )
            t.add_item( count, vec )
            index_to_vector[ count ] = vec_name
            vector_to_index[ vec_name ] = count
            count += 1

    t.build(4 * vec_dim)
    t.save(annoy_db_fname)

    classes.utils.save_py_obj(index_to_vector, "index_to_vector")
    classes.utils.save_py_obj(vector_to_index, "vector_to_index")



def build_feature_pmfs(db, feature):
    logger.info("Building PMFs for feature: {}".format(feature))
    all_feature_values = db._distinct_field(feature, 'symbols', pre_query=train_query)
    for f_val in all_feature_values:
        pmf_from_feature(db, feature, f_val)

def build_feature_pmfs_threaded(feature):
    proc_pool = Pool(processes=32)
    procs, results = [], []

    logger.info("Building PMFs for feature: {}".format(feature))
    if feature in [ "callees", "callers"]:
        all_feature_values = db._distinct_field_len(feature, 'symbols', pre_query=train_query)
    else:
        all_feature_values = db._distinct_field(feature, 'symbols', pre_query=train_query)
    logger.info("Found {} distinct values...".format( len(all_feature_values) ) )
    logger.info("Building PMFs for values...")


    #split into 32 lists
    #each process will create a separate conn to db
    for values in classes.utils.n_chunks(all_feature_values, 32):
        #res = proc_pool.apply_async(pmfs_from_feature, (feature , values), callback=_infer_res, error_callback=_infer_err)
        #procs.append(res)
        pmfs_from_feature(feature , values)
        #, callback=_infer_res, error_callback=_infer_err)
        #procs.append(res)


    for p in procs:
        while True:
            p.wait(1)
            if p.ready():
                results.append(p.get())
                break

    logger.info("Built PMFs for {}".format(feature))

def _infer_err(err):
    logger.error("ERROR INFERRING SYMBOL! {} {}".format( err))
    logger.error("YOU FUCKED UP!")

#thanks, now delete local copy of memory
def _infer_res(res):
    pass
    #gc.collect()


def drop_pmfs(db, features):
    assert(isinstance(features, list))
    for feature in features:
        feature_name = feature.replace(".", "_") + "_pmfs"
        logger.info("Dropping {}".format(feature_name) )
        db.client[feature_name].drop()
        #import IPython
        #IPython.embed()

def build_crf_features(config, name_to_index, symbol_names):

    logger.info("Loading all CGs...")
    comps   = '|'.join( list(map(lambda x: re.escape(x), config.train.compilers)) )
    ops     = '|'.join( list(map(lambda x: re.escape(x), config.train.optimisations)) )
    links   = '|'.join( list(map(lambda x: re.escape(x), config.train.linkages)) )
    bins    = '|'.join( list(map(lambda x: re.escape(x), config.train.bin_names)) )
    archs   = '|'.join( list(map(lambda x: "bin-" + re.escape(x).lower(), config.train.archs)) )

    #special condition for x86_64
    archs = archs.replace("bin-x86_64", "bin")

    pattern = r".*({})_({})_({})_o({})_({}).*".format(archs, links, comps, ops, bins)

    GG = crf.crfs.load_all_cgs(config, pattern)
    #GG = crf.crfs.mp_load_all_cgs(config, pattern)
    config.logger.info("Loaded all CGs")

    config.logger.info(len(symbol_names))
    #0th callgraph is the node itself!
    #for i in [1, 2, 3]:
    for i in [1, 2]:
        logger.info("Building n={} degree callgraph relations".format(i))
        callees, callers, callees_count, callers_count = crf.crfs.build_nth_from_cg(config, symbol_names, GG, i, name_to_index)
        classes.utils.save_py_obj( config, callees, "{}_degree_cg".format(i))
        classes.utils.save_py_obj( config, callers, "{}_degree_cgr".format(i))
        classes.utils.save_py_obj( config, callees_count, "{}_degree_cg_count".format(i))
        classes.utils.save_py_obj( config, callers_count, "{}_degree_cgr_count".format(i))
      

def build_crf_features_debian(config, name_to_index, symbol_names):
    crf.crfs.build_rels_from_db(config, symbol_names, name_to_index)
    return

    pattern = r".*debian_.*"

    GG = crf.crfs.load_all_cgs(config, pattern)
    #GG = crf.crfs.mp_load_all_cgs(config, pattern)
    config.logger.info("Loaded all CGs")

    import IPython
    IPython.embed()

    config.logger.info(len(symbol_names))
    #0th callgraph is the node itself!
    #for i in [1, 2, 3]:
    for i in [1, 2]:
        logger.info("Building n={} degree callgraph relations".format(i))
        callees, callers, callees_count, callers_count = crf.crfs.build_nth_from_cg(config, symbol_names, GG, i, name_to_index)
        classes.utils.save_py_obj( config, callees, "{}_degree_cg".format(i))
        classes.utils.save_py_obj( config, callers, "{}_degree_cgr".format(i))
        classes.utils.save_py_obj( config, callees_count, "{}_degree_cg_count".format(i))
        classes.utils.save_py_obj( config, callers_count, "{}_degree_cgr_count".format(i))

def build_cg_ff_mats_from_db(config, name_to_index, symbol_names):
    dim = len(symbol_names)
    cg_mat = scipy.sparse.csr_matrix( (dim, dim), dtype=np.float128 )
    cgr_mat = scipy.sparse.csr_matrix( (dim, dim), dtype=np.float128 )
    xrefs = db.get_all_xrefs()
    for xr in xrefs:
        ind = name_to_index[ xr['name'] ]
        import IPython
        IPython.embed()
        if xr['type'] == 'callee':
            cg_mat[ ind, : ] = classes.pmfs.PMF.bytes_to_scipy_sparse( xr['pmf'] ).todense()

        elif xr['type'] == 'caller':
            cgr_mat[ ind, : ] = classes.pmfs.PMF.bytes_to_scipy_sparse( xr['pmf'] ).todense()

        else:
            raise Exception("Unknown type of relationship")

    classes.utils.save_py_obj( config, cg_mat, "1_degree_cg".format(i))
    classes.utils.save_py_obj( config, cgr_mat, "1_degree_cgr".format(i))
        
def gen_new_symbol_indexes(db):
    distinct_symbols = set( db.distinct_symbol_names() )
    symbs_from_rels = db.flatten_callees_callers()
    distinct_symbols = list( distinct_symbols.union( set(symbs_from_rels) ) )
    dim = len(distinct_symbols)
    logger.info("Found {} unique symbol names!".format( dim ))

    symbol_to_index = dict( map( lambda x: [distinct_symbols[x], x], range(dim)) )
    index_to_symbol = dict( map( lambda x: [x, distinct_symbols[x]], range(dim)) )
    return distinct_symbols, symbol_to_index, index_to_symbol


def build_and_save_symbol_name_indexes(db, config):
    config.logger.info("Generating distinct symbols and indexes...")
    symbol_names, name_to_index, index_to_name = gen_new_symbol_indexes(db)

    classes.utils.save_py_obj( config, symbol_names, "symbol_names")
    classes.utils.save_py_obj( config, name_to_index, "name_to_index")
    classes.utils.save_py_obj( config, index_to_name, "index_to_name")

if __name__ == "__main__":

    REGEN_SYMBOL_NAME_INDEXES = False
    REBUILD_CGS = False

    config = classes.config.Config()
    logger = config.logger
    logger.setLevel(logging.INFO)
    db = classes.database.Database(config)

    #known = db.get_known_symbol_names()
    #unknown = set(db.distinct_symbol_names()) - known
    #import IPython
    #IPython.embed()

    if REGEN_SYMBOL_NAME_INDEXES:
        #new mapping for symbol name -> index
        build_and_save_symbol_name_indexes(db, config)

    if REBUILD_CGS:
        crf.crfs.build_all_cgs(db)

    #load new mapping
    name_to_index = classes.utils.load_py_obj(config, 'name_to_index')
    index_to_name = classes.utils.load_py_obj(config, 'index_to_name')
    symbol_names = classes.utils.load_py_obj(config, 'symbol_names')



    #fake gindex
    classes.utils.save_py_obj(config, None, 'id_to_gindex')
    classes.utils.save_py_obj(config, None, 'gid_to_annoydb_id')
    classes.utils.save_py_obj(config, None, 'index_to_vector')
    classes.utils.save_py_obj(config, None, 'vector_to_index')

    #train_bins = classes.utils.load_py_obj("train_bins")
    #test_bins = classes.utils.load_py_obj("test_bins")

    train_query_config  = config.train
    test_query_config   = config.test

    #global train_query
    #global test_query
    #global P
    #P = classes.pmfs.PMF(config)
    #train_query = Database.gen_query(train_query_config.__dict__)
    #test_query  = Database.gen_query(test_query_config.__dict__)

    #print(train_query)


    assert(len(name_to_index) == len(index_to_name))
    dim = len(name_to_index)

    #drop_pmfs(db, [ "size", "hash", "opcode_hash", "vex.ntemp_vars", "vex.ninstructions", 'vex_operations_vex_expressions_vex_statements', 'callers', 'callees', 'vex' ] )
    #drop_pmfs(db, [ 'vex_operations_vex_expressions_vex_statements_pmfs'] )

    #build_vex_vectors()
    #for f in [ "size", "hash", "opcode_hash", "vex.ntemp_vars", "vex.ninstructions", "callers", "callees" ]:
    #    build_feature_pmfs_threaded(f)

    #build_crf_features(config, name_to_index, symbol_names)
    build_crf_features_debian(config, name_to_index, symbol_names)
    #build_cg_ff_mats_from_db(config, name_to_index, symbol_names)
