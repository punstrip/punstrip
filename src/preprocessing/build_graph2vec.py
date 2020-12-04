#!/usr/bin/python3
import logging
import numpy as np
import json
import bson
import functools
from io import BytesIO
import binascii
import sys
import os, re
from multiprocessing import Pool
from annoy import AnnoyIndex
import networkx as nx
from networkx.drawing import nx_agraph
import pygraphviz

import context
from classes.config import Config
from classes.database import Database
from classes.symbol import Symbol
import classes.counter
import classes.utils
import preprocessing.build_feature_sets

from scripts.gk_weisfeiler_lehman import GK_WL
from networkx.drawing.nx_pydot import write_dot
#from networkx.drawing.nx_pydot import write_gefx

cfg = Config()
logger = logging.getLogger(cfg.logger)

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

def parse_graph2vec_annoydb(fname):
    """
        Parse resulting vectors from graph2vec and store in Annoy Database
        :param fname: Filename of graph2vec results
    """
    vecs = json.load( open(fname, 'r') )
    annoy_db_fname = cfg.desyl + "/res/" + "cfg_annoy.tree"
    cfg_dim = 1024
    t = AnnoyIndex( cfg_dim )

    ### WARNING ANNOY DOESN'T WORK IF INDEXES ARE NOT CONTINUOUS!
    ### i.e. 0-N. 0-5, 7-N will result in an error.
    ### Graph 6 is the null graph, every graph after 6 needs index decremented.
    gid = 6
    v = [-10] * cfg_dim
    t.add_item(gid, v)
    ### ^ this is required

    for k, v in vecs.items():
        #print(len(v))
        #print(v)
        f = os.path.basename( k )
        p = re.compile(r'(\d+)\.gexf\.g2v3')
        m = p.match(f)
        #entire string and 
        assert(len(m.groups())) == 1
        gid = int( m.group(1) )

        #logger.debug("Adding {} :: {}".format(gid, f))
        t.add_item(gid, v)
    logger.info("Added all items. Building index!")
    t.build(4 * cfg_dim)
    t.save(annoy_db_fname)


def build_subset_annoy_db(global_annoy_db, gid_to_annoydb_id):

    annoy_db_fname = cfg.desyl + "/res/" + "cfg_annoy.CURRENT.tree"
    cfg_dim = 1024
    t = AnnoyIndex( cfg_dim )

    for k, v in gid_to_annoydb_id.items():
        vec = global_annoy_db.get_item_vector(k)
        t.add_item(v, vec)

    t.build(4 * cfg_dim)
    t.save(annoy_db_fname)

def get_symbol_ids_in_query_config(db, query_config):
    """
    Get a list of ObjectIds from the db that are allowed in the training set
    """
    allowed = set()
    query = Database.gen_query((query_config, {}), projection={'_id':1})
    res = db.run_mongo_aggregate( query )
    for r in res:
        allowed.add( bson.ObjectId( r['_id'] ) )
    return allowed

def get_symbol_ids_in_testing(db):
    """
    Get a list of ObjectIds from the db that are allowed in the testing set
    """
    return get_symbol_ids_in_query_config(db, cfg.test)

def get_symbol_ids_in_training(db):
    """
    Get a list of ObjectIds from the db that are allowed in the training set
    """
    return get_symbol_ids_in_query_config(db, cfg.train)




###GRAPH 6 IS THE NULL NODE
##TODO: NULL OUT SYMBOLS THAT ARE NOT IN THE TRAINING SET!
def graph_ids_to_pmf(graphs, ids, name_to_index, allowed_ids, testing_ids):
    dim = len(name_to_index)
    db = classes.database.Database()

    id_to_gindex = {}
    gid_to_annoydb_id = { } 


    total_allowed = allowed_ids.union(testing_ids)

    graph_counter = 0
    annoy_db_counter = 0
    for fingerprint, vs in graphs.items():
        for v in vs:
            #unique graph
            id = v['_id']
            graph = v['graph']
            symbol_ids = set(filter(lambda x: x in total_allowed, set(map(lambda x: bson.ObjectId(x), ids[ id ])) ))
            #symbol_ids = set(map(lambda x: bson.ObjectId(x), ids[ id ]))

            if len(symbol_ids) == 0:
                continue

            #write n to all symbols ids 
            for tsid in symbol_ids:
                id_to_gindex[ str(tsid) ] = graph_counter


            ##if this is the first time were being run, we don't need to perform sanity check but will need to recompute graph2vec vectors!!
            """
            logger.info("performing sanity check")
            SANITY_CHECK = nx.read_gexf(cfg.res + "/gid_to_graph/" + str(graph_counter) + ".gexf")
            assert(nx.is_isomorphic( SANITY_CHECK, graph))
            logger.info("sanity check passed!")
            """

            ##all graphs get added to gid_to_graph but only allowed ids are added to pmf calculation
            symbol_ids = list(filter(lambda x: x in allowed_ids, symbol_ids))


            #save as graph n
            #nx.write_gexf(graph, cfg.res + "/gid_to_graph/" + str(graph_counter) + ".gexf" )

            #get symbol names from ids
            query = { '$match' : { '_id' : { '$in' : symbol_ids } } }
            proj = { '$project' : { 'name' : 1 } }
            res = db.run_mongo_aggregate( [ query, proj ] )
            counter = classes.counter.dCounter()
            for s in res:
                counter += s['name']

            #create and save pmf for graph n
            #pmf = counter.to_npvec_unique_prob((dim, 1), name_to_index)
            pmf = counter.to_npvec_prob((dim, 1), name_to_index)
            if np.max(pmf) > 0.0:
                logger.info("Inserting pmf")
                pmf = P.normalise_numpy_density(pmf)
                db.client["cfg_pmfs"].insert({ "cfg" : graph_counter, 'pmf': numpy_to_bytes(pmf) })

            gid_to_annoydb_id[ graph_counter ] = annoy_db_counter
            annoy_db_counter += 1
            graph_counter += 1

    classes.utils.save_py_obj(id_to_gindex, "id_to_gindex")
    classes.utils.save_py_obj(gid_to_annoydb_id, "gid_to_annoydb_id")

    return id_to_gindex, gid_to_annoydb_id
    ##now need to create gindex_to_vec

def find_unique_graphs(db):
    kern = GK_WL()
    count = 0
    symbols = []
    res = db.client.symbols.find({}, { 'cfg': 1, '_id': 1})
    for r in res:
        symbols.append( [r['cfg'], r['_id'] ] )

    print("symbols saved into local array")

    for s in symbols:
        r = { '_id' : s[1], 'cfg' : s[0] }

        count += 1
        if count % 25 == 0:
            print("{} unique graphs found".format( len(ids.keys()) ) )
            print("On graph: {}".format(count))

        #fingerprint = r['vex']['fingerprintuctions']
        nx_g = nx_agraph.from_agraph( pygraphviz.AGraph( r['cfg'] ) )
        fingerprint = str(len(nx_g.nodes)) + "_" + str(len(nx_g.edges))
        nodes = len(nx_g.nodes)

        if fingerprint not in graphs.keys():
            #add graph with id 
            graphs[fingerprint] = [ { '_id' :  r['_id'] , 'graph' : nx_g } ] 
            #add id list for graph id
            ids[ r['_id'] ] = [ r['_id'] ]
            continue


        unique=True
        for graph_d in graphs[fingerprint]:
            id = graph_d['_id']
            graph = graph_d['graph']

            if nodes == 0:
                ids[ id ].append( r['_id'] )
                unique=False
                break

            if kern.compare(graph, nx_g, h=3, node_label=False) == 1.0:
                ids[ id ].append( r['_id'] )
                unique=False
                break
        if unique:
            graphs[fingerprint].append( { '_id' :  r['_id'] , 'graph' : nx_g } )
            ids[ r['_id'] ] = [ r['_id'] ]

    classes.utils.save_py_obj(ids, "cfg_ids")
    classes.utils.save_py_obj(graphs, "cfg_graphs")
    #print(ids)
    sys.exit()       






if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    name_to_index = classes.utils.load_py_obj('name_to_index')
    index_to_name = classes.utils.load_py_obj('index_to_name')

    train_query_config = cfg.train
    test_query_config = cfg.test

    train_query = Database.gen_query((train_query_config, {}))
    test_query = Database.gen_query((test_query_config, {}))


    assert(len(name_to_index) == len(index_to_name))
    dim = len(name_to_index)

    db = Database()
    P = classes.pmfs.PMF()

    preprocessing.build_feature_sets.drop_pmfs(db, ['cfg'])

    cfg_graphs = classes.utils.load_py_obj('cfg_graphs')
    cfg_ids = classes.utils.load_py_obj('cfg_ids')

    
    allowed_ids = get_symbol_ids_in_training(db)
    testing_ids = get_symbol_ids_in_testing(db)
    id_to_gindex, gid_to_annoydb_id = graph_ids_to_pmf(cfg_graphs, cfg_ids, name_to_index, allowed_ids, testing_ids)

    #Now build temporrary annoy db only containing vectors that hve useful information in. no 0 pmf vectors
    cfg_vec_dim = 1024
    annoy_db_cfg_fname = cfg.desyl + "/res/" + "cfg_annoy.tree"
    global_annoy_db = AnnoyIndex( cfg_vec_dim )
    global_annoy_db.load( annoy_db_cfg_fname )
    build_subset_annoy_db(global_annoy_db, gid_to_annoydb_id)

    sys.exit(-1)

    ### Only run below when retarining graph2vec
    """
    embeddings_f = "/root/graph2vec_tf/embeddings/_dims_1024_epochs_50_lr_0.3_embeddings.txt"
    parse_graph2vec_annoydb(embeddings_f)
    sys.exit()
    """


    ##build buckets of graphs based on vex.ninstructions (don't have bb info)
    graphs = {}
    ids = {}

    """
        Write all graphs as gephs 
    """
    """
    #res = db.client.symbols.find({'cfg' : { '$ne' : "strict digraph {\n}\n"}},  {'cfg': 1, '_id': 1 })
    res = db.client.symbols_stripped.find({'cfg' : { '$ne' : "strict digraph {\n}\n"}},  {'cfg': 1, '_id': 1 })

    logger.info("Loaded graphs. Wrinting to res...")
    for r in res:
        nx_g = nx_agraph.from_agraph( pygraphviz.AGraph( r['cfg'] ) )
        nx.write_gexf(nx_g, cfg.res + "/gexf/" + str(r['_id']) + ".gexf" )

    sys.exit()
    """

    #classes.utils.save_py_obj(ids, "cfg_ids")
    #classes.utils.save_py_obj(graphs, "cfg_graphs")





















    logger.info("Comparing {} graphs!".format( len(graphs) ))

    #comp = kern.compare_list(graphs, h=20, node_label=False)

    unique_graphs = []
    c = 0
    for g in graphs:
        unique = True
        for h in unique_graphs:
            #if nx.is_isomorphic(g, h):
            #    unique = False
            #    break

            if kern.compare(g, h, h=1, node_label=False) == 1.0:
                unique = False
                break

        if unique:
            unique_graphs.append( g )

        c+=1

        logger.info("Compared {} graphs.".format(c))

    logger.info("{} unique graphs.".format( len(unique_graphs) ) )
    c = 1
    for i in unique_graphs:
            write_dot(i, "{}.dot".format(c))
            c+=1

    #cfg_sim = kern.compare( self.cfg, other.cfg, h=kern_iterations, node_label=False )




