#!/usr/bin/python3
import logging
import numpy as np
import scipy
import json
import math
from io import BytesIO
import sys
#from multiprocessing import Pool
from multiprocess import Pool
import functools
from annoy import AnnoyIndex

import context
import classes.config
import classes.database
import classes.utils

class PMF:
    def __init__(self, config):
        classes.utils._desyl_init_class_(self, config)
        self.name_to_index = classes.utils.load_py_obj(self.config, 'name_to_index')
        self.index_to_name = classes.utils.load_py_obj( self.config, 'index_to_name')
        self.id_to_gindex = classes.utils.load_py_obj( self.config, 'id_to_gindex')
        self.gid_to_annoydb_id = classes.utils.load_py_obj( self.config, 'gid_to_annoydb_id')
        assert(len(self.name_to_index) == len(self.index_to_name))
        self.dim = len(self.name_to_index)


        self.vex_vec_dim = 32
        self.annoy_db_vex_fname = self.config.desyl + "/res/" + "vex_annoy.tree"
        self.t = AnnoyIndex( self.vex_vec_dim )
        self.t.load( self.annoy_db_vex_fname )


        self.cfg_vec_dim = 1024
        self.annoy_db_cfg_fname = self.config.desyl + "/res/" + "cfg_annoy.CURRENT.tree"
        self.f = AnnoyIndex( self.cfg_vec_dim )
        self.f.load( self.annoy_db_cfg_fname )

        self.index_to_vector = classes.utils.load_py_obj( self.config, "index_to_vector")
        self.vector_to_index = classes.utils.load_py_obj( self.config, "vector_to_index")

    def uniform_pmf(self, shape=None):
        """
            Return a PMF that represents a uniformly random outcome for the current training model
        """
        if not shape:
            return (1.0 / float(len(self.name_to_index))) * np.ones( ( len(self.name_to_index), 1), dtype=np.float128 )
        else:
            return (1.0 / float( shape[0] )) * np.ones( ( shape[0], 1), dtype=np.float128 )

    @staticmethod
    def vex_to_np_mat( opers, exprs, stmts ):
        opers_dim = 17
        exprs_dim = 10
        stmts_dim = 5

        _o = np.matrix(opers).reshape( (opers_dim, 1) )
        _e = np.matrix(exprs).reshape( (exprs_dim, 1) )
        _s = np.matrix(stmts).reshape( (stmts_dim, 1) )

        return functools.reduce(lambda x, y: np.vstack( (x, y) ), [ _o, _e, _s  ] )

    @staticmethod
    def vec_to_name( vec ):
        return classes.utils.list_to_string( vec.tolist() )

    def punish_pmf(self, npvec):
        """
            Punish the density of a PMF based on how many non zero components it has
        """
        r, c = np.where( npvec > 0.0 )
        N = float(len(r))
        if N == 0:
            return npvec
        return ( 1.0 / N ) * npvec

    @staticmethod
    def _gaussian( mu, sigma, x ):
        mu = float(mu)
        sigma = float(sigma)
        x = float(x)
        a = 2.0 * math.pi
        b = 1.0 / ( sigma * math.sqrt( a ) )
        c = math.pow( (x - mu) / sigma , 2 )
        d = (-1.0 / 2.0) * math.pow( c, 2 )
        e = b * math.exp( d )
        return e

    def zero_pmf(self):
        return np.zeros( (self.dim, 1), dtype=np.float128)

    def ones_pmf(self):
        return np.ones( (self.dim, 1), dtype=np.float128)

    @staticmethod
    def norm_gaussian( mu, sigma, x):
        return PMF._gaussian(mu, sigma, x) / PMF._gaussian(mu, sigma, mu)

    def _compute_gaussian_avg_pmf(self, db, feature, value, sigma_ratio=0.1, selection_ratio=0.1):
        """
            For all sizes within size*selection ratio
        """
        #TODO: Need to normalise over each pmf, and then on each addition, or renormalise after
        base = np.zeros( (self.dim, 1), dtype=np.float128)
        bot = int( value * ( 1.0 - selection_ratio) )
        top = int( value * ( 1.0 + selection_ratio) )
        sigma = value * sigma_ratio

        if feature in [ 'vex.ntemp_vars', 'vex.ninstructions' ]:
            self.logger.info("Looking for {} pmf with values between {} <= x < {} with original value {}".format(feature, bot, top, value))
        for s in range(bot, top):
            ##lookup_pmf
            pmf = self.load_pmf(db, feature, s)
            #if feature in [ 'vex_ntemp_vars', 'vex_ninstructions' ]:
            #    if pmf.sum() < 0.01:
            #        self.logger.error("PMF for {} :: {} is empty".format(feature, s))

            m = PMF.norm_gaussian( value, sigma, s)
            base = np.add( base, np.multiply( m, pmf) )

        return self.normalise_numpy_density( base )

    def _compute_graph_pmf(db, G, graph_to_index):
        pass

    @staticmethod
    def normalise_numpy( npobj ):
        mx  = np.max(np.abs(npobj))
        if mx == 0.0:
            #return npobj
            return np.ones(npobj.shape, dtype=npobj.dtype)
        return npobj / mx

    @staticmethod
    def normalise_n( npobj, _max ):
        """
            Scale a numpy object by flooring it to the maximum value
            (Does not rescale if below max)
        """
        mx  = np.max(npobj)
        if mx <= _max:
            return npobj
        return npobj / mx

    def base_pmf(self):
        return self.uniform_pmf()

    def add_uniform_error_and_normalise_density( self, npobj ):
        return self.normalise_numpy_density( PMF.add_uniform_error( npobj ) )

    @staticmethod
    def add_uniform_error( npobj ):
        r, c = np.shape(npobj)
        assert(r > 5000)
        err = 1.0 / (float(r) ** 2)
        return np.add(err, npobj)

    @staticmethod
    def normalise_numpy_density( npobj ):
        total = np.sum( npobj )
        if total < 1.0:
            return npobj
        return np.multiply( 1.0/total, npobj )

    
    @staticmethod
    #def exp_neg_x(_max, _min, value, sharpness=1.0):
    def exp_neg_x(_max, _stretch, value):
        return _max * math.exp( - (value / _stretch) )


    @staticmethod
    def numpy_to_bytes( npobj ):
        with BytesIO() as b:
            np.save(b, npobj)
            return b.getvalue()

    @staticmethod
    def bytes_to_numpy( npbytes ):
        return np.load(BytesIO( npbytes ))
        
    @staticmethod
    def scipy_sparse_to_bytes( scipy_obj ):
        with BytesIO() as b:
            scipy.sparse.save_npz(b, scipy_obj)
            return b.getvalue()

    @staticmethod
    def bytes_to_scipy_sparse( scipy_bytes ):
        return scipy.sparse.load_npz(BytesIO( scipy_bytes ))

    def load_pmf(self, db, feature, value):
        res = db.client[feature + '_pmfs'].find_one({ feature : value })
        if res == None:
            return np.zeros( (self.dim, 1), dtype=np.float128)
        return PMF.bytes_to_numpy( res['pmf'] )

    @staticmethod
    def highest_indexes(pmf, n=5):
        """
            Given a single PMF
                return the top n indexes for values in the PMF
                    [
                        index_0,
                        ...
                    ]
            :param pmf: numpy vectr representing a Probability Mass Function over all symbol names
            :param n: The number of highest values in pmf to return
            :return: An ordered n length list with the highest values in the pmf
            :rtype: list(int)
        """
        #highest n 
        sorted_pmf = np.argsort(pmf[:,0], axis=None)
        py_list_sorted_indexes = sorted_pmf[-n:].tolist()[::-1]

        highest_probs = []
        for i in range(0, n):
            ith_index = py_list_sorted_indexes[i]
            highest_probs.append(ith_index)
        return highest_probs

    
    def pmf_from_vex( self, db, vex, n ):
        vec = PMF.vex_to_np_mat( vex['operations'], vex['expressions'], vex['statements'] )
        return self.pmf_from_vex_vec(db, vec, n)

    def pmf_from_cfg(self, db, id, n ):
        """
        Calculate PMF for a symbol id based on its CFG. 
        :param db: classes.database.Database instance
        :param id: A symbols MongoDB '_id'
        :param n: Number of closest CFG's to consider
        """
        #lookup vector of symol ID 
        #self.logger.info("Loading graph id {}".format(str(id)))
        if str(id) not in self.id_to_gindex:
            self.logger.error("id does not have a graph!!!!!")
            self.logger.error(str(id))
            return np.zeros( (self.dim, 1), dtype=np.float128)

        gid = self.id_to_gindex[str(id)]

        #temp until g2vec is learned
        #return self.load_pmf(db, "cfg", gid)

        #gid of null graph, has no entry in annoy database
        if gid == 6:
            return self.load_pmf(db, "cfg", 6)
        #self.logger.info("Loading graph index {}".format(gid))
        #return self.pmf_from_cfg_gid(db, gid, n)
        return self.pmf_from_cfg_gid(db, gid, n)

    def pmf_from_cfg_gid(self, db, gid, n):
        feature = "cfg"
        """
            Get the PMF for a CFG Vector
            Searching Annoy database and returns a weighting some of the nearest n
            vectors that are within 5 units. (max sqrt(128))
        """
        #vectors are not unique in annoy database
        #get all closest vectors by distance
        annoy_db_id = self.gid_to_annoydb_id[ gid ]
        nearest_ind, distances = self.f.get_nns_by_item( annoy_db_id, n, include_distances=True )
        #max distance is root vector sixe == sqrt(1024)
        #custom max distance to be considered
        max_d = math.sqrt(1024)
        base = np.zeros( (self.dim, 1), dtype=np.float128)
        for i in range(len(nearest_ind)):
            ind = nearest_ind[i]
            dist = distances[i]
            if dist > max_d:
                break
            w = (max_d - dist) / max_d
            assert(w >= 0.0)
            assert(w <= 1.0)
            pmf = self.load_pmf(db, 'cfg', ind)
            
            m_pmf = np.multiply( w, pmf )
            base = np.add(base, m_pmf)

        return self.normalise_numpy_density( base )

    def pmf_from_vex_vec(self, db, np_vec, n):
        """
            Get the PMF for a VEX Vector
            Searching Annoy database and returns a weighting some of the nearest n
            vectors that are within 5 units. (max sqrt(128))
        """
        feature = "vex"
        assert(np.shape(np_vec) == ( self.vex_vec_dim, 1 ))
        #vec_name = PMF.vec_to_name( np_vec )
        #base = self.load_pmf(db, feature, vec_name)

        nearest_ind, distances = self.t.get_nns_by_vector( np_vec, n, include_distances=True )
        #max distance is root vector sixe == sqrt(128)
        #custom max distance to be considered
        max_d = math.sqrt(32)
        base = np.zeros( (self.dim, 1), dtype=np.float128)
        for i in range(len(nearest_ind)):
            ind = nearest_ind[i]
            dist = distances[i]
            if dist > max_d:
                break
            w = (max_d - dist) / max_d
            assert(w >= 0.0)
            assert(w <= 1.0)
            near_vec_name = self.index_to_vector[ ind ]
            pmf = self.load_pmf(db, feature, near_vec_name)
            
            m_pmf = np.multiply( w, pmf )
            base = np.add(base, m_pmf)

        return self.normalise_numpy_density( base )













    def pmf_from_feature(feature, value):
        db = classes.database.Database(self.config)
        self.logger.debug("{}: {}".format(feature, value))
        match = { '$match' : { feature : value } }
        groupby = { '$group' : { '_id' : '$name' } }
        res = db.run_mongo_aggregate( [ match, groupby ] )

        pmf = np.zeros( (self.dim, 1), dtype=np.uint)
        for r in res:
            name = r['_id']
            pmf[ name_to_index[name] ] = 1

        db.client[feature + '_pmfs'].insert( { feature : value, 'pmf' : PMF.numpy_to_bytes( pmf ) } )
        self.logger.info("Inserted pmf for {}::{}".format( feature, value ))


    def build_feature_pmfs(db, feature):
        self.logger.info("Building PMFs for feature: {}".format(feature))
        all_feature_values = db._rnd_distinct_field(feature, 'symbols')
        for f_val in all_feature_values:
            pmf_from_feature(feature, f_val)

    def build_feature_pmfs_threaded(feature):
        proc_pool = Pool(processes=32)
        procs, results = [], []

        self.logger.info("Building PMFs for feature: {}".format(feature))
        all_feature_values = db._rnd_distinct_field(feature, 'symbols')
        self.logger.info("Found {} distinct values...".format( len(all_feature_values) ) )
        self.logger.info("Building PMFs for values...")

        for value in all_feature_values:
            res = proc_pool.apply_async(pmf_from_feature, (feature , value))
            procs.append(res)

        for p in procs:
            while True:
                p.wait(1)
                if p.ready():
                    results.append(p.get())
                    break

        self.logger.info("Built PMFs for {}".format(feature))




