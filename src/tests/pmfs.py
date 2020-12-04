#!/usr/bin/python3
import sys
import unittest
import binascii
import json
import logging
import r2pipe
import pprint

import context
import classes.config
import classes.database
import classes.pmfs

cfg = classes.config.Config()
logger = logging.getLogger(cfg.logger)

class TestPMFs(unittest.TestCase):

    db = classes.database.Database()
    P = classes.pmfs.PMF()
    id_to_gindex = classes.utils.load_py_obj('id_to_gindex')


    def test_load_pmf(self):
        gid = 3455
        logger.debug("Loading PMF for graph id: {}".format(gid))
        pmf = self.P.pmf_from_cfg_gid(self.db, gid, 10)

    def test_load_pmf_range(self):
        for gid in range(4000):
            logger.debug("Loading PMF for graph id: {}".format(gid))
            pmf = self.P.pmf_from_cfg_gid(self.db, gid, 10)

        #self.assertEqual(a.bin_name, "who")

    def test_loading_pmfs(self):
        symbol_ids = set()
        
        for config in [cfg.train, cfg.test]:
            query = self.db.gen_query( (config, {'_id':1})  )
            res = self.db.run_mongo_aggregate( query )
            for r in res:
                symbol_ids.add( r['_id'] )

        logger.debug("Loaded all symbol ids from test and training set!")
        logger.debug("Loading all pmfs!")
        for id in symbol_ids:
            self.P.pmf_from_cfg(self.db, id, 10)

if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    unittest.main()
