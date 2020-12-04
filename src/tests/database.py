#!/usr/bin/python3
import sys
import unittest
import binascii
import json
import pprint
import pymongo

sys.path.insert(0, "../src/")

from classes.config import Config
from classes.symbol import Symbol
from classes.database import Database

class TestDatabase(unittest.TestCase):
    config = Config()

    test_symb = Symbol(config, name="hello_world", 
                    size=1054,
                    bin_name="who",
                    vaddr=0x004023f4,
                    optimisation=0,
                    compiler="gcc",
                    bytes= b'\x90\x90\x90',
                    hash= b'\x39\x5F\x90\x93',
                    opcode_hash= b'\x39\x5F\x91\x93',
                    path="/root/friendly-corpus/bin/static/gcc/o0/plain/who",
                    linkage="static",
                    type="FUNC",
                    )


    def test_connection(self):
        db_ref = Database(self.config)
        db = db_ref.client
        self.assertNotEqual(db, False)

    def test_one_symbol(self):
        db = Database(self.config)
        s = db.find_one_symbol_by_name("main")
        print(s)

    def test_insert(self):
        db_ref = Database()
        db = db_ref.client
        symbols = db.symbols

        #count symbols
        count = symbols.count()
        #print("#symbols at start: " + str(count) )

        #perform insert
        res = symbols.insert_one( self.test_symb.clone().to_dict() )
        self.assertNotEqual(res, False)
        iid = res.inserted_id

        #count symbols
        acount = symbols.count()
        self.assertNotEqual(acount, False)
        #print("#symbols after insert: " + str(acount) )

        self.assertEqual( count + 1, acount )

        #get and print symbol
        #symb_cursor = symbols.find( { '_id': iid })
        #for doc in symb_cursor:
        #    pprint.pprint( doc )
        #    symb = Symbol.fromJSON( doc )
        #    print("Printing Symbol: ")
        #    print( symb )

        #clean up
        symbols.delete_one( { '_id': iid } )

        #count symbols
        acount = symbols.count()
        self.assertEqual(count, acount)

    def test_flush(self):
        db_ref = Database()
        db = db_ref.client

        symbols = db.symbols
        count = symbols.count()

        db_ref.flush_symbols()

        acount = symbols.count()

        self.assertLessEqual( acount, count )


    def test_symbol_serialise(self):
        db_ref = Database()
        db = db_ref.client

        a = self.test_symb.clone()
        a.save_to_db(db)

        res = a.find_symbol(db)
        self.assertEqual(res.count(), 1)
        self.assertEqual(a, Symbol.fromDatabase(db, res[0]['_id']))
        #print( Symbol.fromDatabase(db, res[0]['_id']))

    def test_serialise(self):
        db_ref = Database()
        db = db_ref.client
        symbols = db.symbols

        #perform insert
        a = self.test_symb.clone()
        res = symbols.insert_one( a.to_dict() )
        self.assertNotEqual(res, False)
        iid = res.inserted_id

        #get and print symbol
        symb_cursor = symbols.find( { '_id': iid })
        self.assertEqual(1, symb_cursor.count() )
        symb = symb_cursor[0]

        ssymb = Symbol.fromJSON( symb )

        self.assertEqual( ssymb.similarity(a), 1.0 )
        self.assertEqual( a, ssymb )

        #remove new symbol
        res = symbols.delete_one( { '_id': iid } )
        self.assertEqual( 1, res.deleted_count )

    def test_map_reduce(self):
        sys.path.insert(0, "../src/scripts/")
        import perform_analysis

        db_ref = Database()
        db = db_ref.client

        unknown = Symbol.fromDatabase( db, db.symbols.find_one( { 'size' : { '$gt' : 59 } } )['_id'] )
        known_query = {}

        print("[+] Inferring: " + str( unknown ))
        inferred = perform_analysis.map_reduce_infer_symbols(unknown, known_query)

        #pprint.pprint(inferred)
        print("[+] Inferred with matching score of "+ str(inferred['score']))

        self.assertEqual( unknown, Symbol.fromDatabase(db, inferred['id'] ))

if __name__ == '__main__':
    unittest.main()
