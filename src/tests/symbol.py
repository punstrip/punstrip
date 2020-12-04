#!/usr/bin/python3
import sys
import unittest
import binascii
import json
import logging
import r2pipe
import pprint

import context
from classes.config import Config
from classes.symbol import Symbol
from classes.basicblock import BasicBlock
from classes.basicblocksequence import BasicBlockSequence
from classes.database import Database


class TestSymbol(unittest.TestCase):
    config = Config()
    logger = config.logger
    #db_ref = Database(config)
    #db = db_ref.client

    def _gen_inline_symbol(self):
        return Symbol(self.config, name="hello", size=101,
                bin_name="who", vaddr=0x5234,
                optimisation=0, compiler="gcc",
                data="4c8b0f4c8b57084d39d1735631c94d8b014d85c0743b4839ca763f488d41014c8904ce498b49084885c9741e0f1f40004839c2742b4c8b014883c0014c8944c6f8488b49084885c975e64c8b57084889c14983c1104d39ca77b44889c8c36690f3c331c0c3",
                path="/root/abc")

    def test_symbol_print(self):
        s = self._gen_inline_symbol()
        print(s.to_str_full())

    def test_new_basic(self):
        self.logger.debug("Testing new basic symbol creation...")
        a = Symbol(self.config, name="hello", size=0, bin_name="who")
        self.assertEqual(a.name, "hello")
        self.assertEqual(a.size, 0)
        self.assertEqual(a.bin_name, "who")

    def test_new_complex(self):
        self.logger.debug("Testing new symbol creation with bytes...")
        a = Symbol( self.config, name="hello", size=3,
                bin_name="who", vaddr=0x5234,
                optimisation=0, compiler="gcc",
                data=binascii.hexlify( b'\x90\x90\x90').decode('utf-8'),
                path="/root/abc")

    def test_clone_similarity(self):
        a = self._gen_inline_symbol()
        b = a.clone()
        self.logger.debug("Testing similarity between cloned symbols...")
        sim = a.similarity(b)
        self.logger.debug("Similarity of cloned symbols: " + str(sim) )
        #self.assertSequenceEqual( list(sim), [1.0] * len( sim ) )

    def test_different_similarity(self):
        a = self.dummy.clone()
        b = Symbol( name="hello", size=4,
                bin_name="who", vaddr=0x5234,
                optimisation=0, compiler="gcc",
                data= b'\x90\x90\x90\xf4',
                path="/root/abc")
        b.analyse("fake")
        self.logger.info("Testing similarity between different symbols...")
        self.logger.debug(a)
        #self.logger.debug(b)
        sim = a.similarity(b)
        self.logger.debug("Similarity of different symbols: " + str(sim) )
        #self.assertSequenceEqual( list(sim), [1.0] * len( sim ) )
        self.assertNotEqual( list(sim), Symbol.similarity_weightings )



    def test_size_similarity(self):
        a = Symbol(size=10, data="90909090909090909090")
        b = Symbol(size=20, data="9090909090900990909090909090909090909090")
        a.analyse("")
        b.analyse("")
        self.logger.debug("Testing similarity between symbol of size 10 and size 20")

        size_sim = a.similarity(b)[0]
        self.logger.debug("Size similarity: " + str(size_sim) )
        self.assertLess( size_sim, 0.75 )

    def test_hash_generation(self):
        a = self.dummy.clone()
        b = self.dummy.clone()


        b.data += b'\x90\x90'
        b.size += 2

        a.gen_hashes()
        b.gen_hashes()

        self.logger.debug("Symb hashes:")
        self.logger.debug(a.asm)
        self.logger.debug(b.asm)

        self.assertNotEqual(a.hash, b.hash)
        self.assertNotEqual(a.opcode_hash, b.opcode_hash)

    #Compare deserialised serialised object to original
    def test_deserialise_serialise(self):
        a = self.dummy.clone()
        #print("[+] Converting to JSON")
        b = a.to_json(print_bbs=True)
        #print("Converted symbol b to JSON with all bbs")
        #print(a.vex)
        self.assertEqual(isinstance( b, str), True)

        #print("[+] Loading from to JSON")
        c = Symbol.from_json( b )
        #print("Converted JSON to symbol c with all bbs")
        #c.analyse("")
        #print(c.vex)

        """
        d = c.to_dict(print_bbs=True)

        bbs = d['bbs']

        abb = bbs[0]

        print(abb)

        for key in abb:
            #print("key: {}, type(key): {}".format(key, type(abb[key])))
            if key == 'vex':
                for vkey in abb[key]:
                    print("\tkey: vex.{}, type(key): {}".format(vkey, type(abb['vex'][vkey])))
                    print(abb['vex'][vkey])
                    if vkey == 'constants':
                        for cvkey in abb[key][vkey]:
                            print("\t\tkey: vex.constants.{}, type(key): {}".format(cvkey, type(abb['vex'][vkey][cvkey])))


        json.dumps( abb )
        sys.exit(-1)
        print("bbs == type {}".format(type(d['bbs']) ) )

        for key in d['bbs']:
            print("{} == type {}".format( key, type(key) ) )

        #for key in d['bbs']:
        #    print("bbs[ {} ] == type {}".format( key, type(d['bbs'][key]) ) )
            #print(json.dumps( a.vex[key], sort_keys=True, indent=2))
            #print(json.dumps( c.vex[key], sort_keys=True, indent=2))
            #if key == "temp_vars":
            #    print( c.vex[key] )
            #    #print( type( c.vex[key][0] ) )
            #self.assertEqual( c.vex[key], a.vex[key] )

        #print(c.to_json(print_bbs=True))
        print("Finished iterating vex")
        """

        """
        with open('/tmp/a.symb', 'w') as f:
            f.write( a.to_json(print_bbs=True) )
        with open('/tmp/c.symb', 'w') as f:
            f.write( c.to_json(print_bbs=True) )
        """

        self.assertEqual(a, c)

if __name__ == '__main__':
    unittest.main()
