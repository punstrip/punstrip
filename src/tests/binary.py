#!/usr/bin/python3
import sys
import pprint
import unittest
import subprocess
import binascii
import json
import hashlib
import logging
import IPython
import unittest
import context

from classes.symbol import Symbol
from classes.binary import Binary
from classes.config import Config
from classes.symbex import SymbEx




class TestBinary(unittest.TestCase):
    config = Config()
    logger = config.logger

    def test_binary_creation(self):
        a = Binary(self.config, path=self.config.corpus + "/new_bins/dynamic/clang/o2/shred")
        self.assertEqual(a.name, "shred")

    def test_binary_analyse(self):
        a = Binary(self.config, path=self.config.corpus + "/new_bins/dynamic/clang/o2/shred")
        a.analyse()
        self.assertEqual(len(a.symbols) + len(a.dyn_imports), 193)

    def test_binary_symbex(self):
        a = Binary(self.config, path=self.config.corpus + "/new_bins/dynamic/clang/o2/shred")
        #a = Binary(self.config, path=self.config.res + "/libnative-lib.so")
        a.analyse()
        symbex = SymbEx(self.config, a)
        IPython.embed()

    def test_binary_reaching_definitions(self):
        #a = Binary(self.config, path=self.config.corpus + "/new_bins/dynamic/clang/o2/shred")
        a = Binary(self.config, path=self.config.res + "/libnative-lib.so")
        a.analyse()
        symbex = SymbEx(self.config, a)
        IPython.embed()
 
 
 
 
    def test_binary_loading(self):
        a = Binary(self.config, path=self.config.corpus + "/bin/dynamic/gcc/og/who")
        bin_data = a.data

        hash_sha256 = hashlib.sha256()
        hash_sha256.update(bin_data)
        hash = hash_sha256.hexdigest()

        #check hash is equal to full file hash
        self.assertEqual(hash, a.sha256())
        sso = subprocess.check_output("sha256sum {} | cut -f 1 -d' ' | tr -d '[:space:]'".format(a.path), shell=True).decode('utf-8')

        #check hash is equal to sha256sum shell command for file
        self.assertEqual(sso, a.sha256())


    def test_extract_symbols_from_symtab(self):
        a = Binary(path=cfg.corpus + "/bin/dynamic/gcc/og/who")
        a.objdump_extract_symbols_from_symtab()
        self.assertNotEqual(len(a.symbols), 0)

        b = Binary(path=cfg.corpus + "/bin-stripped/dynamic/gcc/og/who")
        b.objdump_extract_symbols_from_symtab()
        self.assertEqual(len(b.symbols), 0)

    """
    def test_basic_blocks(self):
        a = Binary(path=cfg.corpus + "/bin/static/gcc/og/who")
        a.analyse_basic_blocks()
        self.assertNotEqual(len(a.bbs), 0)
    """
    """
    def test_r2_stripped_symbol_extraction(self):
        a = Binary(path=cfg.corpus + "/bin-stripped/static/gcc/og/who")
        #b = Binary(path="/root/friendly-corpus/bin/static/gcc/o0/plain/who")
        a_stripped_symbols = a.r2_extract_stripped_symbols(2)
        #b_stripped_symbols = b.r2_extract_stripped_symbols(2)

        #self.AssertEqual(len(a_stripped_symbols), len(b_stripped_symbols))
    """
    def test_nucleus_stripped_symbol_extraction(self):
        a = Binary(path=cfg.corpus + "/bin-stripped/static/gcc/og/who")
        a.nucleus_extract_symbols()

        self.assertGreater(len(a.symbols), 0)

    def test_taint_analysis(self):
        config = Config()
        config.logger.setLevel(logging.INFO)
        config.database.collection_name = "malware"

        #b = Binary(config, path=config.res + "/zzuf")
        #b = Binary(config, path=config.res + "/libnative-lib.so")
        #b = Binary(config, path=config.res + "/lcrack.clang.default", must_resolve_libs=True)
        b = Binary(self.config, path="/root/mirai.orig")
        b.analyse()
        IPython.embed()
        #flows, tracked, stack_vars, heap_vars, stack_args, t_stack_vars, t_heap_vars, t_stack_args, t_code_locs = b.taint_and_track_symb(b.symbols[16])

        #flows = b.taint_func_args_flows(b.symbols[1])
        print("Performing taint analysis")
        b.taint_func_flows()
        print("Finished analysisng taint flows")
        IPython.embed()

if __name__ == '__main__':
    unittest.main()
