#!/usr/bin/python3
import sys, os, re
import logging
import glob, gc
import progressbar
import traceback
from tqdm import tqdm
import lief
import json
import IPython

#from concurrent.futures import ThreadPoolExecutor
from multiprocess import Pool
import multiprocess
from functools import partial
from itertools import repeat

import context
from classes.config import Config
from classes.binary import Binary, MissingLibrary, UnsupportedISA, UnsupportedLang, StrippedBinaryError
from classes.symbol import Symbol
from classes.database import Database, PostgresDB
from classes.basicblock import LiveVariables, TaintTracking
from classes.basicblocksequence import LiveVariableAnalysis, ConstantPropagation, TaintPropagation
from classes.symbolic_execution_engine import SymbolicExecutionEngine
from classes.dwarf import DwarfInfo
import classes.utils

from joblib import Parallel, delayed

FAST_MODE = False

def resolve_shared_objects(db, bin_path):
    try:
        ##add library path
        b = classes.binary.Binary(db.config, path=bin_path, must_resolve_libs=False)
        curr = db.conn.cursor()
        for lib in b.libs:
            ##clean lib name
            lib_name = b._clean_lib_name(lib)
            b.logger.info("Dynamically linked library: {} -> {}".format(lib, lib_name))
            lib_match_re = lib_name + r'[^a-zA-Z]*\.so.*'

            lib_id = db.resolve_library(curr, lib_match_re)
            if lib_id: 
                db.logger.info("Resolved {} to {}".format(lib_name, lib_id))
            else:
                db.logger.error("Failed to resolve {} with `{}`".format(lib_name, lib_match_re))

        db.logger.debug("Finished with binary {}".format(bin_path))

    except (UnsupportedISA, UnsupportedLang, StrippedBinaryError) as e:
        db.logger.warning(e.stderror)
        return

    except Exception as e:
        db.logger.error(e)
        IPython.embed()
        raise e

def is_elf(fname):
    """
    Determine is a file is an ELF executable
    """
    try:
        elf = lief.parse( fname )
        return True
    except lief.bad_file:
        #not an ELF file
        pass
    return False

#recursively find binaries and analyse symbols
def scan_directory_update(config, d):
    db = classes.database.PostgresDB(config)
    db.connect()
    bins = set([])
    lib_re = re.compile(r'^.*\.so\.*.*$')
    obj_re = re.compile(r'^.*\.[oa]\.*.*$')

    g = None
    if os.path.isdir(d):
        g = glob.iglob(d + '/**/*', recursive=True)
    elif os.path.isfile(d):
        g = glob.iglob(d + '*', recursive=True)
    else:
        raise Exception("Unknown path `{}`".format(d))

    for f in g:
        try:
            if os.path.isdir(f):
                continue

            if re.match(obj_re, f):
                config.logger.debug("Skipping ELF object file {}".format(f))
                continue

            if re.match(lib_re, f):
                config.logger.debug("Skipping library {}".format(f))
                continue

            statinfo = os.stat(f)
            if statinfo.st_size == 0:
                continue

            if not is_elf(f):
                config.logger.warning("{} is not an ELF file! Skipping...".format(f))
                continue

            #make executable
            #if not os.access(f, os.X_OK):
            #        #mode in octal
            #        os.chmod(f, 0o755)

            config.logger.info("Analysing binary {}".format(f))

            #single threaded
            resolve_shared_objects(db, f)
        except Exception as e:
            config.logger.error(e)
            raise e
            pass

        config.logger.info("Finished analysing {}".format(d))

if __name__ == "__main__":
    config = Config()
    config.logger.setLevel(logging.DEBUG)
    #b = Binary(config, path='/dbg_elf_bins/libgetdns10/usr/lib/x86_64-linux-gnu/libgetdns.so.10.0.1', must_resolve_libs=False)
    #s = b.analyse_symbol_fast('getdns_dict_util_set_string')
    #IPython.embed()
    #sys.exit()

    config.logger.info("[+] Analsing binaries in {} with {} processes".format(sys.argv[1], config.analysis.THREAD_POOL_THREADS) )
    scan_directory_update(config, sys.argv[1])

