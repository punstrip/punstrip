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
import psycopg2
import timeout_decorator

#from concurrent.futures import ThreadPoolExecutor
from multiprocess import Pool
import multiprocess
from functools import partial
from itertools import repeat

import context
from classes.config import Config
from classes.binary import Binary, MissingLibrary, UnsupportedISA, UnsupportedLang, StrippedBinaryError, FunctionTooLarge, BinaryTooLarge
from classes.symbol import Symbol
from classes.database import Database, PostgresDB
from classes.basicblock import LiveVariables, TaintTracking, NoNativeInstructionsError
from classes.basicblocksequence import LiveVariableAnalysis, ConstantPropagation, TaintPropagation
from classes.symbolic_execution_engine import SymbolicExecutionEngine
from classes.dwarf import DwarfInfo
import classes.utils

from joblib import Parallel, delayed

FAST_MODE = False

#1 1/2 hour timeout
@timeout_decorator.timeout(5400)
def custom_analyse_library(c, bin_path):
    try:
        b = Binary(c, path=bin_path, must_resolve_libs=False)
        #print("Binary opened for analysis as b")
        #IPython.embed()
        #sys.exit()

        db = classes.database.PostgresDB(c)
        db.connect()

        ##add library path
        lib_id = db.add_library(bin_path)
        lib_symbols = b.analyse_fast(library=True)
        #s = b.analyse_symbol_fast('mad_dump_portcapmask')
        #print("Finished analysing symbol grib_md5_add")
        if not lib_symbols:
            db.conn.commit()
            return

        cur = db.conn.cursor()
        for P in tqdm(lib_symbols, desc="Database insertion"):
            cur.execute("""
                INSERT INTO library_prototypes (library, name, real_name, arguments, heap_arguments, local_stack_bytes, num_args, return, tls_arguments)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, ( lib_id,
                P['name'], P['real_name'], json.dumps(P['args']), 
                json.dumps(P['heap_args']), P['local_stack_bytes'],
                P['num_args'], P['ret'], json.dumps(P['tls_args']) )
            )
        db.conn.commit()

    except (UnsupportedISA, UnsupportedLang, StrippedBinaryError, BinaryTooLarge,
            FunctionTooLarge, NoNativeInstructionsError) as e:
        c.logger.error(e.stderror)
        return

    except psycopg2.DatabaseError as e:
        c.logger.error(e.stderror)
        raise e

    except Exception as e:
        c.logger.error(e)
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

            if f == '/dbg_elf_bins/libblocksruntime0/usr/lib/x86_64-linux-gnu/libBlocksRuntime.so.0.0.0':
                continue

            if re.match(obj_re, f):
                config.logger.debug("Skipping ELF object file {}".format(f))
                continue

            if not re.match(lib_re, f):
                config.logger.debug("Skipping {}".format(f))
                continue

            statinfo = os.stat(f)
            if statinfo.st_size == 0:
                continue

            if statinfo.st_size > 1024 * 1024 * 52:
                config.logger.error("Not analysing >52MB binary")
                continue

            if not is_elf(f):
                config.logger.warning("{} is not an ELF file! Skipping...".format(f))
                continue

            if db.library_id(f):
                #already analysed
                config.logger.warning("{} is already in the database! Skipping...".format(f))
                continue

            #make executable
            #if not os.access(f, os.X_OK):
            #        #mode in octal
            #        os.chmod(f, 0o755)

            config.logger.info("Analysing binary {}".format(f))

            #single threaded
            if not FAST_MODE:
                custom_analyse_library(config, f)
            else:
                bins.add(f)
        except Exception as e:
            config.logger.error(e)
            raise e
            pass

    if FAST_MODE:
        res = Parallel(n_jobs=32)(
                delayed(custom_analyse_library)(f) for f in bins
                )
        config.logger.info("Finished analysing {}".format(d))

if __name__ == "__main__":
    config = Config(level=logging.INFO)
    #b = Binary(config, path='/dbg_elf_bins/libgetdns10/usr/lib/x86_64-linux-gnu/libgetdns.so.10.0.1', must_resolve_libs=False)
    #s = b.analyse_symbol_fast('getdns_dict_util_set_string')
    #IPython.embed()
    #sys.exit()

    if len(sys.argv) == 3:
        config.logger.info("Enabling DEBUG output")
        config.logger.setLevel(logging.DEBUG)

    config.logger.info("[+] Analsing binaries in {} with {} processes".format(sys.argv[1], config.analysis.THREAD_POOL_THREADS) )
    scan_directory_update(config, sys.argv[1])

