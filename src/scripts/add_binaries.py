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

#1 1/2 hour timeout
@timeout_decorator.timeout(5400)
def analyse_binary(c, bin_path):
    try:
        b = Binary(c, path=bin_path, must_resolve_libs=True)
        #print("Analysing binary, exiting after shell")
        #IPython.embed()
        #sys.exit(1)

        db = classes.database.PostgresDB(c)
        db.connect()

        bin_id = db.add_binary(b)
        b.analyse(SSE_ANALYSIS=True)
        #s = b.analyse_symbol_fast('mad_dump_portcapmask')
        #print("Finished analysing symbol grib_md5_add")

        imp_funcs = list(map(lambda x: x['name'], b.dyn_imports))

        cur = db.conn.cursor()
        cur.execute("UPDATE public.binary SET dynamic_imports = %s WHERE id = %s", (json.dumps(imp_funcs), bin_id))
        for s in tqdm(b.symbols, desc="Database insertion"):
            s.vex['constants'] = list(s.vex['constants'])
            cur.execute("""
                INSERT INTO binary_functions (binary_id, name, real_name, arguments, heap_arguments, local_stack_bytes, num_args, return, tls_arguments, closure, sha256, opcode_hash, asm_hash, size, binding, vex, callers, callees, cfg, tainted_flows)
                VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                );
            """, ( bin_id,
                s.name, s.real_name, json.dumps(s.arguments), 
                json.dumps(s.heap_arguments), s.local_stack_bytes,
                s.num_args, "UNKNOWN", json.dumps(s.tls) ,
                json.dumps(s.closure), s.sha256(), s.opcode_hash, s.hash,
                s.size, s.binding, json.dumps(s.vex), json.dumps(list(s.callers)), 
                json.dumps(list(s.callees)), classes.utils.nx_to_str(s.cfg),
                json.dumps(s.tainted_flows)
                )
            )
        db.conn.commit()

    except (UnsupportedISA, UnsupportedLang, StrippedBinaryError, BinaryTooLarge, MissingLibrary,
            FunctionTooLarge, NoNativeInstructionsError) as e:
        c.logger.error(e.stderror)
        return

    except psycopg2.DatabaseError as e:
        c.logger.error(e)
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

            if re.match(obj_re, f):
                config.logger.debug("Skipping ELF object file {}".format(f))
                continue

            if re.match(lib_re, f):
                config.logger.debug("Skipping ELF shared object {}".format(f))
                continue

            statinfo = os.stat(f)
            if statinfo.st_size == 0:
                continue

            if not is_elf(f):
                config.logger.warning("{} is not an ELF file! Skipping...".format(f))
                continue

            if db.binary_id(f):
                #already analysed
                config.logger.warning("{} is already in the database! Skipping...".format(f))
                continue

            #make executable
            #if not os.access(f, os.X_OK):
            #        #mode in octal
            #        os.chmod(f, 0o755)

            config.logger.info("Analysing binary {}".format(f))

            analyse_binary(config, f)
        except Exception as e:
            config.logger.error(e)
            raise e
            pass

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

