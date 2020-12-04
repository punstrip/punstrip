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
from classes.database import Database
from classes.basicblock import LiveVariables, TaintTracking
from classes.basicblocksequence import LiveVariableAnalysis, ConstantPropagation, TaintPropagation
from classes.symbolic_execution_engine import SymbolicExecutionEngine
from classes.dwarf import DwarfInfo
import classes.utils

from joblib import Parallel, delayed

config = Config()
logger = config.logger


SSE_ANALYSIS    = True
MUST_HAVE_LIBS  = True 
FAST_MODE       = False 
ANALYSE_LIBS    = True

global pool
global pbar
global t
global results

pbar_config = [ ' [ ',  progressbar.Counter(format='%(value)d / %(max_value)d'), ' ] ',  progressbar.Percentage(), ' [', progressbar.Timer(),
                                progressbar.Bar(), ' (', progressbar.ETA(), ')'
]


# Shortcut to multiprocessing's logger
def error(msg, *args):
        return multiprocess.get_logger().error(msg, *args)

class LogExceptions(object):
        def __init__(self, callable):
                self.__callable = callable

        def __call__(self, *args, **kwargs):
                try:
                        result = self.__callable(*args, **kwargs)

                except Exception as e:
                        # Here we add some debugging help. If multiprocessing's
                        # debugging is on, it will arrange to log the traceback
                        logger.error(traceback.format_exc())
                        error(traceback.format_exc())
                        # Re-raise the original exception so the Pool worker can
                        # clean up
                        raise

                # It was fine, give a normal answer
                return result


def r2_analyse_library(bin_path, coll_name):
        c = Config()
        try:
            b = Binary(c, path=bin_path, collection_name=coll_name, must_resolve_libs=False)
        except (UnsupportedISA, UnsupportedLang, StrippedBinaryError) as e:
            print(e.stderror)
            return

        lib_symbols = b.r2_extract_lib_funcs()
        if len(lib_symbols) == 0:
            print("{}::No symbols found!".format(bin_path))
            return
        N = 128
        db = Database(c)
        ##stop writing 50,000 symbols to database at the same time
        for _symbs in tqdm(classes.utils.chunks_of_size(lib_symbols, N), desc='Saving symbols to DB::{}'.format(coll_name), unit_scale=N):
            try:
                res = db.client[coll_name+db.config.database.symbol_collection_suffix].insert_many( _symbs )
            except Exception as e:
                print("Error occured during db insertion")
                print(e)
                IPython.embed()
                raise e


def custom_analyse_library(bin_path, coll_name):
        c = Config()
        try:
            b = Binary(c, path=bin_path, collection_name=coll_name, must_resolve_libs=False)

            db = classes.database.PostgresDB(c)
            db.connect()

            ##add library path
            lib_id = db.add_library(bin_path)
            lib_symbols = b.analyse_fast()

            cur = db.conn.cursor()
            for P in lib_symbols:
                cur.execute("""
                    INSERT INTO library_prototypes (library, name, real_name, heap_arguments, locals, num_args, return)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, ( lib_id,
                    P['name'], P['real_name'],
                    json.dumps(P['heap_arguments']), json.dumps(P['locals']), P['num_args'], P['ret'] )
                )
            db.conn.commit()

            """
            N = 128
            db = Database(c)

            ##stop writing 50,000 symbols to database at the same time
            for _symbs in tqdm(classes.utils.chunks_of_size(lib_symbols, N), desc='Saving symbols to DB::{}'.format(coll_name), unit_scale=N):
                res = db.client[coll_name+db.config.database.symbol_collection_suffix].insert_many( _symbs )
            """


        except (UnsupportedISA, UnsupportedLang, StrippedBinaryError) as e:
            print(e.stderror)
            return

        #except Exception as e:
        #    print("Error occured during db insertion")
        #    print(e)
        #    IPython.embed()
        #    raise e

def analyse_library(bin_path, coll_name):
        c = Config()
        try:
            b = Binary(c, path=bin_path, collection_name=coll_name, must_resolve_libs=False)
        except (UnsupportedISA, UnsupportedLang, StrippedBinaryError) as e:
            print(e.stderror)
            return

        lib_symbols = []
        try:
            prototypes = DwarfInfo.objdump_get_func_prototypes(b.path)
            for func in prototypes:
                lib_s = { 'path' : bin_path, 'name': func['name'], 
                        'bin_name': os.path.basename(bin_path),  
                        'params': func['params'],
                        'return': func['return']
                        }
                lib_symbols.append(lib_s)
        except Exception as e:
            print(e)
            return

        if len(lib_symbols) == 0:
            print("{}::No symbols found!".format(bin_path))
            return
        N = 128
        db = Database(c)
        ##stop writing 50,000 symbols to database at the same time
        for _symbs in tqdm(classes.utils.chunks_of_size(lib_symbols, N), desc='Saving symbols to DB::{}'.format(coll_name), unit_scale=N):
            try:
                res = db.client[coll_name + db.config.database.symbol_collection_suffix].insert_many( _symbs )
            except Exception as e:
                print("Error occured during db insertion")
                print(e)
                IPython.embed()
                raise e

def create_and_analyse(bin_path, coll_name, must_resolve_libs):
        try:

            c = Config()
            b = Binary(c, path=bin_path, collection_name=coll_name, must_resolve_libs=must_resolve_libs)
            print("[+] Starting binary analysis...")
            #b.analyse()
            #see = SymbolicExecutionEngine(b.config, b)
            #s = b.get_symbol('quotearg_buffer_restyled')
            #s = b.get_symbol('quotearg_buffer')
            #s = b.get_symbol('parse_long_options')
            #s = b.get_symbol('main')
            #s = b.get_symbol('version_etc_ar')
            #lva = LiveVariableAnalysis(s)
            #cpa = ConstantPropagation(s)
            #tpa = TaintPropagation(s)
            #func_args, heap_args, stack_vars, resolved = lva.analyse(see)
            #flows = tpa.analyse(b, func_args, resolved=resolved)
            #print(flows)
            #IPython.embed()
            #sys.exit()
            b.analyse(SSE_ANALYSIS=SSE_ANALYSIS)
            print("[+] Saving symbols to {}".format(coll_name))
            ##below is a hack as the config is deleted for multiprocessing
            db = Database(c, collection_name=coll_name)
            classes.utils._desyl_init_class_(b, c)
            b.save_symbols_to_db(db)

            #free memory
            del b
            del db
            gc.collect()

        except MissingLibrary as e:
            ##pass on missing library
            print(e)
            c.logger.error(e)

        except UnsupportedISA as e:
            #pass on unspoorted ISA
            print(e)
            c.logger.error(e)

        except UnsupportedLang as e:
            print(e)
            c.logger.error(e)

        except Exception as e:
            print(e)
            c.logger.error(e)
            #raise e


def bin_analysis_cb(r, bin_path):
        logger.info("Successfully completed create_and_analyse for {}".format(bin_path))
        logger.info(r)
        t.update(1)

def bin_analysis_err(e, bin_path):
        logger.error("Error occoured in create_and_analyse for binary {}".format(bin_path))
        logger.error(e)
        t.update(1)


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
def scan_directory_update(d, coll_name, existing_bins):
        bins = set([])
        lib_re = re.compile(r'^.*\.so\.*.*$')
        obj_re = re.compile(r'^.*\.[oa]\.*.*$')
        if not os.path.isdir(d):
                raise Exception("[-] Error, {} is not a directory.".format(d))
        for f in glob.iglob(d + '/**/*', recursive=True):
                try:

                    if f in existing_bins:
                            logger.info("Skipping {}, already in DB...".format(f))
                            continue

                    if re.match(obj_re, f):
                        print("Skipping ELF object file", f)
                        continue

                    if ANALYSE_LIBS:
                        if not re.match(lib_re, f):
                            print("Skipping", f)
                            continue
                    else:
                        ##not analysisng libs
                        if re.match(lib_re, f):
                            print("Skipping", f)
                            continue

                    if os.path.isdir(f):
                            continue

                    statinfo = os.stat(f)
                    if statinfo.st_size == 0:
                            continue


                    if not is_elf(f):
                            continue

                    #make executable
                    if not os.access(f, os.X_OK):
                            #mode in octal
                            os.chmod(f, 0o755)

                    logger.info("Adding binary {}".format(f))
                    #r = pool.apply_async(LogExceptions(create_and_analyse), (f, coll_name), callback=bin_analysis_cb, error_callback=bin_analysis_err)
                    #r = pool.apply_async(create_and_analyse, (f, coll_name, False), callback=partial(bin_analysis_cb, bin_path=f), error_callback=partial(bin_analysis_err, bin_path=f))
                    #r = pool.apply_async(create_and_analyse, (f, coll_name, False) )
                    #results.append( r )

                    #single threaded
                    if not FAST_MODE:
                        if ANALYSE_LIBS:
                            custom_analyse_library(f, coll_name)
                        else:
                            create_and_analyse(f, coll_name, MUST_HAVE_LIBS)
                    else:
                        bins.add(f)
                except Exception as e:
                    print(e)
                    raise e
                    pass

        if FAST_MODE:
            if not ANALYSE_LIBS:
                res = Parallel(n_jobs=32)(
                        delayed(create_and_analyse)(f, coll_name, MUST_HAVE_LIBS) for f in bins
                )
            else:
                res = Parallel(n_jobs=32)(
                        delayed(custom_analyse_library)(f, coll_name) for f in bins
                )
        #else:
        #    for f in bins:
        #        #create_and_analyse(f, coll_name, MUST_HAVE_LIBS)
        #        analyse_library(f, coll_name)
        logger.info("Finished analysing {}".format(d))

def tqdm_cb(*args, **kwargs):
        global t
        t.update(1)

def update_db(bins, coll_name, existing_bins):
        global pool
        global t

        t = tqdm(total=len(bins), desc='Binaries')

        for f in tqdm(bins):
                t.set_postfix(f=f.ljust(20)[-20:])
                if f in existing_bins:
                        logger.info("Skipping {}, already in DB...".format(f))
                        continue

                #make executable
                if not os.access(f, os.X_OK):
                        #mode in octal
                        os.chmod(f, 0o755)

                logger.info("Adding binary {}".format(f))
                r = pool.apply_async(LogExceptions(create_and_analyse), (f, coll_name, MUST_HAVE_LIBS), callback=partial(bin_analysis_cb, bin_path=f), error_callback=partial(bin_analysis_err, bin_path=f))
                #r = pool.apply_async(create_and_analyse, (f, coll_name, MUST_HAVE_LIBS), callback=partial(bin_analysis_cb, bin_path=f), error_callback=partial(bin_analysis_err, bin_path=f))
                #create_and_analyse(f, coll_name, True)

if __name__ == "__main__":
        global pool
        global t
        col_name = config.database.mongodb.collection_name
        if ANALYSE_LIBS:
            col_name = "libs_II"
            config.database.mongodb.collection_name = col_name

        results = []
        config.logger.setLevel(logging.INFO)
        #create db and set collection name
        db = Database(config)
        #db.collection_name = col_name 

        existing_binaries = set(db.distinct_binaries())

        #pool = Pool(config.analysis.THREAD_POOL_THREADS)
        #pbar = progressbar.ProgressBar(widgets=pbar_config,max_value=0)

        ##read file list from file
        #bins = classes.utils.read_file_lines(sys.argv[1])
        #libs = classes.utils.read_file_lines(sys.argv[2])

        logger.info("[+] Analsing binaries in {} with {} processes using DB collection {}".format( sys.argv[1], config.analysis.THREAD_POOL_THREADS, col_name) )
        #scan_directory_update(config.corpus + relpath, config.database.collection_name, pbar, all_bin_paths)
        scan_directory_update(sys.argv[1], col_name,
                existing_binaries)
        #update_db(bins, col_name, existing_binaries)

        #pool.close()
        #pool.join()

        """
        for r in results:
                while True:
                        r.wait(1) #1s timeout
                        if r.ready():
                                break
                        pbar.update()
        """

        #pbar.value = pbar.max_value
        #pbar.update()
        #pbar.finish()

