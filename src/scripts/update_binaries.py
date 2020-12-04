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
from classes.binary import Binary, MissingLibrary, UnsupportedISA, UnsupportedLang, StrippedBinaryError, FunctionTooLarge
from classes.symbol import Symbol
from classes.database import Database, PostgresDB
from classes.basicblock import LiveVariables, TaintTracking, NoNativeInstructionsError
from classes.basicblocksequence import LiveVariableAnalysis, ConstantPropagation, TaintPropagation
from classes.symbolic_execution_engine import SymbolicExecutionEngine
from classes.dwarf import DwarfInfo
import classes.utils

from joblib import Parallel, delayed

def update_binary(c, bin_id, bin_path):
    try:
        b = Binary(c, path=bin_path, must_resolve_libs=True)

        db = classes.database.PostgresDB(c)
        db.connect()

        b.analyse()
        imp_funcs = list(map(lambda x: x['name'], b.dyn_imports))

        curr = db.conn.cursor()
        curr.execute("UPDATE public.binary SET dynamic_imports = %s WHERE id = %s", (json.dumps(imp_funcs), bin_id))
        db.conn.commit()

    except (UnsupportedISA, UnsupportedLang, StrippedBinaryError,
            FunctionTooLarge, NoNativeInstructionsError) as e:
        c.logger.warning(e.stderror)
        return

    except psycopg2.DatabaseError as e:
        c.logger.error(e)
        raise e

    except Exception as e:
        c.logger.error(e)
        IPython.embed()
        raise e


if __name__ == "__main__":
    config = Config(level=logging.INFO)

    if len(sys.argv) == 3:
        config.logger.info("Enabling DEBUG output")
        config.logger.setLevel(logging.DEBUG)

    pdb = classes.database.PostgresDB(config)
    pdb.connect()
    curr = pdb.conn.cursor()
    curr.execute("SELECT id, path FROM public.binary")
    for bin_id, path in tqdm(curr.fetchall()):
        update_binary(config, bin_id, path)

    config.logger.info("[+] Success! Finished.")
