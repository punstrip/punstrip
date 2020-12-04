#!/usr/bin/python3
import sys, os
import logging
import glob, gc
import progressbar
import traceback
import re
import r2pipe

#from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat

import context
from classes.config import Config
from classes.binary import Binary
from classes.symbol import Symbol
from classes.database import Database

cfg = Config()

logger = logging.getLogger( cfg.logger )
logging.basicConfig(level=logging.INFO , format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger.setLevel(logging.INFO)

NUM_PROCESSES = 10
global pool
global pbar
global results

pbar_config = [ ' [ ',  progressbar.Counter(format='%(value)d / %(max_value)d'), ' ] ',  progressbar.Percentage(), ' [', progressbar.Timer(),
                progressbar.Bar(), ' (', progressbar.ETA(), ')'
]


# Shortcut to multiprocessing's logger
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)

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



def create_zignatures(f):

    """
    compiler = "gcc" if "gcc" in f else "clang"
    linkage = "dynamic" if "dynamic" in f else "static"

    optimisation_re = r'/o([0-3g])/'
    print(f)
    op = re.search( optimisation_re, f)
    print(op.group(1))
    optimisation = op.group(1)
    """
    logger.info("Creating zignatures for {}".format(f))
    pipe = r2pipe.open(f, ["-2"])
    pipe.cmd("aaaa")
    pipe.cmd("zaF")
    #pipe.cmd("zos {}".format( cfg.corpus + "/zignatures." + linkage + "." + compiler + "." + optimisation + ".sdb" ))
    pipe.cmd("zos {}".format( f + ".sdb" ))
    pipe.quit()
    logger.info("Saved zignature file to {}".format(f + ".sdb"))

def zig_analysis_cb(res):
    global pbar
    #print("caa_cb: {}".format(res))
    pbar.value += 1
    pbar.update()

def zig_analysis_err(ex):
    logger.error("{} :: Error occoured in create_zignature".format(__file__))
    raise Exception(ex)

#recursively find binaries and analyse symbols
def scan_directory(d, coll_name, pbar):
    global pool
    global results

    if not os.path.isdir(d):
        raise Exception("[-] Error, {} is not a directory.".format(d))

    for f in glob.iglob(d + '/**/*', recursive=True):
        if not os.path.isfile(f) or not os.access(f, os.X_OK):
            continue

        pbar.max_value += 1
        pbar.update()

        r = pool.apply_async(LogExceptions(create_zignatures), (f, ), callback=zig_analysis_cb, error_callback=zig_analysis_err)
        results.append( r )

if __name__ == "__main__":
    global pool
    global pbar 
    global results

    results = []

    pool = Pool(NUM_PROCESSES)
    pbar = progressbar.ProgressBar(widgets=pbar_config,max_value=0)

    logger.info("[+] Analsing binaries in {} with {} processes".format( cfg.corpus, NUM_PROCESSES) )

    #scan_directory(cfg.corpus + "/bin/dynamic/clang/o1", "", pbar)
    #./zignatures.dynamic.clang.1.sdb

    #scan_directory(cfg.corpus + "/bin/dynamic", "", pbar)
    #scan_directory(cfg.corpus + "/bin/static", "", pbar)
    scan_directory(cfg.corpus + "/bin-armv7/", "", pbar)

    
    for r in results:
        while True:
            r.wait(1) #1s timeout
            if r.ready():
                break
            pbar.update()

    pbar.value = pbar.max_value
    pbar.update()
    pbar.finish()

