#!/usr/bin/python3
import sys
import os
import glob, time, pprint
import argparse
import progressbar
import logging
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "../classes/")
from binary import Binary
from database import Database
from symbol import Symbol

import perform_analysis as analysis

NUM_THREADS = 32

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

pbar_config = [ '[ ', progressbar.Counter(), '] ', ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
                'Wheels: ', progressbar.AnimatedMarker(markers='◐◓◑◒'),
            ' (', progressbar.ETA(), ') ',
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Analyse binary with DESYL"
    )

    parser.add_argument(
        "-b", "--binary",
        type=str,
        required=True,
        help="The path to the binary."
    )

    parser.add_argument(
        "-c", "--collection_name",
        type=str,
        required=True,
        help="The MongoDB collection name to save analysis to."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        nargs="?",
        default="a.symbs",
        help="The symbol output file to be used with desyl-dress. "
             + "The default output file is './a.symbs'."
    )

    parser.add_argument(
        "-v", "--verbose",
        default=False,
        action='store_true',
        help="Makes verbose output."
    )

    args = parser.parse_args()
    return args

def write_symbols(symbols, filename):
    with open(filename, 'w') as f:
        for symb in symbols:
            f.write(symb.name + "() *" + str(hex(symb.vaddr)) + "^" + str(symb.size) + "\n")

def check_binary_in_database(db, bin_path):
    res = db["temp_{}".format( os.path.basename(bin_path) ) ].count()
    return res > 0

def get_vectorised_symbols(db, COL_NAME):
    res = db[COL_NAME].find({})
    symbs = []
    for symb in res:
        symbs.append( { 'symb_id' : symb['symb_id'], 'vex_vector': symb['vex_vector'] } )
    return symbs

def infer_symbols(db, path, collection_name):
    logger.debug("[+] Loading symbols from database...")
    b = Binary.fromDatabase(db, path, collection_name)

    #print(b.symbols)
    train_query = { 'type': 'symtab' , 'size' : { '$lt' : 5000 } }
    test_symbs = list( filter( lambda x: x.size < 5000, b.symbols) )

    logger.debug("Inferring {} symbols...".format(len(test_symbs)))

    inferred = analysis.infer_symbols_map_reduce_threaded(test_symbs, train_query)
    logger.info("Inferred {} symbols".format( len(inferred) ))
    assert( len(test_symbs) == len(inferred) )

    clf = analysis.get_model('/root/desyl/res/nn.model')
    confident_inferred = []

    for i in range(len(inferred)):
        confident_correct = clf.predict([ inferred[i][3] ])
        if confident_correct:
            confident_inferred.append( inferred[i][0] )

    logger.info("Inferred {} confident symbols".format( len(confident_inferred) ))
    return confident_inferred

def infer_symbols_vector(symbol_vectors, collection_name):
    train_query = {}
    test_symbs = symbol_vectors

    logger.debug("Inferring {} symbols...".format(len(test_symbs)))

    inferred = analysis.infer_symbols_vector_map_reduce_threaded(collection_name, test_symbs, train_query)
    #inferred = analysis.infer_symbols_map_reduce_vector(collection_name, test_symbs, train_query) 
    #symbols = []
    symbols = inferred

    db_ref = Database()
    db = db_ref.client

    #for s in inferred:
    #    pprint.pprint(s)
    #    #symbols.append( Symbol.fromDatabase(db, "symbols", s['symbol_id']) )

    logger.info("Inferred {} symbols".format( len(symbols) ))
    #assert( len(test_symbs) == len(inferred) )
    pprint.pprint(symbols)
    return symbols


#recursively find binaries and analyse symbols
def analyse_binary_from_db(f, output_fname, collection_name, db):
    if not os.path.isfile(f):
        raise Exception("[-] Error, {} is not a directory.".format(f))

    logger.info("Analysing {}...".format(f))

    b = Binary.fromDatabase(db, f, collection_name)
    logger.debug("Starting symbol analysis...")
    b.nucleus_extract_cfg()
    logger.info("Analysis complete. {} symbols found.".format( len(b.symbols) ))
    #print("[+] Saving symbols to collection {}.".format( test_name ))
    #b.save_symbols_to_db(db)




#recursively find binaries and analyse symbols
def analyse_binary(f, output_fname, coll_name, db):
    if not os.path.isfile(f):
        raise Exception("[-] Error, {} is not a directory.".format(f))

    logger.info("Analysing {}...".format(f))

    b = Binary(path=f,collection_name=coll_name)
    logger.debug("Starting symbol analysis...")
    b.analyse()

    """
    logger.info("Analysis complete. {} symbols found.".format( len(b.symbols) ))
    #b.nucleus_extract_symbols()
    logger.info("Finished symbol analysis...")
    logger.info("Extracting CFG...")
    b.nucleus_extract_cfg()
    logger.info("Generating Callgraph...")
    b.calculate_callgraph()
    logger.debug("Done!...")
    b.fill_symbols_callgraph()
    sys.exit(-1)
    for sym in b.symbols:

        sym.analyse(b.r2_hdlr)
        logger.info("Analysed {}".format( sym.name))
        #logger.debug( sym.to_str_full() )
    """
    """
    print("[+] Saving symbols to collection {}.".format( coll_name ))
    for sym in b.symbols:
        #sym.save_to_db(db, coll_name)
        print(sym.to_str_full())
    """

if __name__ == "__main__":

    args = get_args()


    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #ch = logging.StreamHandler()
    #ch.setLevel(logging.DEBUG)
    #logger.addHandler(ch)

    if args.verbose:
        #logger.setLevel(logging.DEBUG)
        logger.info("Starting symbol analysis for {}.".format(args.binary))
        logger.info("Output symbols file to be saved to {}.".format(args.output))

    db_ref = Database()
    db = db_ref.client


    #test_name = "temp_" + str(time.time()).split(".")[0] + "_{}".format( os.path.basename(f) )
    #test_name = "temp_{}".format( os.path.basename(args.binary) )

    #if not check_binary_in_database(db, args.binary):
    #    print("[!] No symbols for {} in the database. Analysing binary...".format(args.binary))
    #analyse_binary_from_db(args.binary, args.output, "unique_symbols",  db)
    analyse_binary(args.binary, args.output, args.collection_name, db)


    sys.exit(-1)
    symbols_to_infer = get_vectorised_symbols(db, "analysis_tmux")

    #inferred = infer_symbols_vector(db, args.binary, test_name)
    inferred = infer_symbols_vector(symbols_to_infer, "analysis")
    logger.info("Saving symbols to {}.".format(args.output))
    write_symbols( inferred, args.output)

    #executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
    #with progressbar.ProgressBar(widgets=pbar_config,max_value=0) as pbar:
        
    #     executor.shutdown(wait=True)
    #     pbar.value = pbar.max_value
    #    pbar.finish()
