#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import r2pipe
import pickle
import IPython

import context
import classes.utils
from classes.config import Config
from classes.binary import Binary
from classes.NLP import NLP


def get_args():
    parser = argparse.ArgumentParser(
        description="Compare symbols from two binaries."
    )

    parser.add_argument(
        "-b", "--bin",
        type=str,
        help="The path of the original binary."
    )
    parser.add_argument(
        "-u", "--unstripped-bin",
        type=str,
        help="The path of the unstripped binary."
    )

    parser.add_argument(
            "-o", "--output",
            type=str,
            help="Path to output this tools debug files for {in,}correct,skipped symbols"
    )
    
    args = parser.parse_args()
    return args

def compare_func_symbol_names(sym_list_a, sym_list_b, ofname, ofname_app):
    symbols_processed = 0
    symbols_skipped = 0
    correct_symbols = 0
    incorrect_symbols = 0


    correct_symbs = []
    incorrect_symbs = []
    skipped_symbs = []



    for symbol in sym_list_a:
        name = symbol.name
        address = symbol.vaddr
        size = symbol.size
        symb_type = symbol.type

        if symb_type != 'FUNC':
            symbols_skipped += 1
            skipped_symbs.append(symbol)
            continue

        symbols_processed += 1

        symbols_at_vaddr = list( filter( lambda x: x.vaddr == address, sym_list_b) )
        if len(symbols_at_vaddr) > 1:
            raise Exception("Error, multiple symbols at address: {} for symbol {}".format(address, name))

        common_names = list(filter( lambda x: nlp.check_word_similarity(x.name, name), symbols_at_vaddr))
        #common_names = filter( lambda x: check_similarity_of_symbol_name(x.name, name), sym_list_b)
        #common_names = filter( lambda x: x.name == name, sym_list_b)
        if len( common_names ) > 0:
            print("[+] Match! - " + name + " -> " + common_names[0].name + " :: " + str(address) + " :: " + str(size)) 
            correct_symbols += 1
            correct_symbs.append(symbol)
        else:
            incorrect_symbols += 1
            incorrect_symbs.append(symbol)

    if isinstance(ofname, str):
        oc_f = open(ofname + ofname_app + "_correct.syms", 'w')
        ic_f = open(ofname + ofname_app + "_incorrect.syms", 'w')
        sk_f = open(ofname + ofname_app + "_skipped.syms", 'w')

        oc_f.write( json.dumps( list(map(lambda x: x.to_str_custom(['name', 'vaddr', 'size']), correct_symbs)), indent=4) )
        ic_f.write( json.dumps( list(map(lambda x: x.to_str_custom(['name', 'vaddr', 'size']), incorrect_symbs)), indent=4) )
        sk_f.write( json.dumps( list(map(lambda x: x.to_str_custom(['name', 'vaddr', 'size']), skipped_symbs)), indent=4) )

        ic_f.close()
        oc_f.close()
        sk_f.close()

    print("//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\\\\")
    print("\t\tSummary")
    print("Symbols Processed (func): " + str(symbols_processed))
    print("Symbols Skipped (non-func): " + str(symbols_skipped))
    print("Correct Symbols: " + str(correct_symbols))
    print("Incorrect Symbols: " + str(incorrect_symbols))

    tp = correct_symbols
    tn = 0
    #fp = incorrect_symbols + symbols_skipped
    fp = len(sym_list_b) - correct_symbols
    fn = len(sym_list_a) - correct_symbols

    if tp > 0:
        precision   = tp / float(tp + fp)
        recall      = tp / float(tp + fn)
        f1 = 2.0 * (precision * recall)/(precision + recall)
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    logger.info("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
    logger.info("\t[+] Precison : {}, Recall : {}, F1 : {}".format( precision, recall, f1))



    print("\\\\-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-//")


if __name__ == '__main__':
    args = get_args()

    config = Config()
    logger = config.logger
    #logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    if not isinstance(args.bin, str) or not isinstance(args.unstripped_bin, str):
        print("Usage: eval_symbol_prediction.py -b orig_bin -u unstripped_bin")
        sys.exit(-1)

    nlp = NLP(config)
    orig_bin = Binary(config, path=args.bin)
    inferred_bin = Binary(config, path=args.unstripped_bin)

    orig_bin.r2_extract_symbols()
    inferred_bin.r2_extract_symbols()

    #orig_symbols = uniq_symb_list( orig_bin.symbols )
    #inferred_symbols = uniq_symb_list( inferred_bin.symbols )


    print("[+] Symbols in binary: " + str(len(orig_bin.symbols)))
    print("[+] Symbols in unstripped binary: " + str(len(inferred_bin.symbols)))

    print("[+] binary | <= unstripped binary")
    compare_func_symbol_names(orig_bin.symbols, inferred_bin.symbols, args.output, ".bin--unstripped-bin")
    #print("[+] unstripped binary | <= binary")
    #compare_func_symbol_names( inferred_bin.symbols, orig_bin.symbols, args.output, ".unstripped-bin--bin")
