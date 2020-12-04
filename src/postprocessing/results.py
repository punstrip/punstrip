#!/usr/bin/python3
import os, sys, gc
import logging, math, resource
import numpy as np
import scipy, dill
import functools
import itertools
import copy, re, subprocess
import random
import json
import glob
from tqdm import tqdm
import IPython
import logging

from multiprocess.pool import Pool

import context
import classes.config
import classes.utils
import classes.NLP

global training_names

def check_sim(chunked_names):
    res = []
    for test_n in chunked_names:
        item = { 'target' :  test_n }
        for train_n in training_names:
            if nlp.check_word_similarity(test_n, train_n) > 0.0:
                CAN_INFER = True
                item['similar'] = train_n
                break
        res.append( item )

    return res


def create_confusion_matrix(data):
    pass

if __name__ == '__main__':
    #config = classes.config.Config(no_logging=True)
    config = classes.config.Config()
    config.logger.removeHandler(config.loggerFileHandler)
    config.logger.setLevel(logging.INFO)

    nlp = classes.NLP.NLP(config)

    name_to_index = classes.utils.load_py_obj(config, 'name_to_index')
    corpus_name_to_index = classes.utils.load_py_obj(config, 'corpus_name_to_index')

    global training_names
    training_names = set(name_to_index.keys())
    testing_only_names = set(corpus_name_to_index.keys()) - training_names


    #import IPython
    #IPython.embed()
    #nlp.check_word_similarity('camlsysprep_operation_tmp_files_entry', 'thisshouldnotbesimilar')
    #sys.exit()

    """

    #Finding the number of impossible symbols to infer produces 0 symbols. 
    #NLP matching too overreaching

    procs = 64
    chunked_names = classes.utils.n_chunks(list(testing_only_names), procs)
    p = Pool(processes=procs)
    res = p.map(check_sim, chunked_names)

    print("impossible to infer stored in res")
    IPython.embed()

    cannot_infer = set([])

    
    #for test_n in tqdm(testing_only_names):
    for test_n in testing_only_names:
        CAN_INFER = False
        for train_n in training_names:
            if nlp.check_word_similarity(test_n, train_n):
                print("{} -> {}".format(test_n, train_n))
                CAN_INFER = True
                break
        if not CAN_INFER:
            print("Could not infer {}".format(test_n))
            cannot_infer.add(test_n)

    IPython.embed()

    """

    #coudl do 3 charts for f1, p and r but they are all the same values
    f1s, acc = [], []

    with open(config.res + '/desyl.log', 'r') as f:
        file_contents = f.read()
        binaries = file_contents.split('[INFO] Using binary')
        for b in binaries:
            lines = b.split()
            print("Testing binary: {}".format(lines[0]))
            if not os.path.isfile(lines[0]):
                print("\tNot valid.")
                continue

            unseen_re = 'Original binary has (\d+) symbols, (\d+) knowns, (\d+) unknowns, and (\d+) symbols we have never seen before'
            m = re.search(unseen_re, b)
            if not m:
                print("Failed to infer")
                continue
            tot_symbols, known, unknown, unseen = list(map(lambda x: int(x), m.groups()))


            acc_re = 'TP : (\d+), TN : (\d+), FP : (\d+), FN : (\d+)' 
            m = re.search(acc_re, b)
            if not m:
                print("Failed to infer")
                continue
            tp, tn, fp, fn = list(map(lambda x: int(x), m.groups()))

            #fp = fp - unseen

            precision   = tp / float(tp + fp)
            recall = precision
            f1 = 2.0 * (precision * recall)/(precision + recall) 
            
            #print("Unseen: {}".format(unseen))
            #print("\t[+] TP' : {}, TN' : {}, FP' : {}, FN' : {}".format(tp, tn, fp, fn))
            #print("\t[+] Precison' : {}, Recall' : {}, F1' : {}".format( precision, recall, f1))

            print("\t[+] TP : {}, TN : {}, FP : {}, FN : {}".format(tp, tn, fp, fn))
            print("\t[+] Precison' : {}, Recall' : {}, F1' : {}".format( precision, recall, f1))
            acc.append(precision)
            f1s.append(f1)


    out_fname = '/tmp/histogram_data.json'
    with open(out_fname, 'w') as f:
        json.dump({'precision' : acc, 'f1': f1s}, f)

    sys.exit()

    num_bins = 20
    n, bins, patches = plt.hist(f1s, num_bins, facecolor='blue', alpha=0.5)
    plt.show()




    IPython.embed()
        
