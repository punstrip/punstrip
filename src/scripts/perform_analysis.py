#!/usr/bin/env python3
import argparse
import json, pprint
import datetime, uuid
import logging
import os, sys, time
import pickle, copy
import math, random
import progressbar
import pystache
import binascii
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import itertools
from itertools import repeat
import enchant
import re
import gc
from numba import jit

import urllib.parse

import logging

from threading import Lock

from bson.code import Code
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from multiprocessing import Process, Value, Array

import context
import scripts.similarity
from classes.binary import Binary
from classes.basicblocksequence import BasicBlockSequence
from classes.database import Database
from classes.symbol import Symbol
from classes.config import Config
import classes.utils
import classes.pmfs
import scripts.zignature_analysis as ZA

global _KNOWN_SYMBOLS
_KNOWN_SYMBOLS = None

global name_to_index
global index_to_name
global dim
global CONFIDENCE_ONLY

#pbar_config = [ '[ ', progressbar.Counter(), '] ', ' [', progressbar.Timer(), '] ',
#                progressbar.Bar(),
#            ' (', progressbar.ETA(), ') ',
#]

pbar_config = [ ' [ ',  progressbar.Counter(format='%(value)d / %(max_value)d'), ' ] ',  progressbar.Percentage(), ' [', progressbar.Timer(),
                progressbar.Bar(), ' (', progressbar.ETA(), ')' ]

def get_args():
    parser = argparse.ArgumentParser( description="Infer symbols in stripped binaries from other unstripped binaries.")
    #parser.add_argument( "-b", "--bins", type=str, help="The path of the original binaries.")
    #parser.add_argument( "-s", "--stripped-bins", type=str, help="The path of the unstripped binary.")
    parser.add_argument( "-n", "--bucket-size", type=int, help="The size of each bucket for cross-validation.")
    parser.add_argument( "-k", "--fold", type=int, help="K fold cross validation")
    parser.add_argument( "-o", "--output", type=str, help="Output file for infered symbols")
    
    args = parser.parse_args()
    return args

## @return [...] of bins. Each bin is a ( test_bin_names, train_bin_names ).
#@jit
def fill_buckets( max_buckets, bucket_size, bins):
    buckets = []
    for i in range(max_buckets):
        new_bins = list( copy.deepcopy( bins ) )
        test_data = []
        j = 0
        while j < bucket_size and ((i*bucket_size) + j) < len(bins):
            #print((i*bucket_size) + j)
            test_data.append( new_bins.pop( (i*bucket_size) ) )
            j+=1
        
        #logger.info(json.dumps( test_data ))
        #stripped test data, original training data
        bucket = (new_bins, test_data) 
        buckets.append(bucket)
    return buckets


def max_symtab_symbols(db, test_config):
    test_query = Database.gen_query(test_config)

    #project each symbol to _id:1 then sum
    sum_agg = { '$group' : {  '_id' : 1, 'count' : { '$sum' : 1} } } 
    test_query.append( sum_agg )

    logger.error(test_query)
    arr_res = db.run_mongo_aggregate(test_query)
    assert( isinstance(arr_res, list ) )
    assert(len(arr_res) == 1)

    count = arr_res[0]['count']
    return count
    

def __hide_known_criterion(symbol):
    #symbol.name = "fcn." + hex(symbol.size)
    symbol.path = symbol.path.replace("/bin/", "/bin-stripped/")
    return symbol

#Return the test symbols and the query for finding all of teh training symbols
def get_test_symbols(db, test_config):
    """
    Return a list of symbol instances for the query
    :param db: classes.database.Database instance
    :param test_config: Database config
    """
    test_query = Database.gen_query(test_config)
    #test_symbols = db.get_symbols('symbols_stripped', test_query)

    print(test_query)

    #Use known symbol starts and vaddr
    test_symbols = db.get_symbols('symbols', test_query)
    test_symbols = list(map(lambda x: __hide_known_criterion(x), test_symbols))

    return test_symbols

def get_number_of_symbols(db, config, collection_name='symbols'):
    """
        Return the number of symbols for a given config
        :param db: desyl.classes.database.Database instance
        :param config: dict config to give to the database for symbol selection
        :param collection_name: String collection name in database to use
        :return: The number of symbols for the config
        :rtype: int
    """
    config = config
    count = { '$count' : 'num_symbols' }

    query = Database.gen_query(config, projection={'_id':1})
    query.append( count )

    res = db.run_mongo_aggregate(query, collection_name=collection_name)
    for r in res:
        return r['num_symbols']
    return 0

def get_unique_symbol_names(db, config, collection_name='symbols'):
    """
        Return the names of all unique symbols for config
    """
    #config, projection= config
    proj_2 = { '$project': { 'name': '$name' } }
    groupby = { '$group' : { '_id' : '$name' } }

    query = Database.gen_query(config)
    query += [ proj_2 , groupby ]
    
    res = db.run_mongo_aggregate(query, collection_name=collection_name)
    return list(map(lambda x: x['_id'], res))


#Return the test symbols and the query for finding all of teh training symbols
def get_train_test_symbols(db, train_config, test_config):
    #train_config, train_projection = train_config
    #test_config, test_projection = test_config

    train_query = Database.gen_query(train_config)
    test_query = Database.gen_query(test_config)

    #print(test_query)
    #sys.exit()

    train_symbols = db.get_symbols('symbols', train_query)
    test_symbols = db.get_symbols('symbols_stripped', test_query)

    return (train_symbols, test_symbols)

#clf = load_model("RandomForestClassifier")
def load_model(name):
    model_fname = cfg.desyl + "/res/" + name + ".pickle" 
    print("[+] Loading model from {}".format(model_fname))
    with open(model_fname, 'rb') as f:
        return pickle.load(f)

#@jit
def _infer_symbol(known_symbols, unknown_sym):
    #Why pass in global known symbols? Because python multiprocessing. If using startmap and pepeating symbols. Uses 99GB RAM. This method uses 20GB and is faster.
    best_symbols = []
    #score, similarity vector, symbol
    worst_symb = (-1.0, [-1.0] * len(Symbol.similarity_weightings), {})
    NUM_TRACK = 5

    #init best_symbols[]
    best_symbols += [ worst_symb ]

    #map reduce style. iterate trough 
    for known_symbol in known_symbols:
        sim_vec = unknown_sym.similarity(known_symbol)
        score = float( sum(sim_vec) ) / len(BasicBlockSequence.similarity_weightings) #normalised  0.0 <= s <= 1.0
    
        #Ignore machine learning prediction
        """
        if clf.predict( sim_vec.reshape(1, -1) )[0] == 1:
            best_symbols.append( ( score, sim_vec, known_symbol) )
            best_symbols = list( reversed( sorted(best_symbols, key=lambda x: x[0]) ) )
        """

        #"""

        # First, store the best N symbols. 
        # Second, if the best L symbols are all named similar, return the symbol.
        if score > worst_symb[0]:
            best_symbols.append( (score, sim_vec, known_symbol) )
            best_symbols = list( reversed( sorted(best_symbols, key=lambda x: x[0]) ) )

            if len(best_symbols) > NUM_TRACK:
                best_symbols = best_symbols[:NUM_TRACK]
            worst_symb = best_symbols[-1]

            #score is already the best
            #if score >= sum( Symbol.similarity_weightings ): 
            #    break
        #"""

    #logger.info("[+] Symbol Inferred!")
    return best_symbols

def _infer_err(err):
    logger.error("ERROR INFERRING SYMBOL! {} {}".format( __file__, err))
    logger.error("YOU FUCKED UP!")

#thanks, now delete local copy of memory
def _infer_res(res):
    gc.collect()

def infer_symbol_pmf(unknown):
    #return unknown.compute_pmf( { 'hash' : 1 } )
    #return unknown.compute_pmf( { 'vex.ninstructions' : 1, 'vex.ntemp_vars' : 1 } )
    #return unknown.compute_pmf( { 'callees' : 1 } )
    #return unknown.compute_pmf( { 'size' : 1 } )
    #return unknown.compute_pmf( { 'vex' : 1, 'cfg': 1 } )
    #return unknown.compute_pmf( { 'vex' : 1, 'hash' : 1, 'opcode_hash': 1, 'vex.ntemp_vars' : 1, 'vex.ninstructions': 1, 'callees': 1, 'callers': 1, 'size': 1 } )
    return unknown.compute_pmf( { 'vex' : 1, 'hash' : 1, 'opcode_hash': 1, 'callees': 1, 'size': 1, 'cfg': 1, 'vex.ninstructions' : 1, 'vex.ntemp_vars': 1 } )

def py_infer_symbols(known, unknown):

    InfSymbPBar = progressbar.ProgressBar(widgets=pbar_config,max_value=len(unknown))

    """
    ### Single threaded
    inferred_symbols = []
    for symb in unknown:
        inferred_symbols.append( _infer_symbol(symb) )
    """

    #### Mulitprocess
    InfSymbPool = Pool(10)
    inferred_res = []
    inferred_symbols = []
    for stripped_symbol in unknown:
        res = InfSymbPool.apply_async(_infer_symbol, (known, stripped_symbol), callback=_infer_res, error_callback=_infer_err)
        inferred_res.append( res )

    for res in inferred_res:
        while True:
            InfSymbPBar.update()
            res.wait(1) #1s timeout
            if res.ready():
                inferred_symbols.append( res.get() )
                InfSymbPBar.value += 1
                break

    InfSymbPBar.value = InfSymbPBar.max_value
    InfSymbPBar.update()
    InfSymbPBar.finish()
 
    return inferred_symbols

#need to get at least 1/e % of names correct for similarity match
def check_similarity_of_symbol_name(correct_name, inferred_name):

        #correct name is a prefix or suffix of inferref name and vica versa
        if correct_name in inferred_name or inferred_name in correct_name:
            logger.debug("Matched on substring! {} -> {}".format(inferred_name, correct_name))
            return True

        #check edit distance as a function of max length
        levehstien_distance = nltk.edit_distance( correct_name, inferred_name )
        m = max( len(correct_name), len(inferred_name) )
        EDIT_DISTANCE_THRESHOLD = 1.0 / 2.0 
        if ( levehstien_distance / m ) <= EDIT_DISTANCE_THRESHOLD:
            logger.debug("Matched on edit distance! {} -> {}".format(inferred_name, correct_name))
            return True

        words_in_inferred_name = re.findall(r'[a-zA-Z]+', inferred_name)
        words_in_correct_name = re.findall(r'[a-zA-Z]+', correct_name)

        THRESHOLD = 1.0 / math.e #0.36 -> 1/2, 2/3, 2/4, 2/5, 3/6, ...
        #INT_THRESHOLD = max( math.ceil( len( words_in_inferred_name ) * THRESHOLD ),
        #                        math.ceil( len( words_in_correct_name ) * THRESHOLD ) )

        ##########################################
        enchant_lock.acquire()
        ##########################################
        try:
            #TODO: filter to uniue and WHOLE words! in a dictionary
            words_in_inferred_name = set( words_in_inferred_name )
            words_in_correct_name = set( words_in_correct_name )

            us_D = enchant.Dict("en_US")
            gb_D = enchant.Dict("en_GB")

            #filter for english words
            words_in_inferred_name = set( filter( lambda w: len(w) > 2 and (us_D.check(w) or gb_D.check(w)), words_in_inferred_name) )
            words_in_correct_name = set( filter( lambda w: len(w) > 2 and (us_D.check(w) or gb_D.check(w)), words_in_correct_name) )
            
            #remove boring stop words
            unique_words_inferred = words_in_inferred_name - set(nltk.corpus.stopwords.words('english'))
            unique_words_correct = words_in_correct_name - set(nltk.corpus.stopwords.words('english'))

            #calculate threshold after english word filter
            INT_THRESHOLD = max(1, max( math.ceil( len( words_in_inferred_name ) * THRESHOLD ),
                                math.ceil( len( words_in_correct_name ) * THRESHOLD ) ) )
            
            stemmer = nltk.stem.PorterStemmer()
            lemmatiser = nltk.stem.wordnet.WordNetLemmatizer()

            stemmed_inferred = set( map( lambda x: stemmer.stem(x), unique_words_inferred) )
            stemmed_correct = set( map( lambda x: stemmer.stem(x), unique_words_correct) )

            lemmatised_inferred = set( map( lambda x: lemmatiser.lemmatize(x), unique_words_inferred) )
            lemmatised_correct = set( map( lambda x: lemmatiser.lemmatize(x), unique_words_correct) )

            matched_lemmatised = lemmatised_inferred.intersection( lemmatised_correct )
            matched_stemmed = stemmed_inferred.intersection( stemmed_correct )

            ##CANNOT JSON DUMPS A SET
            if len( matched_lemmatised ) >= INT_THRESHOLD:
                    logger.debug("Matched on lemmatised: " + str( json.dumps( list(matched_lemmatised) ) ) )
                    logger.debug("\t{} -> {}".format( inferred_name, correct_name))
                    return True

            if len( matched_stemmed ) >= INT_THRESHOLD:
                    logger.debug("Matched on stemmed: " + str( json.dumps( list(matched_stemmed) ) ) )
                    logger.debug("\t{} -> {}".format( inferred_name, correct_name))
                    return True

            #Synonym matching is too broad and inaccurate. Temporarily disabled
            """
            ##### Match on synonyms of words
            #for all teh words in correct and inferred, pairwise match synonyms between them
            syn_cc = 0
            for inf_word in stemmed_inferred.union(lemmatised_inferred):
                for cor_word in stemmed_correct.union(lemmatised_correct):
                    inf_synonyms = set(itertools.chain.from_iterable([word.lemma_names() for word in wn.synsets(inf_word) ]))
                    cor_synonyms = set(itertools.chain.from_iterable([word.lemma_names() for word in wn.synsets(cor_word) ]))

                    #if there is a match on the synonyms, for each combination of words 
                    if len( cor_synonyms.intersection( inf_synonyms ) ) > 0:
                        syn_cc += 1
                        break

            if syn_cc >= INT_THRESHOLD:
                    logger.info("Matched on "+ str( syn_cc ) +" synonyms:")
                    logger.info("\t{} -> {}".format( inferred_name, correct_name))
                    return True
            """

        except Exception as e:
            logger.warn("[!] Could not compute check_similarity_of_symbol_name( {} , {} )".format(correct_name, inferred_name))
            logger.warn(e)

        finally:
            enchant_lock.release()

        return False

#Is the name of A the equivalent of B?
#Find the symbol that is called B in the binary of A
### A is unknown, stripped symbol, B is our guess, inferred, known training symbol
def check_inferred_symbol_name(db, A, name):
    assert(isinstance(A, Symbol))
    assert(isinstance(name, str))

    #print( { 'bin_name' : A.bin_name, 'name': B.name, 'vaddr': A.vaddr, 'linkage': A.linkage, 'compiler': A.compiler, 'optimisation': A.optimisation } )
    res = db.client.symbols.find_one( { 'bin_name' : A.bin_name, 'name': name, 'vaddr': A.vaddr, 'linkage': A.linkage, 'compiler': A.compiler, 'optimisation': A.optimisation } )

    if res != None:
        return True


    res = db.client.symbols.find_one( { 'bin_name' : A.bin_name, 'compiler': A.compiler, 'optimisation': A.optimisation, 'linkage': A.linkage, 'vaddr': A.vaddr } )
    if res != None:
        return check_similarity_of_symbol_name( res['name'], name )

    return False



#Is the name of A the equivalent of B?
#Find the symbol that is called B in the binary of A
### A is unknown, stripped symbol, B is our guess, inferred, known training symbol
def check_inferred_symbol(db, A, B):
    assert(isinstance(A, Symbol))
    assert(isinstance(B, Symbol))
    #print( { 'bin_name' : A.bin_name, 'name': B.name, 'vaddr': A.vaddr, 'linkage': A.linkage, 'compiler': A.compiler, 'optimisation': A.optimisation } )
    res = db.client.symbols.find_one( { 'bin_name' : A.bin_name, 'name': B.name, 'vaddr': A.vaddr, 'linkage': A.linkage, 'compiler': A.compiler, 'optimisation': A.optimisation } )

    if res != None:
        return True


    res = db.client.symbols.find_one( { 'bin_name' : A.bin_name, 'compiler': A.compiler, 'optimisation': A.optimisation, 'linkage': A.linkage, 'vaddr': A.vaddr } )
    if res != None:
        return check_similarity_of_symbol_name( res['name'], B.name )

    return False


def get_model(model_fname):
    with open(model_fname, 'rb') as f:
        return pickle.load(f)

#Is there a symbol in A's binary at A's location from the symtab?
def check_inferred_symbol_possible(db, A, training_symbols):
    #A is the inferred symbol. A comes from nucleus. Is there even a symbol at A's location in the original binary
    #1) Is there a valid symtab symbol in the binary at the location
    #res = db.symbols.find_one( { 'bin_name': A.bin_name, 'type': 'symtab', 'vaddr': A.vaddr } )
    unstripped_path = re.sub(r'bin-stripped', 'bin', A.path)
    res = db.client.symbols.find_one( { 'bin_name': A.bin_name, 'vaddr': A.vaddr, 'path' : unstripped_path  } )

    symb_in_bin_at_vaddr = res != None
    symbol_in_training_set = False

    correct_symbol_name = "{no symbol at this vaddr}"
    if symb_in_bin_at_vaddr:
        correct_symbol_name = res['name']

    #2) Is the true symbol in the training set?
    if symb_in_bin_at_vaddr:
        if correct_symbol_name in training_symbols:
            symbol_in_training_set = True
            logger.debug("Symbol name {} IS in the training set".format( correct_symbol_name ))

    return symb_in_bin_at_vaddr, symbol_in_training_set, correct_symbol_name

def _post_analysis(train_symbol_names, clf, test_symbol, inferred_symbol):
    db = Database()

    if inferred_symbol[0][0] != -1:
        correct = check_inferred_symbol( db, test_symbol, inferred_symbol[0][2])
        confidence = inferred_confidence(inferred_symbol)
        confident = clf.predict([ inferred_symbol[0][1] ])[0]
    else:
        correct = False
        confident = 0.0  
        confidence = 0.0

    theoretically_possible, in_training_set, correct_symbol = check_inferred_symbol_possible( db, test_symbol, train_symbol_names )

    del db
    return (correct, confident, theoretically_possible, confidence, correct_symbol, in_training_set)

def _post_analysis_pmf(train_symbol_names, clf, test_symbol, inferred_pmf):
    """
        Perform post-analysis checks. 
        Check if the symbol was possible to infer
        Check the confidence of our prediction
        Pass the inferred weightings into a ML predictors to determine confident
        Determine if our guess was correct

        :param train_symbol_names: A set of symbol names in the training data
        :param clf: A scikit learn ML model for predicting if we are confident in our guess
        :param test_symbol: The original unknown test symbol that was inferred
        :param inferred_pmf: An np vector representing a Probability Mass Function over the set of all symbols for this unknown symbol
        :return: A set of observations about this symbol inference. 
            returns ( is correct?, ML confident?, is theoretically_possible to infer?, statistical confidence, the correct symbol, was in training set? )
        :rtype: set
    """
    db = Database()

    if np.max(inferred_pmf) != 0.0:
        #print("Highest value in pmf: {}".format( np.max(inferred_pmf) ))
        inferred_symbol_name = index_to_name[ np.argmax( inferred_pmf ) ]
        correct = check_inferred_symbol_name( db, test_symbol, inferred_symbol_name)
        confidence = inferred_confidence_pmf(inferred_pmf)
        logger.info("Confidence: {}".format(confidence))

        #confident = clf.predict([ inferred_symbol[0][1] ])[0]
        confident = 0.0
    else:
        correct = False
        confident = 0.0  
        confidence = 0.0

    theoretically_possible, in_training_set, correct_symbol = check_inferred_symbol_possible( db, test_symbol, train_symbol_names )

    del db
    return (correct, confident, theoretically_possible, confidence, correct_symbol, in_training_set)


def _post_analysis_confidence_only_pmf(train_symbol_names, test_symbol, inferred_pmf):
    """
        Perform post-analysis checks. 
        Check if the symbol was possible to infer
    """
    if np.max(inferred_pmf) != 0.0:
        confidence = inferred_confidence_pmf(inferred_pmf)
    else:
        confidence = 0.0

    in_training_set = True if test_symbol.name in train_symbol_names else False
    return confidence, in_training_set


def threaded_post_analysis(test_symbs, inferred, train_symbol_names, clf):
    PA_pool = Pool(processes=32)
    PA_threads, PA_results = [], []

    pbar = progressbar.ProgressBar(widgets=pbar_config,max_value=len(inferred))
    #initial update
    pbar.value = 0
    pbar.update()
    for i in range(len(inferred)):
        if CONFIDENCE_ONLY:
            res = PA_pool.apply_async(_post_analysis_confidence_only_pmf, (train_symbol_names, test_symbs[i], inferred[i]))
        else:
            res = PA_pool.apply_async(_post_analysis_pmf, (train_symbol_names, clf, test_symbs[i], inferred[i]))

        PA_threads.append(res)

    for t in PA_threads:
        while True:
            pbar.update()
            t.wait(1)
            if t.ready():
                PA_results.append(t.get())
                pbar.value += 1
                break

    pbar.value = pbar.max_value
    pbar.update()
    pbar.finish()

    del PA_threads
    del PA_pool
    gc.collect()
    return PA_results


def print_bucket_results_pmf_confidence_only(inferred_pmfs, test_symbs, confidence_threshold, train_symbol_names, symbols_in_symtab):

    #inferred = [ [ (score, sim, symbol), ... ], ... ] 
    symb_print_proj = [ 'name', 'optimisation', 'bin_name', 'compiler', 'vaddr', 'path' ]
    symb_debug_proj = [ 'name', 'optimisation', 'bin_name', 'compiler', 'vaddr', 'path', 'hash', 'opcode_hash', '_id', 'cfg', 'callers', 'callees' ]
    similarity_file = open(cfg.desyl + '/res/similarities.json', 'a')

    logger.info("Performing threaded post analysis...")
    if len(inferred_pmfs) == 0:
        logger.info("Nothing inferred!")
        return

    s_confidence, s_in_training_set  = zip(*threaded_post_analysis(test_symbs, inferred_pmfs, train_symbol_names, None))

    assert( len(inferred_pmfs) == len(s_confidence) )
    assert( len(s_confidence) == len(s_in_training_set) )
    logger.info("Post analysis complete!")
    logger.info("Summing analysis!")


    #confident is a 0/1 yes no,
    #confidence is the difference between this symbol and others
    correct_count, confident_count, confident_correct_count, theoretical_count, training_set_count = 0, 0, 0, 0, 0
    confidence_count, confidence_correct_count = 0,0
    incorrect_score, correct_score, confident_correct_score, confident_score = 0.0, 0.0, 0.0, 0.0
    confidence_correct_score, confidence_score = 0.0, 0.0


    confident_inferred_symbs = []
    unknown_inferred_symbols = []

    for i in range(len(test_symbs)):
        confidence, in_training_set = s_confidence[i], s_in_training_set[i]

        #logger.debug("//==========================================\\\\")

        #assert( len( inferred[i][0] ) == 3 )
        #__score, __sim, __symbol = inferred[i][0]
        __score = np.max( inferred_pmfs[i] )
        assert( __score <= 1.0 )
        __symbol_name = index_to_name[ np.argmax( inferred_pmfs[i] ) ]


        if confidence > confidence_threshold:
            #logger.debug("CONFIDENCE!!!!   {} > {}".format( confidence, confidence_threshold))
            confidence_count += 1
            confidence_score += __score
            confident_inferred_symbs.append( [ test_symbs[i], __symbol_name, inferred_pmfs[i]  ] )
        else:
            unknown_inferred_symbols.append( [test_symbs[i], __symbol_name, inferred_pmfs[i] ] )
        
        if in_training_set:
            #logger.debug("[+] The correct symbol name WAS in the training set")
            training_set_count += 1
        else:
            #logger.debug("[+] The correct symbol name was NOT in the training set")
            pass

    #save to res
    classes.utils.save_py_obj(confident_inferred_symbs, "confident_inferred_symbs")
    classes.utils.save_py_obj(unknown_inferred_symbols, "unknown_inferred_symbols")
    return

 

 
def print_bucket_results(inferred, test_symbs, confidence_threshold, clf, train_symbols, symbols_in_symtab):

    #inferred = [ [ (score, sim, symbol), ... ], ... ] 
    symb_print_proj = [ 'name', 'optimisation', 'bin_name', 'compiler', 'vaddr', 'path' ]
    similarity_file = open(cfg.desyl + '/res/similarities.json', 'a')

    train_symbol_names = set( map( lambda x: x.name, train_symbols) )

    logger.info("Performing threaded post analysis...")
    if len(inferred) == 0:
        logger.info("Nothing inferred!")
        return

    #s_correct, s_confident, s_theoretical, s_confidence, s_correct_name, s_in_training_set  = zip(*threaded_post_analysis(test_symbs, inferred, train_symbol_names, clf))
    #s_confidence, s_in_training_set  = zip(*threaded_post_analysis(test_symbs, inferred, train_symbol_names, clf))

    #assert( len(inferred) == len(s_correct) )
    #assert( len(s_confidence) == len(s_correct) )
    #assert( len(s_theoretical) == len(s_correct) )
    logger.info("Post analysis complete!")

    #confident is a 0/1 yes no,
    #confidence is the difference between this symbol and others
    correct_count, confident_count, confident_correct_count, theoretical_count, training_set_count = 0, 0, 0, 0, 0
    confidence_count, confidence_correct_count = 0,0
    incorrect_score, correct_score, confident_correct_score, confident_score = 0.0, 0.0, 0.0, 0.0
    confidence_correct_score, confidence_score = 0.0, 0.0

    for i in range(len(test_symbs)):
        correct, confidence, confident, theoretically_possible, correct_name, in_training_set = s_correct[i], s_confidence[i], s_confident[i], s_theoretical[i], s_correct_name[i], s_in_training_set[i]

        logger.info("//==========================================\\\\")

        assert( len( inferred[i][0] ) == 3 )
        __score, __sim, __symbol = inferred[i][0]

        if __score == -1:
            logger.info("[+] Symbol could not be inferred.")
            logger.info( test_symbs[i].to_str_custom( symb_print_proj )  )
            if theoretically_possible:
                logger.info("[+] Symbol could theoretically be inferred.")
                theoretical_count += 1
            else:
                logger.info("[+] Theoretically impossible for symbol to be inferred i.e. No symbol at this vaddr (nucleus error).")
            logger.info("\\\\==========================================//")
        else:
            logger.info("[+] Similarity score: " + str( __score ))
            #logger.info("[+] Matched symbols: " + str( len( test_symbs ) ))
            logger.info("[+] Similarity vector: " + str( __sim.tolist() ))
            logger.info("[+] test symbol:")
            logger.info( test_symbs[i].to_str_custom( symb_print_proj )  )
            logger.info("[+] was inferred to --> training symbol:")
            logger.info( __symbol.to_str_custom( symb_print_proj ) )
            logger.info("[+] Is inferred symbol correct? " + str(correct))
            similarity_file.write( str( __sim.tolist()) + "\t[" + str(int(correct)) + "]" +"\n")

            logger.info("ML Classifier class: " + str( confident ) + "\t-> {}".format("YES" if confident == 1.0 else "NO" ) )
            logger.info("Confidence score: " + str( confidence ) + "\t-> {}".format("YES" if confidence > confidence_threshold else "NO" ) )

            if confidence > confidence_threshold:
                logger.info("CONFIDENCE!!!!   {} > {}".format( confidence, confidence_threshold))
                confidence_count += 1
                confidence_score += __score
            if confident:
                confident_count += 1
                confident_score += __score
            if correct:
                correct_count += 1
                correct_score += __score
                if confidence > confidence_threshold:
                    confidence_correct_count += 1
                    confidence_correct_score += __score
                if confident:
                    confident_correct_count += 1
                    confident_correct_score += __score

            else:
                incorrect_score += __score
            if theoretically_possible:
                logger.info("[+] Symbol could theoretically be inferred.")
                theoretical_count += 1
            else:
                assert(not correct and not in_training_set)
            
            if in_training_set:
                logger.info("[+] The correct symbol name WAS in the training set")
                training_set_count += 1
            else:
                logger.info("[+] The correct symbol name was NOT in the training set")

            logger.info("The correct symbol name was '{}'".format( correct_name ))
            logger.info("\\\\==========================================//")


    logger.info("[+] Inferred {} of {} correctly from a maximum thoretical of {}.".format( str(correct_count), str(len(inferred)), str(theoretical_count)))
    logger.info("[+] Inferred {} of {} correctly from {} in symtab.".format( str(correct_count), str(len(inferred)), str(symbols_in_symtab)))

    counts = ( correct_count, confident_count, confident_correct_count, theoretical_count, training_set_count, confidence_count, confidence_correct_count )
    scores = ( incorrect_score, correct_score, confident_correct_score, confident_score, confidence_correct_score, confidence_score )
    return ( counts, scores )
    
      

def print_bucket_results_pmf(inferred_pmfs, test_symbs, confidence_threshold, clf, train_symbol_names, symbols_in_symtab):

    #inferred = [ [ (score, sim, symbol), ... ], ... ] 
    symb_print_proj = [ 'name', 'optimisation', 'bin_name', 'compiler', 'vaddr', 'path' ]
    symb_debug_proj = [ 'name', 'optimisation', 'bin_name', 'compiler', 'vaddr', 'path', 'hash', 'opcode_hash', '_id', 'cfg', 'callers', 'callees' ]
    similarity_file = open(cfg.desyl + '/res/similarities.json', 'a')

    logger.info("Performing threaded post analysis...")
    if len(inferred_pmfs) == 0:
        logger.info("Nothing inferred!")
        return

    s_correct, s_confident, s_theoretical, s_confidence, s_correct_name, s_in_training_set  = zip(*threaded_post_analysis(test_symbs, inferred_pmfs, train_symbol_names, clf))

    assert( len(inferred_pmfs) == len(s_correct) )
    assert( len(s_confidence) == len(s_correct) )
    assert( len(s_theoretical) == len(s_correct) )
    logger.info("Post analysis complete!")
    logger.info("Summing analysis!")


    #confident is a 0/1 yes no,
    #confidence is the difference between this symbol and others
    correct_count, confident_count, confident_correct_count, theoretical_count, training_set_count = 0, 0, 0, 0, 0
    confidence_count, confidence_correct_count = 0,0
    incorrect_score, correct_score, confident_correct_score, confident_score = 0.0, 0.0, 0.0, 0.0
    confidence_correct_score, confidence_score = 0.0, 0.0


    confident_inferred_symbs = []
    unknown_inferred_symbols = []

    for i in range(len(test_symbs)):
        correct, confidence, confident, theoretically_possible, correct_name, in_training_set = s_correct[i], s_confidence[i], s_confident[i], s_theoretical[i], s_correct_name[i], s_in_training_set[i]

        #logger.debug("//==========================================\\\\")

        #assert( len( inferred[i][0] ) == 3 )
        #__score, __sim, __symbol = inferred[i][0]
        __score = np.max( inferred_pmfs[i] )
        assert( __score <= 1.0 )
        __symbol_name = index_to_name[ np.argmax( inferred_pmfs[i] ) ]


        if confidence > confidence_threshold:
            #logger.debug("CONFIDENCE!!!!   {} > {}".format( confidence, confidence_threshold))
            confidence_count += 1
            confidence_score += __score
            confident_inferred_symbs.append( [ test_symbs[i], __symbol_name, inferred_pmfs[i]  ] )
        else:
            unknown_inferred_symbols.append( [test_symbs[i], __symbol_name, inferred_pmfs[i] ] )
        if confident:
            confident_count += 1
            confident_score += __score
        if correct:
            correct_count += 1
            correct_score += __score
            if confidence > confidence_threshold:
                confidence_correct_count += 1
                confidence_correct_score += __score
            if confident:
                confident_correct_count += 1
                confident_correct_score += __score

        else:
            incorrect_score += __score
        if theoretically_possible:
            #logger.debug("[+] Symbol could theoretically be inferred.")
            theoretical_count += 1
        else:
            #logger.debug("[+] Theoretically impossible for symbol to be inferred. (Nucelus ERROR! No symbols at this functions vaddr) ")
            assert(not correct and not in_training_set)
            #assert(not correct)
        
        if in_training_set:
            #logger.debug("[+] The correct symbol name WAS in the training set")
            training_set_count += 1
        else:
            #logger.debug("[+] The correct symbol name was NOT in the training set")
            pass

        #logger.debug("The correct symbol name was '{}'".format( correct_name ))

        #if not correct and in_training_set and logger.level == logging.DEBUG:
            #logger.debug( "=======DEBUG=======" )
            #logger.debug( __symbol_name )
            #logger.debug( test_symbs[i].to_str_custom( symb_debug_proj )  )
            #logger.debug( "PMF:" )
            #logger.debug( inferred_pmfs[i] )

           # sorted_pmf = np.argsort(inferred_pmfs[i][:,0], axis=None) 
            #for j in range(len(name_to_index), len(name_to_index) -10, -1):
                #logger.debug("- {} :: P(x) == {}".format( index_to_name[ sorted_pmf[j - 1] ], inferred_pmfs[i][sorted_pmf[j-1], 0] ) )

            #logger.debug( "=====END=DEBUG=====" )

        #logger.debug("\\\\==========================================//")


    #save to res
    classes.utils.save_py_obj(confident_inferred_symbs, "confident_inferred_symbs")
    classes.utils.save_py_obj(unknown_inferred_symbols, "unknown_inferred_symbols")

    logger.info("[+] Inferred {} of {} correctly from a maximum thoretical of {}.".format( str(correct_count), str(len(inferred_pmfs)), str(theoretical_count)))
    logger.info("[+] Inferred {} of {} correctly from {} in symtab.".format( str(correct_count), str(len(inferred_pmfs)), str(symbols_in_symtab)))

    counts = ( correct_count, confident_count, confident_correct_count, theoretical_count, training_set_count, confidence_count, confidence_correct_count )
    scores = ( incorrect_score, correct_score, confident_correct_score, confident_score, confidence_correct_score, confidence_score )
    return ( counts, scores )
 
def inferred_confidence_pmf( inferred_pmf ):
    """
        Given a PMF
        Return the confidence of teh highest inferred symbol name.

        This is done by checking the highest N=5 symbols and finding the first symol name the fails the 
        NLP name similarity check.
        Then return the difference in their probabilities

        :param inferred_pmf: A numpy vector representing the probability mass function for an inferred symbol
        :return: The confidence in the highest probabilities
        :rtype: float
    """
    #ABS_THRESHOLD = 0.05 # 1% change that the symbol is correct normalised over all symbols
    ABS_THRESHOLD = 10.0 / float(len(name_to_index))    #100 times better than unifornly random

    N = 10
    indexes = classes.pmfs.PMF.highest_indexes( inferred_pmf, N )

    if inferred_pmf[ indexes[0] ] < ABS_THRESHOLD:
        logger.info("NOT CONFIDENT! {} BELOW THRESHOLD".format( inferred_pmf[ indexes[0] ] ))
        return 0.0

    assert(len(indexes) == N)
    #each item is a tuple of N items
    for i in range(1, N):
        #each matched symbol is tuple of ( score, weightings, symbol obj )
        if inferred_pmf[ indexes[i] ] == 0.0:
            #no other symbol inferred
            return 1.0

        similar = check_similarity_of_symbol_name( index_to_name[ indexes[0] ], index_to_name[ indexes[i] ] )
        if similar:
            continue

        if not similar:
            #calculate the score difference as a percentage between 5%
            #score_diff = inferred_pmf[ indexes[0] ] - inferred_pmf[ indexes[i] ]
            #score_percentage = inferred_pmf[ indexes[i] ] / inferred_pmf[ indexes[0] ]
            score_percentage = inferred_pmf[ indexes[0] ] / ( inferred_pmf[ indexes[i] ] + inferred_pmf[ indexes[0] ] )
            #if score_diff < 0.05:
            #    logger.warn("[!] Confidence score of {} between: {} and {}".format(score_diff, inferred_symbols[0][2].name, inferred_symbols[i][2].name))
            #return score_diff
            return score_percentage

    return 1.0



    
#sort though top matching symbols
#find the next symbol that is of a different name.
#check that it has at least a 5% difference in similarity -> ? statistically different
def inferred_confidence( inferred_symbols  ):
    #each item is a tuple of N items
    for i in range(1, len(inferred_symbols)):
        #each matched symbol is tuple of ( score, weightings, symbol obj )
        if inferred_symbols[i][0] == -1:
            #no other symbol inferred
            return 1.0

        similar = check_similarity_of_symbol_name( inferred_symbols[0][2].name, inferred_symbols[i][2].name )
        if similar:
            continue

        if not similar:
            #calculate the score difference as a percentage between 5%
            score_diff = inferred_symbols[0][0] - inferred_symbols[i][0]
            #if score_diff < 0.05:
            #    logger.warn("[!] Confidence score of {} between: {} and {}".format(score_diff, inferred_symbols[0][2].name, inferred_symbols[i][2].name))
            return score_diff

    return 1.0


def perform_analysis(db, buckets, train_config, test_config):

    clf = load_model("RandomForestClassifier")
    # score top / (score next + score top)
    """
        min confidence for prediction is 0.5 i.e top symbol is same same as next biggest symbol
    """
    confidence_threshold = 1 - ((2 / math.e) / 2)  # peak n times as big as another one
    confidence_threshold = 0.55  # peak n times as big as another one

    logger.info("Confidence threshold : {}".format( confidence_threshold ))

    total_correct_score, total_incorrect_score = 0.0, 0.0
    total_test, total_train, total_correct = 0, 0, 0
    total_theoretical, total_in_training = 0, 0
    total_confident, total_confident_correct, total_confident_correct_score, total_confident_score      = 0, 0, 0.0, 0.0
    total_confidence, total_confidence_correct, total_confidence_score, total_confidence_correct_score   = 0, 0, 0.0, 0.0
    total_symbols_in_symtab = 0
    total_zig_correct, total_zig_incorrect = 0, 0

    try: 
        for b in buckets:
            assert(isinstance(b[0], list))
            assert(isinstance(b[1], list))
            train_config[0]['bin_names'] = b[0]
            test_config[0]['bin_names'] = b[1]
            logger.info("[+] Getting training and testing symbols")
            #train_symbs, test_symbs = get_train_test_symbols(db, train_config, test_config)
            test_symbs = get_test_symbols(db, test_config)
            print("#test symbols: {}".format( len(test_symbs) ))
            num_train_symbols = get_number_of_symbols(db, train_config)
            assert(isinstance(num_train_symbols, int))

            symbols_in_symtab = max_symtab_symbols(db, test_config)
            total_symbols_in_symtab += symbols_in_symtab

            logger.info("[+] Fetched {} training symbols and {} testing symbols.".format( num_train_symbols, len(test_symbs)) )
            logger.info("[+] Inferring test symbols from training data...")
            #inferred = infer_symbols_threaded(test_symbs, train_symbs)

            if num_train_symbols == 0:
                logger.error("Impossible to infer any symbols with no training data!")
                continue

            if len(test_symbs) == 0:
                logger.error("No test symbols to infer!")
                continue

            mp = Pool(processes=32)
            unknown_pmfs = mp.map(infer_symbol_pmf, test_symbs)

            """
            unknown_pmfs = []
            for unknown in test_symbs:
                unknown_pmfs.append( infer_symbol_pmf(unknown) )
            """


            logger.info("Inference complete!")
            logger.info("Fetching unique symbol names from training data...")
            train_symbol_names = set( get_unique_symbol_names(db, train_config ) )

            logger.info("[+] Performing post analysis on inferred symbols!")
            #counts, scores = print_bucket_results(inferred_symbols, test_symbs, confidence_threshold, clf, train_symbs, symbols_in_symtab)
            if not CONFIDENCE_ONLY:
                counts, scores = print_bucket_results_pmf(unknown_pmfs, test_symbs, confidence_threshold, clf, train_symbol_names, symbols_in_symtab)
            else:
                print_bucket_results_pmf_confidence_only(unknown_pmfs, test_symbs, confidence_threshold, train_symbol_names, symbols_in_symtab)
                counts = 0, 0, 0, 0, 0, 0, 0
                scores = 0.0,0.0,0.0,0.0,0.0,0.0

            zig_correct = -1
            zig_incorrect = -1
            if False:
                zig_correct, zig_incorrect = ZA.r2_infer_symbols(train_config, test_config)
                logger.info("[+] Radare2 zignatures inferred {} correctly and misclassified {} symbols.".format( zig_correct, zig_incorrect))

            total_zig_correct += zig_correct
            total_zig_incorrect += zig_incorrect
            

            correct_count, confident_count, confident_correct_count, theoretical_count, training_set_count, confidence_count, confidence_correct_count = counts
            incorrect_score, correct_score, confident_correct_score, confident_score, confidence_correct_score, confidence_score = scores

            total_correct += correct_count
            total_confident += confident_count
            total_confidence += confidence_count
            total_confidence_correct += confidence_correct_count
            total_confident_correct += confident_correct_count
            total_theoretical += theoretical_count
            total_in_training += training_set_count

            total_incorrect_score += incorrect_score
            total_correct_score += correct_score
            total_confident_score += confident_score
            total_confidence_score += confidence_score
            total_confident_correct_score += confident_correct_score
            total_confidence_correct_score += confidence_correct_score

            total_test += len(test_symbs)
            total_train += num_train_symbols

            logger.info("[+] Bucket complete")
        
            classes.utils.save_py_obj(test_symbs, "test_symbols")
            classes.utils.save_py_obj(unknown_pmfs, "unknown_pmfs")

            #del inferred_symbols
            #del test_symbs
            gc.collect()
            ###########################
            #only do 1 iteration 
            break
            ###########################

    except KeyboardInterrupt:
        print("Caught Ctrl-C. Terminating analysis.")


    logger.info("\n[====================================================]")
    logger.info("\tOverall Results:\n")
    logger.info("\t[+] Inferred {} of {} correctly using {} training symbols with an average score of {:.2f}.".format(total_correct, total_test, total_train, 0 if total_correct == 0 else total_correct_score / total_correct))


    logger.info("\t[+] {} symbols with confidence above {:.2f}% with an average socre of {:.2f}".format( total_confidence, confidence_threshold, 0 if total_confidence == 0 else float(total_confidence_score / total_confidence) ) )

    logger.info("\t[+] {} symbols confidently inferred with an average socre of {:.2f}".format( total_confident, 0 if total_confident == 0 else float(total_confident_score / total_confident) ) )

    logger.info("\t[+] Maximum theoretically inferable symbols: {}.".format( total_theoretical ) )
    logger.info("\t[+] {} / {} symbols were in the training set.".format( total_in_training, total_test ) )

    total_incorrect = total_test - total_correct
    logger.info("\t[+] {} incorrect symbols with an average score of {:.2f}".format( total_incorrect, 0 if total_incorrect == 0 else float(total_incorrect_score / total_incorrect) ) )

    logger.info("\t[+] Inferred {:.2f}% of symbols correctly out of a maximum thoretical {:.2f}%!".format( 0.0 if total_test == 0 else 100.0 * (float(total_correct) / float(total_test)) , 0.0 if total_test == 0 else 100.0 * (float(total_theoretical)/float(total_test)) ) )

    logger.info("\t[+] Infered {:.2f}% of the symbols from symtab ({} / {})".format( 0.0 if total_symbols_in_symtab == 0 else 100.0 * (total_correct / total_symbols_in_symtab ) , total_correct, total_symbols_in_symtab) )

    logger.info("\t[+] Infered {:.2f}% of the maximum theoretically inferrable symbols ({} / {})".format( 0.0 if total_theoretical == 0 else 100.0 * (total_correct / total_theoretical ) , total_correct, total_theoretical) )

    logger.info("\t[+] Infered {:.2f}% of the confident symbols correctly ({} / {}) with an average score of {:.2f}".format( 0.0 if total_confident == 0 else 100.0 * (total_confident_correct / total_confident ), total_confident_correct, total_confident, 0.0 if total_confident == 0 else 100.0 * total_confident_correct_score / total_confident) )

    logger.info("\t[+] Infered {:.2f}% of the symbols with high confidence correctly ({} / {}) with an average score of {:.2f}".format( 0.0 if total_confidence == 0 else 100.0 * (total_confidence_correct / total_confidence ), total_confidence_correct, total_confidence, 0.0 if total_confidence == 0 else 100.0 * total_confidence_correct_score / total_confidence) )

    logger.info("\t[+] Infered {:.2f}% of the symbols that were in the training set ({} / {})".format( 0.0 if total_in_training == 0 else 100.0 * (total_correct / total_in_training ) , total_correct, total_in_training) )

    logger.info("\t[+] Symbol names cross over {:.2f}% ({}/{})".format( 0.0 if total_test == 0 else 100.0 * (float(total_in_training) / float(total_test)) , total_in_training, total_test))

    tp = total_confidence_correct
    tn = total_test - total_theoretical
    fp = total_confidence - total_confidence_correct 
    #fn = total_theoretical - total_confidence
    fn = total_theoretical - total_confidence_correct

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


    logger.info("[+] Overall, Radare2 zignatures inferred {} correctly and misclassified {} symbols.".format( total_zig_correct, total_zig_incorrect))


    logger.info("[+] Done")
    return True

#Get a list of binaries that covers the config
def get_list_of_bin_names(db, config):
    query = Database.gen_query(config, projection = { 'bin_name': 1 } )
    query.append( { '$group' : { '_id' : '$bin_name' } } )

    res = db.client.symbols.aggregate( query )
    bin_names = []
    for b in res:
        bin_names.append( b['_id'] )
    return random.sample( bin_names, k=len(bin_names) )

if __name__ == '__main__':
    global CONFIDENCE_ONLY
    CONFIDENCE_ONLY = False

    logger.setLevel(logging.DEBUG)

    """
    base_ratio = 0.15
    P = classes.pmfs.PMF()

    for value in range(50):
        mult = P.exp_neg_x(10, 10, value)
        r = base_ratio * (1.0+mult)
        min = value - (r*value)
        max = value + (r*value)

        print("Selection ratio for value {} is: {}".format( value, r ))
        print("\tSelection range: {} <= x < {}".format(min, max))
    sys.exit()
    """

    name_to_index = classes.utils.load_py_obj('name_to_index')
    index_to_name = classes.utils.load_py_obj('index_to_name')
    assert(len(name_to_index) == len(index_to_name))
    dim = len(name_to_index)

    log_file = logging.FileHandler(cfg.desyl + "/res/analysis.log")
    logger.addHandler(log_file)
    logger.setLevel(logging.INFO)

    args = get_args()
    wn.ensure_loaded()

    min_projection = {
        'name' : 1,
        'path' : 1,
        'bin_name': 1,
        'vaddr': 1,
        'compiler' : 1,
        'optimisation' : 1,
        'linkage' : 1
    }

    #Reduce memory by projecting only symbol attributes used in similarity function
    full_projection = { 'size' : 1, 'hash': 1, 'opcode_hash': 1, 'cfg': 1,
        'vex.statements': 1,
        'vex.operations': 1,
        'vex.expressions': 1,
        'vex.ntemp_vars': 1,
        'vex.temp_vars': 1,
        'vex.sum_jumpkinds': 1,
        'vex.jumpkinds': 1,
        'vex.ninstructions': 1,
        'vex.constants': 1,
        'callers': 1,
        'callees': 1,
    }

    vex_proj = { 'vex' : 1 }
    flirt_projection = { 'hash' : 1 }
    hashes_projection = { 'hash' : 1, 'opcode_hash': 1 }

    test_proj = { 'cfg' : 1 }
    #test_proj.update( hashes_projection )


    #min_projection.update( flirt_projection )
    min_projection.update( full_projection )
    #min_projection.update( test_proj )

    test_query_config = cfg.test
    train_query_config = cfg.train

    train_config = ( train_query_config, min_projection )
    test_config = ( test_query_config, min_projection )

    if isinstance(args.fold, int) and isinstance(args.bucket_size, int):
        logger.error("[!] Error, cannot have value for k-fold validation and bucket size at the same time")
        #sys.exit(-1)

    #by default perform 10 fold cross validation
    if args.fold is None and args.bucket_size is None:
        args.fold = 10

    db = Database()

    logger.info('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    logger.info('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    logger.info('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    logger.info("New analysis :: {}".format(datetime.datetime.now() ))
    logger.info("\tTrain config :: {}".format( json.dumps(train_config) ))
    logger.info("\tTest config :: {}".format( json.dumps(test_config) ))


    """
    logger.debug("[+] Getting unique names of all the binaries in the database for configuration...")
    #pprint.pprint(config)
    test_bins = get_list_of_bin_names(db, test_query_config)
    train_bins = get_list_of_bin_names(db, train_query_config)


    logger.debug("Test bins: len={}, {}".format( len(test_bins), test_bins ) )
    logger.debug("Train bins: len={}, {}".format( len(train_bins), train_bins ) )


    #cannot have a binary that is in th etest set and not training set
    # i.e. trainset needs to overlap test set
    sbin_train = set(test_bins)
    sbin_test = set(train_bins)

    #assert( sbin_test - sbin_train == set([]) and "Cannot have binaries in the test set that are not in the training set)
    bins = list( sbin_train.union(sbin_test) )

    #logger.debug(json.dumps(bins))

    #compute bucket_size from fold
    #TODO: use round-robin algorithm to fill exactly n buckets
    if args.bucket_size == None:
        args.bucket_size = int( math.ceil( float( len(bins) ) / float(args.fold)  ) )


    #separate into buckets
    max_buckets = int( math.ceil( float( len(bins) ) / float( int(args.bucket_size) ) ) )
    logger.info("Bins: " + str(len(bins)))
    logger.info("Buckets: " + str(max_buckets))
    buckets = fill_buckets( max_buckets, int(args.bucket_size), bins) 
    #print(buckets[0])

    classes.utils.save_py_obj(buckets[0][0], "train_bins")
    classes.utils.save_py_obj(buckets[0][1], "test_bins")
    """

    train_bins = cfg.train['bin_names']
    test_bins = cfg.test['bin_names']

    #total_confidenceprint(train_bins)
    #test_bins = [ test_bins[7] ]

    buckets = [ [train_bins, test_bins] ]
    perform_analysis(db, buckets, train_config, test_config )
