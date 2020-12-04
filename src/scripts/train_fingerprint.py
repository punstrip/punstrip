#!/usr/bin/python3
import sys
import pickle
import json
import pprint
import gc
import numpy as np
import scipy as sp
import matplotlib
#matplotlib.use('Qt5Agg')
import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from incremental_trees.trees import StreamingRFC
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import pandas as pd
import IPython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
import tensorflow_addons
from tensorflow_addons.losses import sparsemax_loss


#from pandas.tools.plotting import scatter_matrix

import context
import classes.config
import classes.database
import classes.symbol
import classes.NLP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import classes.experiment
import classes.crf
import classes.utils
import sklearn
from sklearn.ensemble import RandomForestRegressor


#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def split_test_train(x, y):
    train_samples = int(0.9 * len(y))  # 90%
    x_train, x_test = x[:train_samples], x[train_samples:]
    y_train, y_test = y[:train_samples], y[train_samples:]
    assert(len(y_train) + len(y_test) == len(y))
    return ( (x_train, y_train), ( x_test, y_test) )


 
def top_n_nlp_accuracy(clf, n, nlp, E, x, y):
    y_inf = clf.predict(x)
    items, symb_vec = np.shape(y_inf)

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(items):
        correct_ind = y[i].argmax()
        top_n = y_inf[i, :].argsort()[-n:][::-1]
        COORECT = False
        for ind in top_n:
           if nlp.check_word_similarity(E.name_vector[correct_ind], E.name_vector[ind]):
               CORRECT = True
               break

        if CORRECT:
            tp += 1
        else:
            fp += 1

    accuracy = 0.0
    if (tp+tn+fp+fn) > 0:
        accuracy = (tp+tn) / (tp+tn+fp+fn)

    return accuracy, tp, fp


def top_n_accuracy(clf, n, x, y):
    y_inf = clf.predict_proba(x)
    items, symb_vec = np.shape(y_inf)

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(items):
        #correct_ind = y[i].argmax()
        correct_ind = y[i]
        top_n = y_inf[i, :].argsort()[-n:][::-1]
        ##map top_n into sklearn classes
        top_n_classes = [clf.classes_[i] for i in top_n]
        assert(len(top_n) == n)
        assert(len(top_n_classes) == n)
        if correct_ind in top_n_classes:
            tp += 1
        else:
            fp += 1

    accuracy = 0.0
    if (tp+tn+fp+fn) > 0:
        accuracy = (tp+tn) / (tp+tn+fp+fn)

    return accuracy, tp, fp

def fit_and_predict(clf, name, x_train, y_train, x_test, y_test):
    print("Training {}...".format( name ) )
    clf.fit(x_train, y_train)

    pred = clf.predict(x_train)
    score = metrics.accuracy_score(y_train, pred)
    print("{} training accuracy:\t{:0.3f}".format( name, score) )

    pred = clf.predict(x_test)
    score = metrics.accuracy_score(y_test, pred)
    print("{} test accuracy:\t{:0.3f}".format( name, score ))

    #save_model(clf, "".join(name.split()))


def save_model(model, name):
    model_fname = cfg.desyl + "/res/" + name + ".pickle" 
    print("[+] Saving model to {}".format(model_fname))
    with open(model_fname, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(name):
    model_fname = cfg.desyl + "/res/" + name + ".pickle" 
    print("[+] Loading model from {}".format(model_fname))
    with open(model_fname, 'rb') as f:
        return pickle.load(f)

def learn_logistic_regression(x, y):
    ( (x_train, y_train), ( x_test, y_test) ) = split_test_train(x, y)
    lr = LogisticRegression()
    fit_and_predict( lr, "Logistic Regression", x_train, y_train, x_test, y_test)


def get_correct_predictions(x, y):
    correct = []
    for i in range(len(y)):
        if y[i] == 1:
            correct.append( x[i, :] )

    return np.array( correct )

def fit_data(x, y):
    ( (x_train, y_train), ( x_test, y_test) ) = split_test_train(x, y)

    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier(n_estimators=100)
    nnc = MLPClassifier(solver='adam', learning_rate="invscaling", hidden_layer_sizes=(100,100,100, 100))
    
    for clf, name in [ (lr, "Logistic Regression"), (gnb, "Gaussian Naive Bayes"), (svc, "Support Vector Machine Classifier"), (rfc, "Random Forest Classifier") ]: #, (nnc, "Neural Network Classifier") ]:
        fit_and_predict( clf, name, x_train, y_train, x_test, y_test)


def shuffle_in_unison(a, b):
    assert(a.shape[0] == b.shape[0])
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

def batch_generator(in_x, in_y, batch_size, epochs):
    n_batches_for_epoch = in_x.shape[0]//batch_size
    for j in range(epochs):
        ##shuffle input data
        x, y = shuffle_in_unison(in_x, in_y)
        for i in range(n_batches_for_epoch):
            index_batch = range(x.shape[0])[batch_size*i:batch_size*(i+1)]
            #x_batch = x[index_batch,:].todense()
            x_batch = x[index_batch,:]
            y_batch = y[index_batch,:]
            #print("Check x_batch, y_batch")
            #IPython.embed()
            #y_batch = np.vstack(y[index_batch])
            yield(x_batch,y_batch)
            #yield(np.array(x_batch),y_batch)

if __name__ == '__main__':
    ####
    USE_CACHED_DATASET  = True
    USE_CACHED_PCA      = False
    ####

    config = classes.config.Config()
    db   = classes.database.Database(config)
    nlp = classes.NLP.NLP(config)
    E = classes.experiment.Experiment(config)
    E.load_settings()

    if not USE_CACHED_DATASET:
        ##desyl symbol embeddings need to be created by running 
        ## ./src/scripts/generate_symbol_embeddings.py
        df = pd.read_pickle("/tmp/symbol_embeddings_df")
        print("Loaded dataframe")

        ##remove calculable known functions such as main
        df = df[~df['name'].isin( classes.crf.CRF.calculable_knowns )]
        IPython.embed()

        #binaries = df['binary'].unique()
        #train_binaries, test_binaries = train_test_split(binaries, test_size=0.05)
        train_binaries = classes.utils.pickle_load_py_obj(config, 'train_binaries')
        test_binaries = classes.utils.pickle_load_py_obj(config, 'test_binaries')

        print('filtered names...')
        train_df    = df[df['binary'].isin(train_binaries)]
        test_df     = df[df['binary'].isin(test_binaries)]

        print("Stacking vectors")
        ##regression vs classification
        #y_train = scipy.sparse.vstack(train_df['name'].apply(lambda x: E.to_sparse_csc_vec('name_vector', [x])).values)
        #y_test  = scipy.sparse.vstack(test_df['name'].apply(lambda x: E.to_sparse_csc_vec('name_vector', [x])).values)
        y_train = np.vstack(train_df['name'].apply(lambda x: E.to_index('name_vector', x)).values)
        y_test  = np.vstack(test_df['name'].apply(lambda x: E.to_index('name_vector', x)).values)

        x_train = np.vstack(train_df['embedding'].values)
        x_test  = np.vstack(test_df['embedding'].values)

        for model in [ "x_train", "y_train", "x_test", "y_test", "train_df", "test_df" ]:
            print("Saving", model)
            classes.utils.pickle_save_py_obj(config, locals()[model], model)
    else:
        for model in [ "x_train", "y_train", "x_test", "y_test", "test_df" ]:
            print("Loading", model)
            locals()[model] = classes.utils.pickle_load_py_obj(config, model)

    print("Loaded dataset")
    print("Training")
    #y_train = y_train.tocsr()
    #y_test  = y_test.tocsr()

    if not USE_CACHED_PCA:
        ##apply standardscaling
        sc = StandardScaler()
        sc.fit(np.vstack([x_train, x_test]))

        x_train_sc  = sc.transform(x_train)
        x_test_sc   = sc.transform(x_test)

        ###first apply Incremental PCA
        BATCH_SIZE = 25000
        n_elems, n_feats = x_train.shape
        pca = IncrementalPCA(n_components=256)
        for i in tqdm.tqdm(range(1 + (n_elems // BATCH_SIZE)), desc='Fitting PCA'):
            pca.partial_fit(x_train_sc[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

        print("Applying PCA")
        x_test_pca = pca.transform(x_test_sc)
        x_train_pca = pca.transform(x_train_sc)

        classes.utils.pickle_save_py_obj(config, sc, 'sc')
        classes.utils.pickle_save_py_obj(config, pca, 'pca')
        classes.utils.pickle_save_py_obj(config, x_test_pca, 'x_test_pca')
        classes.utils.pickle_save_py_obj(config, x_train_pca, 'x_train_pca')
    else:
        sc = classes.utils.pickle_load_py_obj(config, 'sc')
        pca = classes.utils.pickle_load_py_obj(config, 'pca')
        x_test_pca = classes.utils.pickle_load_py_obj(config, 'x_test_pca')
        x_train_pca = classes.utils.pickle_load_py_obj(config, 'x_train_pca')

    ##converts y to n_samples x name_vector_dims matrix
    #y_train = np.apply_along_axis(lambda x,exp=E: exp.to_vec('name_vector', [x[0]]), 1, y_train)
    #y_test = np.apply_along_axis(lambda x,exp=E: exp.to_vec('name_vector', [x[0]]), 1, y_test)

    #converts y to n_samples x 1 class labels
    #y_train_cl  = np.apply_along_axis(lambda x,exp=E: exp.to_index('name_vector', x[0]), 1, y_train)
    #y_test_cl   = np.apply_along_axis(lambda x,exp=E: exp.to_index('name_vector', x[0]), 1, y_test)
    y_test_cl = classes.utils.pickle_load_py_obj(config, 'y_test_cl')
    y_train_cl = classes.utils.pickle_load_py_obj(config, 'y_train_cl')

    #"""
    #cl = np.array(list(range(E.name_vector_dims)))

    ##inversely weight samples 
    weighting = np.zeros((E.name_vector_dims, ), dtype=np.float64)
    for sample in tqdm.tqdm(y_train_cl, desc="Counting class samples"):
        weighting[sample] += 1.0

    inv_weighting = weighting
    for i in tqdm.tqdm(range(weighting.shape[0]), desc='Building inverse weight for classes'):
        if weighting[i] == 0.0:
            continue

        inv_weighting[i] = 1.0 / weighting[i]

    ##produce inv_weighting for each sample
    y_train_weightings = np.zeros(y_train_cl.shape, dtype=np.float64)
    for i, cl in tqdm.tqdm(enumerate(y_train_cl), desc="Building samples weight matrix"):
        y_train_weightings[i] = inv_weighting[cl]

    print("Check weightings matrices")
    #IPython.embed()

    class_weights_dict = {}
    for i, w in tqdm.tqdm(enumerate(inv_weighting),desc='Compute class_weights'):
        if w == 0.0:
            continue
        class_weights_dict[i] = w


    print("About to train model")
    IPython.embed()
    #clf = RandomForestRegressor(n_estimators=256, n_jobs=128)
    #clf = LogisticRegression(n_jobs=32)
    #clf = MLPRegressor(solver='adam', learning_rate="invscaling", hidden_layer_sizes=(100,))
    #clf = SGDRegressor()
    knc_clf = KNeighborsClassifier(n_neighbors=24, weights='distance')
    #clf = KNeighborsRegressor(n_neighbors=128, weights='distance', n_jobs=120)
    #gnb_clf = GaussianNB()
    #rfc_clf = StreamingRFC(n_jobs=120, n_estimators_per_chunk=2, class_weight=class_weights_dict)
    #rfc_clf = StreamingRFC(n_jobs=120, n_estimators_per_chunk=2)

    IPython.embed()


    nitems, cols = np.shape(x_train_pca)
    batch_size = 100000
    nepochs = 1
    for e in range(nepochs):
        for i in tqdm.tqdm(range(1 + (nitems//batch_size)), 'Training fingerprint model'):
            knc_clf.partial_fit(
                                         x_train_pca[i*batch_size:(i+1)*batch_size], 
                                          y_train_cl[i*batch_size:(i+1)*batch_size],
                    #sample_weight=y_train_weightings[i*batch_size:(i+1)*batch_size],
                    classes=np.unique(y_train_cl))


    accuracy, tp, fp = top_n_accuracy(knc_clf, 5, x_test_pca, y_test_cl)
    print("ACC@5: {}, TP: {}, FP: {}".format(accuracy, tp, fp))

    #fit_and_predict( clf, "RandomForest", x_train, y_train, x_test, y_test)
    #fit_and_predict( clf, "MLP", x_train, y_train, x_test, y_test)
    #classes.utils.pickle_save_py_obj(config, clf, "clf")

    print("Finished training...")
    IPython.embed()

