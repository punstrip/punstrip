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
from sklearn.neighbors import KNeighborsRegressor
from incremental_trees.trees import StreamingRFC
from sklearn.decomposition import IncrementalPCA

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
import classes.bin_mod
import classes.NLP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import classes.experiment
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
    USE_CACHED_DATASET = True 
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
        y_train = scipy.sparse.vstack(train_df['name'].apply(lambda x: E.to_sparse_csc_vec('name_vector', [x])).values)
        y_test  = scipy.sparse.vstack(test_df['name'].apply(lambda x: E.to_sparse_csc_vec('name_vector', [x])).values)
        #y_train = np.vstack(train_df['name'].apply(lambda x: E.to_index('name_vector', x)).values)
        #y_test  = np.vstack(test_df['name'].apply(lambda x: E.to_index('name_vector', x)).values)

        x_train = np.vstack(train_df['embedding'].values)
        x_test  = np.vstack(test_df['embedding'].values)

        for model in [ "x_train", "y_train", "x_test", "y_test", "train_df", "test_df" ]:
            print("Saving", model)
            classes.utils.pickle_save_py_obj(config, locals()[model], model)
    else:
        for model in [ "x_train", "y_train", "x_test", "y_test" ]:
            print("Loading", model)
            locals()[model] = classes.utils.pickle_load_py_obj(config, model)

    print("Loaded dataset")
    print("Training")
    #y_train = y_train.tocsr()
    #y_test  = y_test.tocsr()

    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, kernel_initializer='orthogonal'),
        keras.layers.Dropout(0.2),
        #keras.layers.Dense(323, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.softmax, activity_regularizer=l2(0.1))
        ])

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #sgd = optimizers.SGD(lr=0.05, momentum=0.1, decay=0.0, nesterov=False)


    #model.compile(optimizer='adam',
    model.compile(optimizer=sgd,
            #loss='sparse_categorical_crossentropy',
            #loss='binary_crossentropy',
            #loss=sparsemax_loss(tf.Tensor()),
            loss='categorical_crossentropy',
            #loss='squared_hinge',
            #loss='mean_squared_error',
            metrics=['accuracy'])

 
    batch_size = 2500 
    steps_per_epoch = x_train.shape[0] // batch_size
    epochs = 10

    #model.fit(x_train, y_train, epochs=5)
    #model.fit_generator( generator=batch_generator(x_train, y_train, batch_size, epochs), epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=(x_test, y_test))
    IPython.embed()
    model.fit_generator( generator=batch_generator(x_train, y_train, batch_size, epochs), epochs=epochs, steps_per_epoch=steps_per_epoch)

    #test_loss, test_acc = model.evaluate(x_test, y_test)
    test_loss, test_acc = model.evaluate_generator(generator=batch_generator(x_test, y_test, batch_size, epochs), steps=x_test.shape[0]//batch_size)
    print('Test accuracy:', test_acc)
    IPython.embed()

    sys.exit()
    predictions = model.predict(test_images)




    accuracy, tp, fp = top_n_accuracy(clf, 1, x_test, y_test)
    print("ACC@1: {}, TP: {}, FP: {}".format(accuracy, tp, fp))
    accuracy, tp, fp = top_n_accuracy(clf, 5, x_test, y_test)
    print("ACC@5: {}, TP: {}, FP: {}".format(accuracy, tp, fp))
    #accuracy, tp, fp = top_n_nlp_accuracy(clf, 1, nlp, E, x_test, y_test)
    #print("ACC: {}, TP: {}, FP: {}".format(accuracy, tp, fp))
    #accuracy, tp, fp = top_n_nlp_accuracy(clf, 5, nlp, E, x_test, y_test)
    #print("ACC: {}, TP: {}, FP: {}".format(accuracy, tp, fp))
    IPython.embed()

    ##save model to current experiment configuration
    #bin_form = classes.utils.py_obj_to_bytes( clf )
    #E.update_experiment_key('fingerprint_model', bin_form)

    print("Done")
    sys.exit()
    print("Inferring symbols...")

    inferred_symbols = []
    ##iinfer symbols in test_x
    for i, test_s in test_df.iterrows():
        inferred_index = clf.predict( np.transpose( test_s['vec'] ) )
        inf_name = E.name_vector[ inferred_index[0] ]

        inferred_symb  = classes.symbol.Symbol(conf, name = inf_name, vaddr=test_s['vaddr'], size=test_s['size'])
        inferred_symbols.append(inferred_symb)

    print("Adding to binary")
    classes.utils.save_py_obj(conf, inferred_symbols, "inferred_symbols")


    strip_path = '/root/mirai.stripped'
    BM = classes.bin_mod.BinaryModifier(conf, strip_path)

    BM.add_symbols(inferred_symbols)

    out_path = '/tmp/mirai.inferred'
    print("[+] Writing unstripped binary to {}".format(out_path))
    BM.save(out_path)

    sys.exit(-1)

    #first make some fake data with same layout as yours
    data = pd.DataFrame(_x, columns=['x1', 'x2', 'x3',\
                        'x4','x5','x6','x7','x8','x9','x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'])

    #now plot using pandas 
    pd.plotting.scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()
