#!/usr/bin/python3
import sys
import pickle
import json
import pprint
import numpy as np
import scipy as sp
import matplotlib
#matplotlib.use('Qt5Agg')
import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.sparse

from sklearn import tree
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


import scipy.sparse as sp
from joblib.parallel import cpu_count, Parallel, delayed
import seaborn as sns
import pandas as pd
import IPython


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

#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

features =  [
            "size in bytes", 
            "hash",
            "opcode hash",
            "number of VEX IR instructions",
            "sum_jumpkinds :: types", 
            "sum_jumpkinds :: number and type", 
            "temp_vars :: types", 
            "temp_vars :: number and type", 
            "jumpkinds :: order",
            "constants matching :: type and value",
            "statements :: catagorised",
            "operations :: catagorised",
            "expressions :: catagorised",
            "CFG number of callees",
            "CFG number of callers", 
            "gk_weisfeiler_lehman graph kernel" 
]


def show_box_plot(weights):

    fig, ax1 = plt.subplots(figsize=(7, 6))
    bp = plt.boxplot(weights, notch=0, sym='+', vert=1, whis=1.5)
    plt.title('Boxplot with outliers for each feature of the similarity score')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16], features)
    plt.xticks(rotation=90)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    plt.show()


# Show how each wieghtings correlates to the class
def show_correlation(weights):
    correlation_matrix = np.corrcoef(weights)
    #correlation_matrix = np.cov( weights )
    #print( np.isfinite(correlation_matrix).all() )
    #pprint.pprint( correlation_matrix )
    print(correlation_matrix)
    # plt.figure(figsize=(10,8))
    plt.figure(figsize=(7, 6))

    df = pd.DataFrame(correlation_matrix, index=features, columns=features)
    ax = sns.heatmap(df, vmax=1, square=True, annot=True, cmap='RdYlGn')
    plt.title('Correlation of features in a symbols similarity score')
    plt.show()


X = []
Y = []

# with open('../res/similarities.json', 'r') as f:


def learn_pseudo_inverse(x, y):
    print("shape of x is {}".format(x.shape))
    print("shape of y is {}".format(y.shape))

    print("Calculating pseudo inverse of similarity features:")
    Ainv = np.linalg.pinv(x)

    print("shape of x^-1 is {}".format(Ainv.shape))
    print("shape of y^t is {}".format(np.transpose(y).shape))
    w = np.dot(Ainv,  y)

    print("shape of w is {}".format(w.shape))
    print(w)


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
    y_inf = clf.predict(x)
    items, symb_vec = np.shape(y_inf)

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(items):
        correct_ind = y[i].argmax()
        top_n = y_inf[i, :].argsort()[-n:][::-1]
        if correct_ind in top_n:
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
    #score = metrics.accuracy_score(y_train, pred)
    #print("{} training accuracy:\t{:0.3f}".format( name, score) )

    pred = clf.predict(x_test)
    #score = metrics.accuracy_score(y_test, pred)
    #print("{} test accuracy:\t{:0.3f}".format( name, score ))

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


def _predict(estimator, X, method, start, stop):
    return getattr(estimator, method)(X[start:stop])

def parallel_predict(estimator, X, n_jobs=1, method='predict', batches_per_job=3):
    n_jobs = max(cpu_count() + 1 + n_jobs, 1)  # XXX: this should really be done by joblib
    n_batches = batches_per_job * n_jobs
    n_samples = len(X)
    batch_size = int(np.ceil(n_samples / n_batches))
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(delayed(_predict)(estimator, X, method, i, i + batch_size)
                       for i in range(0, n_samples, batch_size))
    if sp.issparse(results[0]):
        return sp.vstack(results)
    return np.concatenate(results)


if __name__ == '__main__':
    conf = classes.config.Config()
    #switch to symbols db
    #conf.database.collection_name = 'd'

    KFOLD = 2

    db   = classes.database.Database(conf)
    nlp = classes.NLP.NLP(conf)
    E = classes.experiment.Experiment(conf)
    E.load_settings()

    #IPython.embed()


    if False:
        ##get symbols from database
        projection = { 
            '$project' : { 
                'vex':1, 'name': 1, 'size': 1, 'optimisation':1, 'compiler':1, 'path':1,
                'bin_name': 1, 'callers': 1, 'callees': 1, 'num_args': 1, 'sse': 1, 'vaddr': 1
            }
        }
        agg_q = [ { '$match' : {  'size' : { '$gt': 0}} }, projection ]
        s = db.get_symbols(agg_q)
        for symb in s:
            if symb.vex == None or symb.vex == {}:
                print("wtfwtf")
                IPython.embed()

        print("Creating dataframe!")

        selems = list(map(lambda x: {'name': x.name, 'path': x.path, 'vaddr': x.vaddr, 'size': x.size, 'bin_name': x.bin_name, 'compiler': x.compiler, 'optimisation': x.optimisation, 'vec' : x.to_vec(E)}, s))
        df = pd.DataFrame(selems)
        df.to_pickle("/tmp/df") 
    else:
        #classes.utils.load_py_obj(conf, "df")
        df = pd.read_pickle("/tmp/df")

    print("Loaded dataframe")

    #y_test = list(map(lambda x: E.to_sparse_vec('name_vector', [x]), test_df['name']))
    #train_df = df[df.compiler=='gcc']
    #test_df = df[df.compiler=='clang']

    df['arr_vec'] = df['vec'].map(lambda x: x.flatten())

    for KFOLD in range(3, 11):

        training = classes.utils.read_file_lines( db.config.desyl + "/res/128_kfold_10/{}/training.bins".format(KFOLD))
        testing = classes.utils.read_file_lines( db.config.desyl + "/res/128_kfold_10/{}/testing.bins".format(KFOLD))

        train_df    = df[df['path'].isin(training)]
        test_df     = df[df['path'].isin(testing)]

        x_train = np.vstack( train_df['arr_vec'] )
        x_test  = np.vstack( test_df['arr_vec'] )

        y_train = np.vstack( train_df['name'].map( lambda x: E.to_index('name_vector', x) ) )
        y_test  = np.vstack( test_df['name'].map( lambda x: E.to_index('name_vector', x) ) )


        #r, c = np.shape(x_train)
        #d = E.name_vector_dims

        """
        y_train = scipy.sparse.lil_matrix((r, d), dtype=np.float64)
        for i, name in enumerate(tqdm.tqdm(train_df['name'])):
            y_train[i,:] = E.to_sparse_lil_vec('name_vector', [name])

        y_train = y_train.tocsr()
        """

        #y_train = classes.utils.load_py_obj(conf, "y_train")
        #y_test = classes.utils.load_py_obj(conf, "y_test")

        #y_train = list(map(lambda x: E.to_sparse_vec('name_vector', [x]).reshape(-1), train_df['name']))
        #y_train = list(map(lambda x: E.to_sparse_vec('name_vector', [x]), train_df['name']))
        #y_train = np.vstack(list(map(lambda x: E.to_sparse_lil_vec('name_vector', [x]), train_df['name'])))

        #x_test = np.vstack( test_df['vec'].values.tolist() )
        #y_test = list(map(lambda x: E.to_sparse_vec('name_vector', [x]).reshape(-1), test_df['name']))
        #y_test = list(map(lambda x: E.to_sparse_vec('name_vector', [x]), test_df['name']))
        #y_test = sp.vstack(list(map(lambda x: E.to_sparse_lil_vec('name_vector', [x]), test_df['name'])))

        #r_test, c_test = np.shape(x_test)

        """
        y_test = scipy.sparse.lil_matrix((r_test, d), dtype=np.float64)
        for i, name in enumerate(tqdm.tqdm(test_df['name'])):
            y_test[i,:] = E.to_sparse_lil_vec('name_vector', [name])

        y_test = y_test.tocsr()
        """


            
        #show_box_plot( _x )
        #show_correlation( np.transpose(_x) )
        #sys.exit(-1)

        #fit_data(x, y)

        print("Training")
        #IPython.embed()
        #( (x_train, y_train), ( x_test, y_test) ) = split_test_train(x, y)
        #clf = RandomForestRegressor(n_estimators=32, n_jobs=32)
        #clf = RandomForestClassifier(n_estimators=32, n_jobs=8)
        #clf = KNeighborsClassifier(n_neighbors=24)
        #rfc = LogisticRegression(n_jobs=32)
        clf = GaussianNB()
        #clf = MultinomialNB()


        cl = np.array(list(range(len(E.name_vector))))

        nitems, cols = np.shape(x_train)
        batch_size = 10000
        for i in tqdm.tqdm(range(nitems//batch_size)):
            clf.partial_fit(
                    x_train[i*batch_size:(i+1)*batch_size], 
                    y_train[i*batch_size:(i+1)*batch_size],
                    classes=cl)

        print('Fitted model')
        #IPython.embed()


        #fit_and_predict( rfc, "Logistic Regression", x_train, y_train, x_test, y_test)


        #clf = MLPRegressor(solver='adam', learning_rate="invscaling", hidden_layer_sizes=(100,))

        #fit_and_predict( clf, "RandomForest", x_train, y_train, x_test, y_test)
        #fit_and_predict( clf, "MLP", x_train, y_train, x_test, y_test)
        classes.utils.save_py_obj(conf, clf, "clf.kfold.{}".format(KFOLD))
        #rfc = classes.utils.load_py_obj(conf, "rfc")

        continue


        pred = parallel_predict(clf, x_test)
        metrics.accuracy_score(y_test, pred)

        accuracy, tp, fp = top_n_accuracy(clf, 1, x_test, y_test)
        print("ACC: {}, TP: {}, FP: {}".format(accuracy, tp, fp))
        accuracy, tp, fp = top_n_accuracy(clf, 5, x_test, y_test)
        print("ACC: {}, TP: {}, FP: {}".format(accuracy, tp, fp))

        #accuracy, tp, fp = top_n_nlp_accuracy(clf, 1, nlp, E, x_test, y_test)
        #print("ACC: {}, TP: {}, FP: {}".format(accuracy, tp, fp))
        #accuracy, tp, fp = top_n_nlp_accuracy(clf, 5, nlp, E, x_test, y_test)
        #print("ACC: {}, TP: {}, FP: {}".format(accuracy, tp, fp))
        print('Finished training')
        IPython.embed()

        ##save model to current experiment configuration
        #bin_form = classes.utils.py_obj_to_bytes( clf )
        #E.update_experiment_key('fingerprint_model', bin_form)

        print("Done, but haven't saved clf model")
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
