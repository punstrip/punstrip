import context
from classes.config import Config
from classes.experiment import Experiment
import classes.utils
from sklearn.cluster import DBSCAN, SpectralClustering, MiniBatchKMeans
import collections
import numpy as np
import IPython

def labels_to_classes(clf, names, n_clusters):
    fname_classes   = {}
    ##get name of cluster as closest in symbol2vec
    cluster_centers = clf.cluster_centers_
    for c in range(n_clusters):
        cluster_indexes = np.where(clf.labels_ == c)[0]
        name_indexes    = set(names[x] for x in cluster_indexes)
        fname_classes[c] = name_indexes

    return fname_classes

if __name__ == '__main__':
    config  = Config()
    exp     = Experiment(config)
    exp.load_settings()
    exp.parseFastTextvecFile('/root/desyl/res/symbol2vec/symbol2vec.cbow.vec')
    n_clusters = 32

    symbol2vec = collections.OrderedDict(exp.symbol2vec)
    names   = list(symbol2vec.keys())
    vectors = list(symbol2vec.values())

    X = np.ndarray((len(names), exp.symbol2vec_dims), dtype=np.float64)
    for ind, v in enumerate(vectors):
        if ind == 0:
            continue
        X[ind, :] = v

    #too much RAM, > 1TB
    #clf = DBSCAN(eps=3, min_samples=5, n_jobs=-1)
    #clf = SpectralClustering(assign_labels='discretize', n_clusters=128, random_state=42, n_jobs=64)
    clf = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clf.fit(X)
    #clf.labels_
    fname_classes = labels_to_classes(clf, names, n_clusters)
    elems = list(map(lambda x: len(x[1]), fname_classes.items()))
    classes.utils.pickle_save_py_obj(config, fname_classes, 'fname_classes.{}'.format(n_clusters))
    IPython.embed()
