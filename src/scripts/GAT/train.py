import numpy as np
import context
import classes.config
import classes.database
import classes.symbol
import classes.utils
import classes.experiment
import classes.callgraph
import classes.NLP
import dgl
import matplotlib.pyplot as plt
import logging
import networkx as nx
import pandas as pd
##batched graph dataset
from dgl.data import MiniGCDataset
from torch.utils.data import DataLoader, Dataset
import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn.functional as F
import IPython
import functools


class DesylDataset(Dataset):
    """
    DESYL Dataset loader
    config, exp, coll_name
    """

    def __init__(self, config, exp, collection_name, transform=None):
        self.transform = transform
        config.database.collection_name = collection_name
        self.config = config
        self.db = classes.database.Database(config)
        self.binaries_frame = pd.DataFrame(db.distinct_binaries())
        self.exp = exp

    def __len__(self):
        return len(self.binaries_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bin_path = self.binaries_frame.iloc[idx, 0]
        query = { '$match' : { 'path' : bin_path } }
        cg = classes.callgraph.build_callgraph_for_query(self.db, self.exp, query)
        dgl_cg = CGtoDGLG(self.exp, cg)

        if self.transform:
            dgl_cg = self.transform(dgl_cg)


        print("Need to convert DGLG to tensor")
        import IPython
        IPython.embed()


        return (dgl_cg, bin_path)


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits



def CGtoDGLG(cg):
    num_nodes   = len(cg.nodes())
    num_edges   = len(cg.edges())
    node2index  = dict(map(lambda x, y: (x, y), cg.nodes(), range(num_nodes)))
    index2node  = dict(map(lambda x, y: (y, x), cg.nodes(), range(num_nodes)))

    num_edge_features = 1

    if len(cg.nodes()) == 0:
        return dgl.DGLGraph(), None, None, None, None, None

    ex_node = list(cg.nodes())[0]
    example_fp = cg.nodes[ ex_node ]['fingerprint']
    in_dim = np.shape(example_fp)[1]
    
    features   = torch.zeros((num_nodes, in_dim), dtype=torch.float32)
    labels     = torch.zeros((num_nodes, exp.ml_name_vector_dims), dtype=torch.float32)
    mask       = torch.zeros((num_nodes), dtype=torch.bool)
    edge_features = torch.zeros((num_edges, num_edge_features), dtype=torch.float32)
    
    G = dgl.DGLGraph()
    G.add_nodes(num_nodes)

    edge_it = 0
    
    for name in cg.nodes():
        i = node2index[name]
        mask[i] = False
        ##keeps all non text funcs as zeros, TODO: change this
        if cg.nodes[name]['text_func']:
            mask[i] = True
            fp = cg.nodes[name]['fingerprint']
            features[i:] = torch.from_numpy(fp)
        #labels[i] = exp.to_index('label_vector', exp.func_to_label_map[name])
        labels[i,:] = torch.from_numpy(exp.to_vec('ml_name_vector', nlp.canonical_set(name)))
    
        for edge in cg[name]:
            #print(edge)
            j = node2index[edge]
            jk = 1 if cg[name][edge]['call_ref'] else 0
            G.add_edge(i, j)
            edge_features[edge_it,0] = jk
            edge_it += 1

    G.ndata['h'] = features
    G.edata['h'] = edge_features

    return G, features, labels, mask, index2node, node2index



def accuracy(logits, labels):
    ##multiclass accuracy, hamming loss
    predicted = torch.clamp(torch.round(logits), 0, 1)
    #correct = torch.sum(predicted == labels)
    correct = torch.sum(predicted.eq(labels))
    return correct.item() * 1.0 / labels.shape.numel()

import math

def hamming_loss(P, Y):
    """
    Computes the hamming loss between prediction and labels
    
    0 is perfect, 1 is completly incorrect
    
    HL = 1/kn SUM_n[ SUM_k ( XOR(X_nk, Y_nk) )]
    """
    #pred = rescale_plus_minus_one(P)
    #loss = torch.logical_xor(1.0 + pred.int(), 1.0 + Y.int())

    pred = torch.clamp(torch.round(P), 0, 1).int()
    loss = torch.logical_xor(pred, Y.int())

    t_elem = P.shape.numel
    return (1.0 / P.shape.numel()) * torch.sum(loss).item()

def multilabel_precision(P, Y):
    """
    Precision is the proportion of predicted correct laels to the
    total number of actual labels, averaged over all instances
    
    P = 1/n SUM_n( |Y_i AND Z_i| / |Z_i| )
    """
    pred = torch.clamp(torch.round(P), 0, 1)
    #pred = rescale_plus_minus_one(P)
    n, c = P.shape
    class_p = []
    for _c in range(c):
        #Z_i = torch.sum(pred[:,_c]).item()
        Z_i = len(torch.where(pred[:,_c] >= 0.5)[0])
        if Z_i == 0:
            class_p.append(1.0)
            continue
            
        pred_corr_and_true_correct = 0.0
        for i in range(n):
            if pred[i, _c] >= 0.5 and Y[i, _c] >= 0.5:
                pred_corr_and_true_correct +=1.0
        p = pred_corr_and_true_correct / Z_i
        print("Predicted correct and correct", pred_corr_and_true_correct)
        print("Predicted labels", Z_i)
        class_p.append(p)
    return torch.FloatTensor(class_p)

def rescale_plus_minus_one(X):
    """
        Rescale tensor to +-1 
    """
    m = torch.ones(X.shape, dtype=torch.float)
    r, c = torch.where(X <= 0.0)
    for i in range(len(r)):
        _r = r[i]
        _c = c[i]
        m[_r, _c] = -1.0
    return m
    
    
def multilabel_recall(P, Y):
    """
    Recall is the proportion of predicted correct
    labels to the total number of predicted labels
    averaged over all instances
    
    R = 1/n SUM_n( |Y_i AND Z_i| / |Y_i| )
    """
    #import IPython
    #IPython.embed()
    pred = torch.clamp(torch.round(P), 0, 1)
    #pred = rescale_plus_minus_one(P)
    n, c = P.shape
    class_r = []
    for _c in range(c):
        #Y_i = torch.sum(Y[:,_c]).item()
        Y_i = len(torch.where(Y[:,_c] >= 1.0)[0])
        if Y_i == 0:
            class_r.append(1.0)
            continue
        
        pred_corr_and_true_correct = 0.0
        for i in range(n):
            if pred[i, _c] >= 0.5 and Y[i, _c] >= 0.5:
                pred_corr_and_true_correct +=1.0
        r = pred_corr_and_true_correct / Y_i
        class_r.append(r)
    return torch.FloatTensor(class_r)

def generic_f1(p, r):
    return 2.0*p*r/(p+r)
    

def multilabel_accuracy(logits, labels):
    x, num_labels = list(labels.shape)
    conf_mat      = torch.zeros([num_labels, num_labels], dtype=torch.int32)
    predicted = torch.clamp(torch.round(logits), 0, 1)
    tp_c, fp_c, fn_c = [], [], []
    for i in range(num_labels):
        #number of predicted labels where the label was in fact correct
        tp = torch.sum( labels[:, i] * predicted[:, i])
        ##number of samples we incorrectly predicted as this class
        fp = torch.sum(predicted[:, i]) - tp
        #the number of samples of this class that we missed
        fn = torch.sum(labels[:, i]) - tp
        
        tp_c.append(tp)
        fp_c.append(fp)
        fn_c.append(fn)
        
    return sum(tp_c), sum(fp_c), sum(fn_c)

def multilabel_micro_f1(logits, labels):
    tp, fp, fn = multilabel_accuracy(logits, labels)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p +1)
    return f1, p , r


def confusion_matrix(logits, labels):
    x, num_labels = list(labels.shape)
    conf_mat      = torch.zeros([num_labels, num_labels], dtype=torch.int32)
    predicted = torch.clamp(torch.round(logits), 0, 1)
    tp_c, fp_c, fn_c = [], [], []
    for sample in range(x):
        pred_c = predicted[sample, :]
        lab_c  = labels[sample,:]
        
        
        for pred_c in range(num_labels):
            
            conf_mat[true_c, pred_c]
            
            #number of predicted labels where the label was in fact correct
            tp = torch.sum( labels[:, i] * predicted[:, i])
            ##number of samples we incorrectly predicted as this class
            fp = torch.sum(predicted[:, i]) - tp
            #the number of samples of this class that we missed
            fn = torch.sum(labels[:, i]) - tp

            tp_c.append(tp)
            fp_c.append(fp)
            fn_c.append(fn)
        
    return sum(tp_c), sum(fp_c), sum(fn_c)

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)





config = classes.config.Config()
config.logger.setLevel(logging.DEBUG)

collection_name = "dataset_IIII"
config.database.collection_name = collection_name
db = classes.database.Database(config)
nlp = classes.NLP.NLP(config)
binaries = db.distinct_binaries()
exp = classes.experiment.Experiment(config)
exp.load_settings()
print(len(binaries), "distinct binaries")

print("Do you want to do anything before we start?")
IPython.embed()


#"""
if True:
    ##load all binary callgraphs with fingerprints
    if True:
        cgs = []
        for path in tqdm.tqdm(binaries, desc='Binary Callgraphs'):
            query = { '$match' : { 'path' : path } }
            cg = classes.callgraph.build_callgraph_for_query(db, exp, query)
            cgs.append(cg)

        classes.utils.save_py_obj(config, cgs, "cgs")
    else:
        cgs = classes.utils.load_py_obj(config, "cgs")

    #batch graphs for node classification
    #cg_conv_f = functools.partial(CGtoDGLG, exp)
    #dgl_gd = list(map(cg_conv_f, tqdm.tqdm(cgs, desc='NetworkX -> DGL Graph')))
    dgl_gd = list(map(CGtoDGLG, tqdm.tqdm(cgs, desc='NetworkX -> DGL Graph')))
    dgl_g, dgl_f, dgl_l, dgl_m, index2_nodes, node2indexes = zip(*dgl_gd)

    ##save data to pickle for easy loading
    classes.utils.save_py_obj(config, dgl_gd, "dgl_gd")
else:
    dgl_gd = classes.utils.load_py_obj(config, "dgl_gd")

IPython.embed()

#Split dataset into training and testing set
train_size = int(0.95 * len(dgl_gd))
test_size = len(dgl_gd) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dgl_gd, [train_size, test_size])
train_dgl_g, train_dgl_f, train_dgl_l, train_dgl_m, train_index2_nodes, train_node2indexes = zip(*train_dataset)
test_dgl_g,test_dgl_f,  test_dgl_l, test_dgl_m, test_index2_nodes, test_node2indexes = zip(*test_dataset)
train_bg, test_bg             = dgl.batch(train_dgl_g), dgl.batch(test_dgl_g)
train_features, test_features = torch.cat(train_dgl_f), torch.cat(test_dgl_f)
train_labels, test_labels     = torch.cat(train_dgl_l)[:,4], torch.cat(test_dgl_l)[:,4]
train_mask, test_mask         = torch.cat(train_dgl_m), torch.cat(test_dgl_m)


in_features = list(train_bg.ndata['h'].shape)[1]
print(in_features)

train_labels = train_labels.unsqueeze(1)
test_labels   = test_labels.unsqueeze(1)
n_classes = 1
#"""


#desyl_dataset = DesylDataset(config, exp, collection_name)
#data_loader = DataLoader(desyl_dataset, batch_size=4, shuffle=True)

#"""


"""
##implement subsampler
pos_train_mask = torch.where(train_labels == 1.0)
neg_train_mask = torch.where(train_labels == 0.0)

len_pos = len(pos_train_mask[0])
mod_neg_mask = neg_train_mask[0][:len_pos], neg_train_mask[1][:len_pos]

mod_train_labels = torch.cat((train_labels[ pos_train_mask ], train_labels[ mod_neg_mask ]))


pos_features = train_features[pos_train_mask[0], :]
mod_neg_features = train_features[neg_train_mask[0][:len_pos], :]

mod_train_features = torch.cat((pos_features, mod_neg_features))

train_feature   = mod_train_features
train_labels    = mod_train_labels
"""

#calculate the positive weight tensor for Binary Cross Entropy loss
pos_weight = torch.zeros((n_classes,), dtype=torch.float)
#samples, classes = labels.shape
samples = len(train_labels)
for i in range(n_classes):
    #pos_samples = torch.sum(labels[:,i]).item()
    pos_samples = torch.sum(train_labels).item()
    neg_samples = samples - pos_samples
    if pos_samples == 0:
        pos_weight[i] = 1.0
    else:
        #pos_weight[i] = ( neg_samples / pos_samples)
        #pos_weight[i] = ( pos_samples / neg_samples )
        pos_weight[i] = 2 * ( pos_samples + neg_samples ) / pos_samples


#"""

#n_classes = len(exp.ml_name_vector)


negative_slope = 0.3
attention_drop = 0.5
in_drop        = 0.3
num_hidden     = 4
out_dim        = n_classes
num_heads      = 6
num_layers     = 3
num_out_heads  = 1
epochs         = 5000
residual       = True

#assert(train_features.size()[1] == test_features.size()[1])
heads = ([num_heads] * num_layers) + [num_out_heads]
net = GAT(train_bg,
                num_layers,
                in_features,
                num_hidden,
                n_classes,
                heads,
                F.relu,
                in_drop,
                attention_drop,
                negative_slope,
                residual)
#"""

#####
##### loss function defintiion
#####
#loss_fcn = torch.nn.CrossEntropyLoss()
#loss_fcn = torch.nn.MultiLabelMarginLoss()
#loss_fcn = torch.nn.MultiLabelSoftMarginLoss()
#loss_fcn = torch.nn.BCELoss()
#loss_fcn = torch.nn.BCEWithLogitsLoss()
loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')

#####
##### optimizer
#####
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=5e-5)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.8)
#optimizer = torch.optim.SparseAdam(net.parameters())



#test_labels.resize((len(test_labels), 1))

"""
##rescale label space to +-1
train_lab_shape = train_labels.shape
train_neg_ones = torch.ones(train_lab_shape, dtype=torch.float)
train_labels = (2*train_labels) - train_neg_ones

test_lab_shape = test_labels.shape
test_neg_ones = torch.ones(test_lab_shape, dtype=torch.float)
test_labels = (2*test_labels) - test_neg_ones
"""

train_labels = train_labels.float()
test_labels = test_labels.float()

# main loop
epoch_losses = []
for epoch in range(epochs):
    epoch_loss = 0
    #for iter, (bg, labels) in enumerate(data_loader):
        
    ### Applying softmax doesn't make sense here, each label is independent 
    ### and will be rescaled to 1 if applied
    #import IPython
    #IPython.embed()
    raw_train_logits = net(train_features)   
    #raw_train_logits = net(train_bg)   
    #raw_train_logits = net(bg)   
    #train_logits = F.log_softmax(raw_train_logits, 1)
    #train_logits = F.softmax(raw_train_logits)
    train_logits = raw_train_logits



    #loss = loss_fcn(train_logits[train_mask], train_labels[train_mask].float())
    #loss = loss_fcn(bg, labels)
    loss = loss_fcn(train_logits, train_labels)
    
    #predicted = torch.clamp(torch.round(train_logits), 0, 1).double()
    #loss = loss_fcn(predicted[train_mask], train_labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    if epoch % 2 == 0:
        ml_p = multilabel_precision(train_logits, train_labels)
        ml_r = multilabel_recall(train_logits, train_labels)
        ml_f1 = generic_f1(torch.mean(ml_p), torch.mean(ml_r))

        #print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | Train F1 {:.4f}, Train P {:.4f}, Train R {:.4f}, HL: {:.4f}".format(
        #    epoch, loss.item(), t1 - t0, f1, p, r, hamming_loss(train_logits[train_mask], train_labels[train_mask])))
        
        print("Epoch {:05d} | Loss {:.4f} | Train F1 {:.4f}, Train P {:.4f}, Train R {:.4f}, HL: {:.4f}".format(
            epoch, loss.item(), ml_f1, torch.mean(ml_p), torch.mean(ml_r), hamming_loss(train_logits[train_mask], train_labels[train_mask])))

        #print(rescale_plus_minus_one(train_logits[:10,:]))
        #print(train_labels[:10,:])
        #epoch_loss += loss.detach().item()
    #print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))

net.g = test_bg
test_logits = net(test_features)
test_acc = accuracy(test_logits[test_mask], test_labels[test_mask])

print("Test Acc {:.4f}".format(test_acc))
IPython.embed()



