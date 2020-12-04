#!/usr/bin/python3
import sys
import pickle
import json
import numpy as np
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pprint
import seaborn as sns
import pandas as pd
import logging

import networkx as nx
#from networkx.drawing.nx_pydot import write_dot
import pygraphviz
from networkx.drawing import nx_agraph


import context
from classes.binary import Binary
from classes.database import Database
from classes.symbol import Symbol
from classes.config import Config
from scripts.gk_weisfeiler_lehman import GK_WL

np.set_printoptions(threshold=np.inf,linewidth=np.inf)

cfg = Config()

logger = logging.getLogger( cfg.logger )
logging.basicConfig(level=logging.INFO , format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger.setLevel(logging.INFO)

db_ref = Database()
db = db_ref.client

sym_a = db.symbols.find_one({'size': { '$gt' : 300, '$lt': 500} })
sym_b = db.symbols.find_one({'size': { '$gt' : 900, '$lt': 1200  } })


with open("graph5.dot", "w") as f:
	f.write(sym_a['bbs_cfg'])

with open("graph6.dot", "w") as f:
	f.write(sym_b['bbs_cfg'])

sa = Symbol.fromDict(sym_a)
sb = Symbol.fromDict(sym_b)


#print(sa.to_str_full())
#print(sb.to_str_full())

print(sa)
print(sb)

print( sa.similarity(sb) )

print("SA BBS_CFG:")
print(sa.bbs_cfg)

#sys.exit(0)

kern_it = 10

kern = GK_WL()
res = kern.compare( sa.bbs_cfg, sa.bbs_cfg, h=kern_it, node_label=False )
print( res )


sc = sb.clone()
sc.bbs_cfg.remove_node( list(sc.bbs_cfg.nodes())[-1] )
res = kern.compare( sb.bbs_cfg, sc.bbs_cfg, h=kern_it, node_label=False )
print( res )


res = kern.compare( sb.bbs_cfg, sa.bbs_cfg, h=kern_it, node_label=False )
print( res )
