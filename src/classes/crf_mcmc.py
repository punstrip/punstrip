#!/usr/bin/pyhton3
import copy
import random
import math
import os, sys, functools
import networkx as nx
from networkx.drawing import nx_agraph
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
import logging
from threading import Thread, Lock
import collections

import context
from classes.config import Config
import classes.utils
import classes.NLP
from classes.crf import CRF

import IPython


class MarkovChainMonteCarloCRFInference(CRF):
    def __init__(self, crf):
        super().__init__(crf.config, crf.knowns, crf.unknowns, crf.callgraph, crf.fingerprints, crf.relationships, crf.name_to_index, crf.orig_constraints)

