#!/usr/bin/python3
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from networkx.drawing.nx_pydot import read_dot
import networkx as nx
import sys, os
import glob
import json
import cfg_analysis as desyl_cfg


"""
    This file build a Dependency DAG of symbols. This is used in reverse topological order for analysis.
"""

def draw_graph(G):
    plt.subplot(111)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


#return [ (faddr, deps addrs) ]
def read_func_deps(d):
    deps = []
    for filename in glob.iglob(d + "/" + "*.transitions"):
        with open(filename, 'r') as file:
            #print(filename)
            for line in file.readlines():
                #print(line)
                trans = line.split(" -> ")
                #print(trans)
                assert( len(trans) == 2)
                deps.append( ( int(trans[0]), json.loads(trans[1]) ) )
    return deps

if __name__ == '__main__':

    db_ref = Database()
    db = db_ref.client

    G = nx.DiGraph()
    if os.path.exists("graph.dot"):
        G = read_dot('graph.dot')
    else:
        transitions = read_func_deps("cfgs")
        G = desyl_cfg.build_digraph(transitions)
        write_dot(G, 'graph.dot')

    print("Number of elements in digraph: {}".format( len(list(G) ) ))
    G = desyl_cfg.filter_graph_nodes(Gi, db, '/root/')
    G = desyl_cfg.reverse_sort_node_dependencies(G)
    draw_graph(G)
