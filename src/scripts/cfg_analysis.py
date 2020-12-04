#!/usr/bin/python3
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from networkx.drawing.nx_pydot import read_dot
import networkx as nx
import sys, os
import glob
import json
import itertools

sys.path.insert(0, "../classes/")
from binary import Binary
from database import Database
from symbol import Symbol



def symbol_in_database(db, path, vaddr):
    print("Filtering CFG nodes to function start addresses for binary {}".format(path))
    res = db.symbols.aggregate([
            { '$match':     { 'path': path } },
            { '$project' :  { 'max' : { '$add' : ['$vaddr', '$size'] }, 'min' : '$vaddr' } },
            { '$match' :    { 'min' : { '$lte' : vaddr }, 'max' : { '$gt' : vaddr } } }
    ])

    symbs = []
    for symb_proj in res:
        #modify Digraph to fix nodes and/or add additional edges
        symb = Symbol.fromDatabase(db, symb_proj['_id'])
        symbs.append(symb)

    unique_symbs = []
    for symb in symbs:
        if len(list(filter( lambda x: x.vaddr == symb.vaddr, unique_symbs) ) ) == 0:
            unique_symbs.append( symb )

    print("Found {} unique symbol(s) which matched {}".format(len(unique_symbs), vaddr))
    return unique_symbs

#find if address is in database
#find if node is between address and address+size -> set to address
def filter_graph_nodes(G, db, bin_path):
    for node in list( G.nodes() ):
        print("Checking node: {}".format(node))
        faddr = int( node )
        new_nodes = list( map( lambda x: x.vaddr, symbol_in_database(db, bin_path, faddr)) )

        print("Getting out and in nodes")
        #get list of in and out edges
        in_nodes = list( map(lambda x: x[0], list(G.in_edges(node))) )
        out_nodes = list( map(lambda x: x[1], list(G.out_edges(node))) )

        #replace old node with new node(s)
        G.remove_node(node)

        print("Adding out and in nodes to new nodes")
        #NB: Add loops between similar nodes
        for n in new_nodes:
            G.add_node(n)
            for j in in_nodes:
                if len(new_nodes) > 1:
                    print("Adding edge between {} -> {}".format(j, n))
                G.add_edge(j, n)
            for k in out_nodes:
                if len(new_nodes) > 1:
                    print("Adding edge between {} -> {}".format(n, k))
                G.add_edge(n, k)

        #don't add loops
        #print("Adding combination of all new nodes")
        #if len(new_nodes) > 1:
        #    for comb in itertools.permutations( new_nodes, 2 ):
        #        print(comb)
        #        G.add_edge( comb[0], comb[1] )
    return G
        
def reverse_sort_node_dependencies(G):
    return list( reversed( list( nx.algorithms.dag.topological_sort(G) ) ) )

def add_func_deps(G, faddr, child_addrs ):
    print( faddr )
    print( child_addrs )
    assert( type(faddr) == type(1) )
    assert( type(child_addrs) == type([]) )

    G.add_node(faddr)
    for addr in child_addrs:
        G.add_node(addr)
        G.add_edge(faddr, addr)
    return G

def build_digraph(transitions):
    G = nx.DiGraph()
    print(transitions)
    for t in transitions:
        print(t)
        faddr, deps = int(t), transitions[t]
        G = add_func_deps(G, faddr, deps)
    return G


