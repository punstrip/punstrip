#!/usr/bin/pyhton3

import copy
import random
import json
import os, sys, functools
import re
import r2pipe
import binascii
import subprocess
import pymongo
import pprint
import math
import hashlib
import re
import networkx as nx
from networkx.drawing import nx_agraph
from networkx.drawing.nx_pydot import write_dot
import logging
from intervaltree import Interval, IntervalTree
from threading import Thread, Lock
import IPython
import claripy
import tqdm

from asciitree import LeftAligned
from collections import OrderedDict as OD

import context
from classes.symbol import Symbol
from classes.database import Database
from classes.binary import Binary
from classes.basicblock import BasicBlock
from classes.config import Config
from classes.symbolic_execution_engine import SymbolicExecutionEngine
from classes.basicblocksequence import LiveVariableAnalysis, ConstantPropagation, TaintPropagation
from classes.basicblock import LiveVariables
import classes.utils

tr = LeftAligned()

class UAFScanner:
    bad_rets = [
        "read", "write", 
        "open", "openat", "creat", "openat2", "close",
        "fopen", "fclose", "fdopen", "freopen",
        "malloc"
    ]

    allocs = set([ 'malloc', 'alloc', 'calloc', 'realloc', 'brk', 'kmalloc', 't_malloc0'])
    frees   = set(['free', 'kfree'])

    def __init__(self, config, binary):
        classes.utils._desyl_init_class_(self, config)
        self.binary = binary
        #self.binary.analyse(SSE_ANALYSIS=True)
        self.see    = SymbolicExecutionEngine(config, binary)

    def taint_return_from_calls(self, s:Symbol, call_vaddrs:set):
        """
            Find variable name of variables that need to be tainted
        """
        tainted_vars = set()
        for bb in s.bbs:
            for exit_vaddr, exit_type in bb.exits:
                if exit_type == 'Ijk_Boring' and exit_vaddr in call_vaddrs:
                    ##return variable is returned from this function
                    #tainted_vars.add('__FUNC_RET__')
                    ##find functions that call this function, and return those instead
                    for caller in s.callers:
                        caller_s = self.binary.get_symbol(caller)
                        tainted_vars |= self.taint_return_from_calls(caller_s, {s.vaddr})
                if exit_type == 'Ijk_Call' and exit_vaddr in call_vaddrs:
                    ##need to add variable assignment to taint
                    ##get the assumed return basicblock
                    for ev, et in bb.exits:
                        if et == 'Ijk_AssumedRet':
                            ###taint all return registers 
                            tainted_vars.add('reg_bb{}_rax'.format(ev))
        return tainted_vars

    def taint_targets_bb(self, s:Symbol, targets:set):
        """
            returns flows and tainted basicblocks
        """
        tpa = TaintPropagation(self.config, s)
        lva = LiveVariableAnalysis(self.config, s)
        live_args, live_heap, live_thread_local_storage, live_locals, local_stack_bytes, num_locals, resolved = lva.analyse(self.see)
        flows = tpa.basicblock_analyse(self.binary, targets, resolved=resolved)
        print("Finished taint analysing for", s.name, "using", str(targets))
        return flows

    def taint_targets(self, s:Symbol, targets:set):
        """
            returns flows and tainted basicblocks
        """
        tpa = TaintPropagation(self.config, s)
        lva = LiveVariableAnalysis(self.config, s)
        live_args, live_heap, live_thread_local_storage, live_locals, local_stack_bytes, num_locals, resolved = lva.analyse(self.see)
        flows = tpa.analyse(self.binary, targets, resolved=resolved)
        print("Finished taint analysing for", s.name, "using", str(targets))
        tainted_bbs = set()
        for bb_ind, state in tpa.state.items():
            if len(state[0]) > 0:
                ##bb is tainted prior to execution
                tainted_bbs.add(s.bbs[bb_ind].vaddr)

        return flows, tainted_bbs



    def frack_targets_bb(self, s:Symbol, targets:set):
        ##produce a subgraph of where taint touches in binary
        ##funcs should touch targets
        fracked = set()
        if len(targets) == 0:
            return fracked

        flows = uafs.taint_targets_bb(s, targets)
        for bb_start, bb_end, tainted in flows:
            if len(tainted) > 0:
                ##hello
                if isinstance(bb_start, int):
                    fracked.add(bb_start)
                if bb_end == '__FUNC_RET__':
                    ##find call site in every calling function and aint from then onwards
                    for u in s.callers:
                        ##ignor erecursive calls
                        if u == s.name:
                            continue

                        caller = self.binary.get_symbol(u)
                        if not caller:
                            print("Could not find caller")
                            IPython.embed()
                            fracked.add(u.vaddr)

                        callee_taint_targets = self.taint_return_from_calls(caller, [s.vaddr])
                        target_bbs = set()
                        for target in callee_taint_targets:
                            if 'reg_bb' not in target or '_rax' not in target:
                                print("malformed taregt")
                                print(target)
                                IPython.embed()
                            target_bbs.add( int(re.match(r'reg_bb(\d+)_rax', target).group(1)) )
                        for ret_vaddr in target_bbs:
                            fracked.add(int(ret_vaddr))
                            for t in tainted:
                                fracked.add("{}::{}::{}".format(bb_start, ret_vaddr, t))

                        fracked |= self.frack_targets_bb(caller, callee_taint_targets)
                    continue

                if isinstance(bb_end, int):
                    fracked.add(bb_end)
                for t in tainted:
                    fracked.add("{}::{}::{}".format(bb_start, bb_end, t))
    
                #non_const_exit callee
                if not isinstance(bb_end, int):
                    continue

                callee = self.binary.symbol_mapper[bb_end]
                if not callee:
                    self.logger.error("Could not locate callee - {}".format(bb_end))
                    continue
                ##ignore recursive calls
                if callee.name == s.name:
                    continue
                fracked |= self.frack_targets_bb(callee, tainted)
        return fracked

    def frack_targets(self, cg:nx.DiGraph, s:Symbol, targets:set):
        ##produce a subgraph of where taint touches in binary
        ##funcs should touch targets
        fracked = set()
        tainted_bbs = set()
        if len(targets) == 0:
            return fracked

        flows, _tainted_bbs = uafs.taint_targets(s, targets)
        tainted_bbs |= _tainted_bbs
        for func, tainted in flows:
            if len(tainted) > 0:
                ##hello
                for t in tainted:
                    fracked.add("{}::{}::{}".format(s.name, func, t))

                if func == '__FUNC_RET__':
                    if s.name in BasicBlock.NON_RETURNING_FUNCTIONS:
                        ##cannot return value from exit
                        continue
                    ##find call site in every calling function and aint from then onwards
                    for u, v in cg.in_edges(s.name):
                        assert(s.name == v)
                        fracked.add(u)
                        caller = self.binary.get_symbol(u)
                        if not caller:
                            print("Could not find caller")
                            IPython.embed()
                        callee_taint_targets = self.taint_return_from_calls(caller, [s.vaddr])
                        _fracked, _tainted_bbs = self.frack_targets(cg, caller, callee_taint_targets)
                        fracked     |= _fracked
                        tainted_bbs |= _tainted_bbs
                    continue

                fracked.add(func)
                callee = self.binary.get_symbol(func)
                if not callee:
                    self.logger.error("Could not locate callee - {}".format(func))
                    continue
                _fracked, _tainted_bbs = self.frack_targets(cg, callee, tainted)
                fracked     |= _fracked
                tainted_bbs |= _tainted_bbs
        return fracked, tainted_bbs

    def possible_uaf_traces(self):
        malloc_sources  = list(filter(lambda x: len(allocs & s.closure) > 0, self.binary.symbols))
        free_sinks      = list(filter(lambda x: len(frees & s.closure) > 0, self.binary.symbols))

        ##walk binary callgraph and find execution traces from sources to sinks
        cg = self.binary.cg
        possible_executions = []
        for source in malloc_sources:
            for sinks in free_sinks:
                if nx.has_path(cg, source, sink):
                    possible_executions.append((source, sink))

        print(possible_executions)

    def find_calls_to(self, symbol_name):
        if isinstance(symbol_name, set):
            return list(filter(lambda x: len(symbol_name.intersection(x.callees)) > 0, self.binary.symbols))
        return list(filter(lambda x: symbol_name in x.callees, self.binary.symbols))

    def in_the_path_of(cg:nx.DiGraph, nodes:set):
        """
            Returns a subgraph of nodes that have a feasible path to or from ANY node in nodes
        """
        subgraph_nodes = set([])
        nodes_in_cg = set(cg.nodes()).intersection(nodes)
        for n in cg.nodes():
            for node in nodes_in_cg:
                if nx.has_path(cg, n, node) or nx.has_path(cg, node, n):
                    subgraph_nodes.add(n)
                    continue

        return cg.subgraph(subgraph_nodes)

    def eliminate_noncritical_paths(cg:nx.DiGraph, start, end):
        """"
            Remove basicblocks not on critical path
        """
        pass

    def prettify_bb_cfg(self, cfg:nx.DiGraph, alloc_targets:set, free_targets:set):
        new_cfg = nx.DiGraph()
        for node in cfg.nodes():
            if not isinstance(node, int):
                continue
            s   = self.binary.symbol_mapper[node]
            bb  = self.binary.basicblock_mapper[node]
            if not bb:
                ###bb is a call to linked library
                """
                for interval in self.binary.vaddr_to_name_tree.at(node):
                    label       = '{} :: imp'.format(interval.data)
                    fillcolor   = 'orange'
                    style       = 'filled'
                    new_cfg.add_node(node, label=label, fillcolor=fillcolor, style=style)
                """
                continue

            alloc_bb, free_bb = False, False
            for exit_vaddr, exit_type in bb.exits:
                if exit_vaddr in free_targets:
                    free_bb = True

            if bb.vaddr in alloc_targets:
                alloc_bb = True

            if bb.vaddr in free_targets:
                free_bb = True

            label       = '{} :: {} :: {}'.format(s.name, node, hex(node))
            fillcolor   = 'green' if alloc_bb else 'red' if free_bb else 'white'
            style       = 'filled'
            new_cfg.add_node(node, label=label, fillcolor=fillcolor, style=style)

        ##copy edges after modifying node data
        ##node indexes are the same
        for u, v in cfg.edges():
            new_cfg.add_edge(u, v)

        return new_cfg

    def feasible_paths_to(cg:nx.DiGraph, nodes:set):
        """
            Returns a subgraph of nodes that have a feasible path to ANY node in nodes
        """
        subgraph_nodes = set([])
        nodes_in_cg = set(cg.nodes()).intersection(nodes)
        for n in cg.nodes():
            for node in nodes_in_cg:
                if nx.has_path(cg, n, node):
                    subgraph_nodes.add(n)
                    break

        return cg.subgraph(subgraph_nodes)

    def _rec_find_flows_from(self, s, all_flows, tainted=set([])):
        tree = { s.name : OD([]) }
        func_flows = {}
        flows = self.see.execute_function( s, orig_tainted=tainted )
        #consolidate flows into single func->args
        for flow in flows:   
            if flow in all_flows:
                continue
            all_flows.add(flow)

            flow_name, flow_arg = flow
            if flow_name == s.name:
                continue

            if flow_name in func_flows:
                func_flows[ flow_name ].add(flow_arg)
            else:
                func_flows[ flow_name ] = set([flow_arg])

        ##propagate taints
        for f, a in func_flows.items():
            if len(list(filter(lambda x: x.name == f, self.see.binary.symbols))) != 0:
                tree[s.name][f + ":" +str(a)] = OD( self._rec_find_flows_from( self.see.binary.get_symbol(f), all_flows, tainted=a) )[f]
            else:
                tree[s.name][f + ":" + str(a)] = {}

        print("Done")
        return tree

    def uses_tainted(self, bb:BasicBlock, tainted:set):
        s = self.binary.symbol_mapper[bb.vaddr]
        cpa         = ConstantPropagation(self.config, s)
        resolved    = cpa.propagate(see=self.see)

        lv = LiveVariables(self.config, bb)
        V = lv.live_variables(resolved_variables=resolved)
        tainted_and_live = tainted & V
        if len(tainted_and_live):
            print("Bingo!")
            return True
        return False

    def rec_scan(self):
        while True:
            TAINT_RETS = len(self.see.taint_rets_from)
            for s in self.see.binary.symbols:
                flows = self.see.execute_function(s)
                for name, arg in flows:
                    if name == s.name and arg == 'rax':
                        ###return is tainted
                        self.see.taint_rets_from.add(name)

            if TAINT_RETS == len(self.see.taint_rets_from):
                break
        
        return TAINT_RETS

    def scan(self):
        with tqdm.trange(len(self.see.binary.symbols), desc='Symbols') as t:
            for sym_index in t:
                t.set_postfix(name=self.see.binary.symbols[sym_index].name.ljust(20)[-20:])
                s = self.see.binary.symbols[sym_index]
                flow_tree = uafs._rec_find_flows_from( s, set([]) )
                ascii_tree = tr(flow_tree)
                if 'free' in ascii_tree:
                    print(ascii_tree)

        ##find arguments passed to free
        ##find if they are shared arguments to other functions
        ##take walks on a CFG

if __name__ == '__main__':
    config = Config()
    config.logger.setLevel(logging.INFO)
    binaries = classes.utils.pickle_load_py_obj(config, 'train_binaries')
    for path in binaries:
        #path = '/dbg_elf_bins/libatlas-test/usr/lib/x86_64-linux-gnu/atlas/xzr1time'
        #path = '/root/uaf'
        path= '/dbg_elf_bins/sane/usr/bin/xscanimage'
        b = Binary(config, path=path, must_resolve_libs=False)
        b.analyse()
        cg = b.callgraph

        #see = SymbolicExecutionEngine(config, a)
        uafs = UAFScanner(config, b)
        alloc_funcs = uafs.find_calls_to(UAFScanner.allocs)
        alloc_funcs_names = set(map(lambda x: x.name, alloc_funcs))

        ##combine set of execute from main and alloc_funcs
        executed_from_main = UAFScanner.in_the_path_of(cg, {'main'})
        executed_alloc_funcs = list(filter(lambda x: x.name in executed_from_main.nodes(), alloc_funcs))

        ##find potential use after frees
        alloc_vaddrs    = list(map(lambda x: x['vaddr'], filter(lambda x: x['name'] in UAFScanner.allocs, b.dyn_imports)))
        free_vaddrs     = list(map(lambda x: x['vaddr'], filter(lambda x: x['name'] in UAFScanner.frees, b.dyn_imports)))


        fracked_bbs = set()
        for s in executed_alloc_funcs:
            targets = uafs.taint_return_from_calls(s, alloc_vaddrs)
            tainted = uafs.frack_targets_bb(s, targets)
            bbs = set(filter(lambda x: isinstance(x, int) and uafs.binary.basicblock_mapper[x], tainted))
            tainted_cfg = uafs.binary.cfg.subgraph(bbs)

            target_bbs = set()
            for target in targets:
                target_bbs.add( int(re.match(r'reg_bb(\d+)_rax', target).group(1)) )

            pretty_tainted_cfg = uafs.prettify_bb_cfg(tainted_cfg, target_bbs, free_vaddrs)
            classes.utils.save_graph(pretty_tainted_cfg, '/tmp/fracked_cfg.dot')

            fracked_flowgraph = nx.DiGraph()
            for taint in tainted:
                if isinstance(taint, str):
                    if '::' in taint:
                        u, v, _t = taint.split('::')
                        if u == 'non_const_exit' or v == 'non_const_exit':
                            uafs.logger.warning("Removing non_const_exit")
                            continue

                        node_u, node_v = int(u), int(v)
                        if uafs.binary.symbol_mapper[node_u] and uafs.binary.symbol_mapper[node_v]:
                            fracked_flowgraph.add_edge(node_u, node_v)

            pretty_fracked_flowgraph = uafs.prettify_bb_cfg(fracked_flowgraph, target_bbs, free_vaddrs)
            classes.utils.save_graph(pretty_fracked_flowgraph, '/tmp/fracked_flow.dot')

            freeing_nodes = set()
            for bb in tainted_cfg.nodes():
                bb_obj = uafs.binary.basicblock_mapper[bb]
                if not bb_obj:
                    continue
                for vaddr, jk in bb_obj.exits:
                    if vaddr in free_vaddrs:
                        freeing_nodes.add(bb)

            for free_call_site in freeing_nodes:
                for node in pretty_fracked_flowgraph.nodes():
                    if node in freeing_nodes:
                        continue
                    if nx.has_path(pretty_fracked_flowgraph, free_call_site, node):
                        ##extarct tainted variables into node
                        uafs.logger.info("Found potential UAF flow...")

                        bb_tainted = set()
                        for t in tainted:
                            if isinstance(t, str):
                                u, v, var = t.split('::')
                                if v == 'non_const_exit':
                                    continue
                                if node == int(v):
                                    bb_tainted.add(var)

                        if uafs.uses_tainted(uafs.binary.basicblock_mapper[node], bb_tainted):
                            uafs.logger.critical("Potential use after free found! Allocated in {}, freed in {} and then used in {}".format(target_bbs, free_call_site, node))

                            sp_flowgraph = pretty_fracked_flowgraph.subgraph(nx.shortest_path(pretty_fracked_flowgraph, source=free_call_site, target=node))
                            classes.utils.save_graph(sp_flowgraph, '/tmp/fracked_flow_shortest.dot')


                            print("Paused...")
                            IPython.embed()

        continue
        sys.exit()

        """
        BELOW IS FUNCTION LEVEL ANALYSIS

        """

        fracked_nodes = set()
        for s in executed_alloc_funcs:
            targets = uafs.taint_return_from_calls(s, alloc_vaddrs)
            tainted, bbs = uafs.frack_targets(cg, s, targets)
            fracked_nodes |= tainted
            if 'free' in tainted:
                print("I found an allocation that propagates it's taint to free")
                ##check number of incoming nodes to free
                ##if more than 1 -> potential for double free

                target_bbs = set()
                for target in targets:
                    target_bbs.add( int(re.match(r'reg_bb(\d+)_rax', target).group(1)) )

                #fracked_cg_nodes = set(filter(lambda x: '::' not in x, tainted))
                #fracked_cg = copy.deepcopy(cg.subgraph(fracked_cg_nodes))
                g = nx.DiGraph()
                for taint in tainted:
                    if '::' in taint:
                        u, v, _t = taint.split('::')
                        g.add_edge(u, v)

                for u, v in g.in_edges('__FUNC_RET__'):
                    for a, b in cg.in_edges(u):
                        g.add_edge(u, a)
                g.remove_node('__FUNC_RET__')

                IPython.embed()
                sub_symbs   = list(filter(lambda x: x, list(map(lambda x: uafs.binary.get_symbol(x), g.nodes()))))
                sub_cfg = uafs.binary.build_cfg_from_symbols(sub_symbs)

                pretty_cfg = uafs.prettify_bb_cfg(sub_cfg, target_bbs, free_vaddrs)

                for vaddr in list(target_bbs) + free_vaddrs:
                    if vaddr in pretty_cfg.nodes():
                        pretty_cfg.nodes[vaddr]['fontname'] = 'times-bold'
                        pretty_cfg.nodes[vaddr]['shape']    = 'hexagon'
                classes.utils.save_graph(pretty_cfg, '/tmp/alloction_to_free.dot')
                IPython.embed()

            else:
                ##we have a meory leak
                config.logger.warning("Memory leak detected")
                g = nx.DiGraph()
                for taint in tainted:
                    if '::' in taint:
                        u, v, _t = taint.split('::')
                        g.add_edge(u, v)

                for u, v in g.in_edges('__FUNC_RET__'):
                    for a, b in cg.in_edges(u):
                        g.add_edge(u, a)
                g.remove_node('__FUNC_RET__')

                g.nodes[s.name]['fillcolor'] = 'green'
                g.nodes[s.name]['style'] = 'filled'
                classes.utils.save_graph(g, '/tmp/memleak.dot')
                IPython.embed()

        ##TODO: Do this on a basicblock level, then find tainted path from free to another tainted bb (given parallel execution points)

        ##fracked nodes subgraph is more accuracte that just using the callgraph, function cal has to pass the taint
        print("Finsihed fracking allocations")
        fracked_cg_nodes = set(filter(lambda x: '::' not in x, fracked_nodes))
        fracked_cg = cg.subgraph(fracked_cg_nodes)
        IPython.embed()

        sys.exit()



        ###below is for binary rewriting, subgraph of original functions

        ##subgraph containing critical paths that allocate memory
        allocs_cg = UAFScanner.in_the_path_of(cg, alloc_funcs_names)
        
        free_funcs = uafs.find_calls_to(UAFScanner.frees)
        free_funcs_names = set(map(lambda x: x.name, free_funcs))
        sub_cg = UAFScanner.feasible_paths_to(allocs_cg, free_funcs_names)
        ##subgraph that calls free after allocating memory
        if len(sub_cg.nodes()) == 0:
            print("{} has no possible UAFs".format(path))
            continue

        print("============")
        print("...potential")
        IPython.embed()

