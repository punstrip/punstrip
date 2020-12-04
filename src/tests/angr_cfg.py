#!/usr/bin/python2
import os
import socket
import angr
from angrutils import *
import argparse
import json

#Start and use a single Angr Project for each binary!
def create_angr_project(bin_path):
    print "[+] Creating ANGR Project for binary: {}".format(bin_path)
    return angr.Project(bin_path, load_options={'auto_load_libs': True})

def load_settings(conn):
    """
    data_frame = {
        bin_path: "",
        black_list: { name: "", vaddr: 0 }
        white_list: { name: "", vaddr: 0 }
    }
    """
    data = conn.recv(1024 * 1024 * 1024)
    buf = json.loads(data.decode('utf-8'))
    return buf['bin_path'], buf['black_list'], buf['white_list']

def do_analysis(ap, call_depth=3, context_sensitivity=1):
        #calculate entry point
        #start_addrs=[ ap.entry ].extend( white_list_addrs )
        #start_addrs=[ 0x000402c50 ]
        start_addrs = [ 0x0000038c0 ]

        print "\t[+] Starting CFG analysis..."
        cfg = ap.analyses.CFGAccurate(call_depth=call_depth, starts=start_addrs, context_sensitivity_level=context_sensitivity)
        netx_cfg = cfg.graph
        netx_callgraph = cfg.functions.callgraph

        print "\t[+] Creating CFG"
        plot_cfg( cfg, "test.cfg" )
        print "\t[+] Creating CG"
        plot_cg( cfg.kb, "test.cg" )

        print "\t[+] Creating DFG"
        DataFlowGraph           = ap.analyses.DFG(cfg)
        plot_dfg( DataFlowGraph, "test.dfg" )

        #Then need to find which functions contain loops 
        LoopsInWholeBinary      = ap.analyses.LoopFinder()

        for symb_addr in white_list_addrs:
            print "\t[+] Creating DDG"
            DataDependencyGraph     = ap.analyses.DDG(cfg, symb_addr, call_depth=call_depth, block_addrs=black_list_addrs)
            plot_ddg( DataDependencyGraph, "test.ddg")

            print "\t[+] Creating CDG"
            ControlDependencyGraph  = ap.analyses.CDG(cfg, symb_addr)
            plot_cdg( cfg, ControlDependencyGraph, "test.cdg")

            print "\t[+] Creating VFG"
            ValueFlowGraph          = ap.analyses.VFG(cfg=cfg,function_start=symb_addr,avoid_runs=black_list_addrs)
            ValueSetAnalysis_DataDependencyGraph    = ap.analyses.VSA_DDG( vfg=ValueFlowGraph, start_addr=symb_addr, interfunction_level=1)
            return

        #Could use ANGR girlscout to find functions in binaries?
        #ap.analyses.girlscount()


        #graphs are in the form of network x
        #graph.difference
        #graph.is_isomorphic
        cfg.draw()



if __name__ == '__main__':
    print "[+] Starting Python2 ANGR Analyser!"
    bin_path = "/root/friendly-corpus/bin//dynamic//gcc//o2/cat"
    ap = create_angr_project( bin_path )
    do_analysis(ap)
