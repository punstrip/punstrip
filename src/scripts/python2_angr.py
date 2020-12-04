#!/usr/bin/python2
import os
import socket
import angr
from angrutils import *
import argparse
import json
import pickle
import networkx as nx
import glob
from networkx.drawing.nx_pydot import write_dot


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

def do_analysis(ap, white_list, black_list, call_depth=1, context_sensitivity=1):
        #calculate entry point
        #start_addrs=[ ap.entry ].extend( white_list_addrs )
        white_list_addrs = list( map( lambda x: x['vaddr'], white_list) )
        black_list_addrs = list( map( lambda x: x['vaddr'], black_list) )

        #delete old cfgs
        files = glob.glob('/root/desyl/src/scripts/cfgs/*')
        for f in files:
            os.remove(f)
        os.chdir('/root/desyl/src/scripts/cfgs')

        desyl_cfg = {}
        for func in white_list:

            start_addrs = [ func['vaddr'] ]
            name = func['name']

            print "[+] Starting CFG analysis for {}...".format(name)
            cfg = ap.analyses.CFGAccurate(call_depth=call_depth, starts=start_addrs, context_sensitivity_level=context_sensitivity, avoid_runs=black_list_addrs,enable_symbolic_back_traversal=False,enable_advanced_backward_slicing=False)
#cfg = ap.analyses.CFGFast(call_depth=call_depth, avoid_runs=black_list_addrs,enable_symbolic_back_traversal=False,symbols=False,force_complete_scan=False,function_prologues=False,start_at_entry=False,function_starts=start_addrs)
            write_dot(cfg.graph, "{}.cfg".format(name))
            deps = []
            for vaddr, function in  cfg.kb.functions.iteritems():
                print "{} -> {} # {} -> {}".format(func['vaddr'], vaddr, name, function.name)
                if vaddr != func['vaddr']:
                    deps.append( vaddr )
            desyl_cfg[ func['vaddr'] ] = deps
        print func['name']
        return desyl_cfg

if __name__ == '__main__':
    print "[+] Starting Python2 ANGR Analyser!"

    #pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)

    address = "../../res/python2_angr.unix.socket"
    #remove previous occourance
    if os.path.exists( address ):
        os.unlink(address)

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind( address )
    s.listen(1)
    print "[+] Listening to {}\n".format( address )
    while True:
        conn, addr = s.accept()
        print "[+]Accepted new connection!\n"
        #print "Connected by ", str(addr)
        #pool.submit( do_vexing, conn )
        bin_path, black_list, white_list = load_settings( conn )
        ap = create_angr_project( bin_path )
        entry_addr = 0
        cfg_transitions = do_analysis(ap, white_list, black_list)

        conn.send( json.dumps( cfg_transitions).encode('utf-8') )
        print "\t[+] Done!\n"
        conn.close()


