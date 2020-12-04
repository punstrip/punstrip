#!/usr/bin/python2
import socket, archinfo, pyvex, json, binascii
import concurrent.futures
import vex_classification as vexc
import functools
import copy
import numpy as np

def _count_operations(self, operations):
    op_dict = {}
    for op in operations:
        if op not in op_dict:
            op_dict[op] = 1
        else:
            op_dict[op] += 1
    return op_dict

def _count_tags(self, vex_object_list):
    tag_dict = {}
    for vex_obj in vex_object_list:
        tag = vex_obj.tag
        if tag not in tag_dict:
            tag_dict[tag] = 1
        else:
            tag_dict[tag] += 1
    #tag_list = []
    #for tag in tag_dict:
    #    tag_list.append( [ tag, tag_dict[tag] ] )
    #return tag_list
    return tag_dict





def __vex_sum_dict_elements(x, y):
    if y in x:
        x[y] += 1
    else:
        x[y] = 1
    return x

def _vex_gen_constants( consts, r2_hndlr ):
        #remove jumps to +- 128 from pc
        near_and_long_jumps = set(filter(lambda x: x > self.vaddr + 128 or x < self.vaddr - 128, consts))

        remove_small_consts = set(filter(lambda x: x > 256, near_and_long_jumps))

# I cannot represent a set in JSON but want a uniuqe set of constants and types.
# I cannot represent this as a set ( type, value ) because ordering is lost when printing to string (used in __eq__)
# Option a) use a dict { type: a, value: b }, or just use a single list [a,b]
# Cannot use dict{ a : b } as there are multiple a's
# Cannot use dict{ b : a } as b is an int
# [ [a,b], [a,b], [a,b], .... ]

def _vex_uniq_constants( const_2d_list ):
    uniq_consts = set( map( lambda x: str(x[0]) + "\t" + str(x[1]) , const_2d_list) )
    return list( map( lambda x: [ x.split("\t")[0] , x.split("\t")[1] ], uniq_consts ) )








def do_vexing(conn):
    #try:
    res = conn.recv(1024 * 1024 * 1024)
    #print res
    query = json.loads(res.decode('utf-8'))
    #print query

    vaddr = query['vaddr']
    print "Input data:"
    print query['data']
    data = binascii.unhexlify(query['data'][2:-1])
    arch = query['arch']

    assert( len(data) > 0 )

    #print "[+] Converting bytes to VEX IR"
    #recompile libvex and change

    if arch == "x86_64":
        irsb = pyvex.IRSB(data, vaddr, archinfo.ArchAMD64())
    elif arch == "ARMv7":
        ### Iend_BE == big endian
        #irsb = pyvex.IRSB(data, vaddr, archinfo.ArchARM('Iend_BE'))
        #irsb = pyvex.IRSB(data, vaddr, archinfo.ArchARM('Iend_BE'))
        irsb = pyvex.IRSB(data, vaddr, archinfo.ArchARM())
        #irsb = pyvex.IRSB(data, vaddr, archinfo.ArchAArch64())
    elif arch == "PPC64":
        irsb = pyvex.IRSB(data, vaddr, archinfo.ArchPPC64())

    #import IPython
    #IPython.embed()

    vex = {}
    #print "converted to VEX IR"
    #irsb.pp()

    vex['ntemp_vars'] = copy.deepcopy(irsb.tyenv.types_used)
    #print "copied ntemp vars"
    vex['temp_vars'] = functools.reduce( __vex_sum_dict_elements, irsb.tyenv.types, {} )
    ### mongodb need strings as keys and can only handle 8 byte ints, get 18446744073709550091
    #print "got temp vars"
    vex['constant_jump_targets'] = { str(k) : value for k,value in  copy.deepcopy(irsb.constant_jump_targets_and_jumpkinds).items() }
    consts = list( map( lambda x: [ str(x.type), str(x) ] , irsb.all_constants) )

    #print "got constants"
    vex['constants'] = _vex_uniq_constants( consts )


    #print "got uniq consts"
    ##add basic block callees #i.e. this basic block calls
    constant_jump_targets = list(irsb.constant_jump_targets )

    #print "about to calculate categorisies"
    vex_statements = functools.reduce( lambda x, y: x + y, map( lambda x: vexc.catagorise_vex_statement(x), irsb.statements ), np.zeros( (len(vexc.CAT_INST_LIST),), dtype=np.uint) )
    #print "got statements"
    vex_expressions = functools.reduce( lambda x, y: x + y, map( lambda x: vexc.catagorise_vex_expression(x), irsb.expressions ), np.zeros( (len(vexc.CAT_EXPR_LIST),), dtype=np.uint) )
    #print "got expressions"
    vex_operations = functools.reduce( lambda x, y: x + y, map( lambda x: vexc.catagorise_vex_operation(x), irsb.operations),  np.zeros( (len(vexc.CAT_OPER_LIST),), dtype=np.uint ) )
    #print "got operations"

    vex['operations'] = vex_operations.tolist()
    vex['expressions'] = vex_expressions.tolist()
    vex['statements'] = vex_statements.tolist()

    vex['jumpkind'] = irsb.jumpkind
    vex['ninstructions'] = irsb.instructions #number of vex instructions! 200 bytes -> 3 vex instr

    print "sending " + json.dumps( vex ).encode('utf-8')
    conn.send( json.dumps( vex ).encode('utf-8') )
    conn.close()


pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

address = "/tmp/python2_vex.unix"
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#s.bind( ("localhost", 3001) )
s.bind( address )
s.listen(1)
#print "[+] Listening to 127.0.0.1:3001"
print "[+] Listening to %s" % address
while True:
    conn, addr = s.accept()
    print "Connected by ", str(addr)
    #pool.submit( do_vexing, conn )
    do_vexing(conn)
