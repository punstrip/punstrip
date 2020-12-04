#!/usr/bin/python2 -u
import sys

sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/pyvex-7.7.9.14-py2.7.egg')

import socket, archinfo, pyvex, json, binascii
import concurrent.futures

def _count_operations(operations):
    #    operations = get_operations_from_cfg(cfg)
    op_dict = {}
    for op in operations:
        if op not in op_dict:
            op_dict[op] = 1
        else:
            op_dict[op] += 1
    return op_dict

def _count_tags(vex_object_list):
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

def do_vexing(b):
    #try:
    #print a
    #print a[0][2:-1]
    vaddr = 0x4000
    bytes = binascii.unhexlify(b[2:-1])
    #bytes = binascii.unhexlify(a[0][2:-1])
    #bytes = binascii.unhexlify(b)
    #print "Received {}", str(vaddr), "\t", str(bytes)

    #assert(vaddr > 0)
    assert( len(bytes) > 0 )
    print len(bytes)

    #print "[+] Converting bytes to VEX IR"
    #recompile libvex and change
    irsb = pyvex.IRSB(bytes, vaddr, archinfo.ArchAMD64(), max_bytes=30000, max_inst=30000)
    #print "[+] Done"

    vex = {}
    vex['temp_var_types'] = irsb.tyenv.types
    vex['ntemp_vars'] = len( irsb.tyenv.types )
    vex['operations'] = list( map( lambda x: str(x), irsb.operations) )
    vex['expressions'] = list( map( lambda x: { 'tag':x.tag, 'value': str(x) }, irsb.expressions) )
    vex['constants'] = list( map( lambda x: { 'type':  x.type, 'value': str(x) } , irsb.all_constants) )
    vex['statements'] = list( map( lambda x: { 'tag': x.tag, 'value': str(x) }, irsb.statements) )

    vex['sum_statements'] = _count_tags( irsb.statements )
    vex['sum_operations'] = _count_operations( irsb.operations )
    vex['sum_expressions'] = _count_tags( irsb.expressions )

    vex['jumpkind'] = irsb.jumpkind
    vex['ninstructions'] = irsb.instructions #number of vex instructions! 200 bytes -> 3 vex instr

    #print("Sending response!!!!")
    #print(vex)
    print "Done!"


with open('large_func_hex', 'r') as f:
    xs = f.read()
    #binary = binascii.unhexlify(xs)
    do_vexing(xs)

