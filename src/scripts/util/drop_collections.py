#!/usr/bin/python3
import sys
import os
import re

sys.path.insert(0, "../../classes/")
from database import Database

if __name__ == "__main__":

    #get all symbols
    db_ref = Database()
    db = db_ref.client

    map_reduce_re = r'(.*)_map_reduce'
    mrre = re.compile(map_reduce_re)

    temp_re = r'temp_(.*)'
    tmpre = re.compile(temp_re)

    print("[+] Droping MongoDB collections that match {} or {}".format(map_reduce_re, temp_re))

    colls = db.collection_names()
    for col in colls:
        if mrre.match(col) or tmpre.match(col):
            #print("[+] {} matches!".format(col))
            print("[+] Dropping {}".format(col))
            db[col].drop()
        #else:
        #    print("[-] {} does not match!".format(col))

