#!/usr/bin/python3
import sys
import math
import os

from multiprocessing.pool import ThreadPool
import functools
import itertools

import context
from classes.database import Database

if __name__ == "__main__":

    FROM_ARCH    = None
    TO_ARCH     = "x86_64"
    COL      = "symbols"
    #get all symbols
    db = Database()

    res = db.client[COL].find({ "arch" : FROM_ARCH})
    for r in res:
        db.client[COL].update_one( {'_id' : r['_id']} , { '$set' : { 'arch' : TO_ARCH } } )
