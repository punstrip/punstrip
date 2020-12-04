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
    COL      = "symbols_debian"
    #get all symbols
    db = Database()

    res = db.client[COL].removeMany({ "name" : ""})
    #res = db.client[COL].update({}, { "$pull" : { "callers" : "" })


    #db.symbols_debian.updateMany({}, { $pull : { "callers" :""} } )
    #db.symbols_debian.updateMany({}, { $pull : { "callees" :""} } )
