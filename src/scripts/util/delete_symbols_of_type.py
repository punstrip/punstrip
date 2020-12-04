#!/usr/bin/python3
import sys

sys.path.insert(0, "../../classes/")
from database import Database

db = Database()
#db.client.symbols.remove({ 'optimisation': 'g', 'compiler': 'clang' })
db.client.symbols.remove({ 'type': 'inferred-r2' })
