#!/usr/bin/python3
import sys

sys.path.insert(0, "../../classes/")
from database import Database

db = Database()
db.flush_symbols()
