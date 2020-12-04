#!/usr/bin/python3
import sys

sys.path.insert(0, "../../classes/")
from database import Database
from symbol import Symbol

db = Database()
db_ref = db.client

##try dropping previous unique symbols
try: 
    print("[+] Dropping desyl.unique_symbols...")
    db.drop_collection('unique_symbols')
except e:
    print("Exception droping unique_symbols. {}".format(e))


#add unqiue symbols back
print("[+] Fetching symbols with unique hashes")
res = db.client.symbols.aggregate([
    { 
        '$group' : { 
            '_id' : '$hash', 
            'ref_id': { '$first' :  '$_id'}
        } 
    }
], allowDiskUse=True)

print("[+] Adding unique symbols to desyl.unique_symbols...")
for s in res:
    sym = Symbol.fromDatabase(db_ref, 'symbols', s['ref_id'])
    sym.save_to_db(db_ref, 'unique_symbols')
