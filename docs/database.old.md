Create Indexes on 

'''
> db.symbols_debian.getIndexes()
[
        {
                "v" : 1,
                "key" : {
                        "_id" : 1
                },
                "name" : "_id_",
                "ns" : "desyl.symbols_debian"
        },
        {
                "v" : 1,
                "key" : {
                        "linkage" : 1,
                        "bin_name" : 1,
                        "compiler" : 1,
                        "arch" : 1,
                        "optimisation" : 1,
                        "path" : 1,
                        "name" : 1
                },
                "name" : "linkage_1_bin_name_1_compiler_1_arch_1_optimisation_1_path_1_name_1",
                "ns" : "desyl.symbols_debian"
        },
        {
                "v" : 1,
                "key" : {
                        "path" : 1
                },
                "name" : "path_1",
                "ns" : "desyl.symbols_debian"
        }
]
'''


```mongod --auth --wiredTigerCacheSizeGB 120.0 --nojournal --bind_ip_all```
