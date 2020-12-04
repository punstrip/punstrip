import pickle
import json
import networkx as nx
import sys

print "Loading {}".format(sys.argv[1])
nx.draw( json.loads( pickle.load( open( sys.argv[1], 'rb') ) ) )

