import context
from classes.config import Config
import numpy as np
import scipy as sp

class BPMessage():
    """
        Belief Propagation Messages are dirceted messages. Messages are vectors across all discrete instances.
        Messages can become invalid if messages incoming have changed
    """
    def __init__(self, dim, start, end, constraints):
        self._dim = dim

        ##sparsity matrix element wise multiply hack##
        #d = sp.sparse.lil_matrix((dim, dim))
        #d.setdiag(constraints)
        #####
        #self._value = d * sp.sparse.csr_matrix( (dim, 1), dtype=sp.float64 ) 
        #self._value = np.ones( (dim, ), dtype=np.float64 ) * constraints
        self._value = np.ones( (dim, ), dtype=np.float64 )
        self._start = start
        self._end = end
        self._valid = True

    def __hash__(self):
        """
            hashed by (start, end). No 2 (start, end) in the same set
        """
        return hash(self.direction)

    def __cmp__(self, other):
        return self.value == other.value

    def __str__(self):
        return "[{:^7}][{} -> {}][{}]".format("valid" if self.valid else "invalid", self.start, self.end, self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def direction(self):
        #return "{}-{}".format(self._start, self._end)
        return (self._start, self._end)
    
    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def valid(self):
        return self._valid

    @valid.setter
    def valid(self, isvalid):
        self._valid = isvalid

