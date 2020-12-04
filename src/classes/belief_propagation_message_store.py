import re
import context
from classes.config import Config
from classes.belief_propagation_message import BPMessage

class MessageStore(dict):
    """
    A dictionary to store message in loopy belief propagation
    Messages are automatically invalidated when one of its dependents is updated
    """
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    @staticmethod
    def key_to_start_and_end(key):
        """
                Parse message store key into start and end nodes
        """
        #if not isinstance(key, str):
        #    raise TypeError("MessageStore key is not a string")
        #if not isinstance(key, tuple):
        #    raise TypeError("MessageStore key is not a tuple")

        return key[0], key[1]
        #old slow implementation
        """
        m = re.match(r"(\d+)-(\d+)", key)
        if not m:
            raise ValueError("MessageStore key not in format {\d+}_{\d+}")
        start   = int(m.group(1))
        end     = int(m.group(2))
        return start, end
        """

    def add(self, msg):
        """
        Add message
        """
        self.__setitem__(msg.direction, msg)

    def __setitem__(self, key, value):
        """
                Invalidate dependents on update
        """
        start, end = MessageStore.key_to_start_and_end(key)

        if not isinstance(value, BPMessage):
            raise Exception("MessageStore can only store BPMessages")

        #invalidate all messages that are emitted from end
        for k in list(self.keys()):
            kstart, kend = MessageStore.key_to_start_and_end(k)
            if kstart == end:
                self[k].valid = False

        return dict.__setitem__(self, key, value)

    @staticmethod
    def direction_to_key(start, end):
        # returns the key for a given start and end direction
        #return "{}-{}".format(start, end)
        return (start, end)

    def is_valid(self, key):
        """
        Check is message direction is present and valid
        """
        if key in self:
            return self[key].valid
        return False

    def valid_msg_iter(self, isvalid):
        """
        Get all messages that are isvalid
        """
        if not isinstance(isvalid, bool):
            raise TypeError("isvalue needs to be True or False. Returns an iteratior to all messages that are/not valid")

        for v in self.values():
            if v.valid == isvalid:
                yield v

    def to_node_iter(self, node):
        """
            Iteration to all message to go TO an node
        """
        for direction in list(self.keys()):
            start, end = MessageStore.key_to_start_and_end(direction)

            if node == end:
                yield self[direction]

    def from_node_iter(self, node):
        """
            Iteration to all message to go FROM an node
        """
        for direction in list(self.keys()):
            start, end = MessageStore.key_to_start_and_end(direction)

            if node == start:
                yield self[direction]
