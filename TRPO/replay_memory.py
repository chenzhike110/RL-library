import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))

class Memory(object):
    def __init__(self) -> None:
        super().__init__()
        self.memory = []
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self):
        return Transition(*zip(*self.memory))
    
    def __len__(self):
        return len(self.memory)