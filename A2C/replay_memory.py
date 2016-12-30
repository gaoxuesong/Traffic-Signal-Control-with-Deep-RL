
# coding: utf-8

# In[1]:

from collections import deque
import random
import numpy as np
import pdb

# In[ ]:

class ReplayMemory(object):
    
    def __init__(self, random_seed=123):
        
        #self.buffer_size = buffer_size
        self.count = 0
        self.buf = deque()
        random_seed = random_seed
        
    def add(self, s, a, r, s2, t):
        
        # Current state, Action, Reward, Next State, terminal?
        exp = (s, a, r, s2, t)        
        self.buf.append(exp)
        self.count += 1
    def size(self):
        return self.count
    
    def sample_batch(self, batch_size):
        
        s_batch = np.array(self.buf)[:,0].tolist()
        a_batch = np.array(self.buf)[:,1].tolist()
        r_batch = np.array(self.buf)[:,2].tolist()
        s2_batch = np.array(self.buf)[:,3].tolist()
        t_batch = np.array(self.buf)[:,4].tolist()
        
        self.clear()
        return s_batch, a_batch, r_batch, s2_batch, t_batch
    
    def clear(self):
        
        self.buf.clear()
        self.count = 0

