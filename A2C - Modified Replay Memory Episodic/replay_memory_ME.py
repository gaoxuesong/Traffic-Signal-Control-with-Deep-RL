
# coding: utf-8

# In[1]:

from collections import deque
import random
import numpy as np
import pdb
try:
    import cPickle
except:
    import _pickle as cPickle
# In[ ]:

class ReplayMemory(object):
    
    def __init__(self, buffer_size, random_seed=123):
        
        self.buffer_size = buffer_size
        self.count = 0
        self.last_episdode = -1
        self.buf = deque()
        self.buf_ep = deque()
        random_seed = random_seed
        
    def add(self, s, a, r, s2, t):
        
        # Current state, Action, Reward, Next State, terminal?
        exp = (s, a, r, s2, t)        
        self.buf.append(exp)
        
    def episode_end(self):
        
        if self.count > self.buffer_size:
            self.buf_ep.popleft()
            self.count -= 1
        self.buf_ep.append(list(self.buf))
        self.buf.clear()
        self.count += 1
        
    def size(self):
        return self.count
    
    def sample_batch(self, batch_size):
        
        if self.count > batch_size:
            samples = random.sample(self.buf_ep, batch_size)
        else:
            samples = random.sample(self.buf_ep, self.count)
        C = [item for sublist in samples for item in sublist]
        #if len(samples) == 1:
        #    C = samples
        
        s_batch = np.array(C)[:,0].tolist()
        a_batch = np.array(C)[:,1].tolist()
        r_batch = np.array(C)[:,2].tolist()
        s2_batch = np.array(C)[:,3].tolist()
        t_batch = np.array(C)[:,4].tolist()
        
        return s_batch, a_batch, r_batch, s2_batch, t_batch
    
    def clear(self):
        
        self.buf_ep.clear()
        self.buf.clear()
        self.count = 0
        self.last_episdode = -1
        
    def save_buf(self, filename):
        with open(filename+'RM_buffer', 'wb') as f:  # Python 3: open(..., 'wb')
            cPickle.dump([self.count, self.buf_ep], f)
    
    def load_buf(self, filename):
        with open(filename+'RM_buffer', 'rb') as f:  # Python 3: open(..., 'wb')
            [self.count, self.buf_ep] = cPickle.load(f)

