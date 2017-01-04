
# coding: utf-8

# In[1]:

from collections import deque
import random
import numpy as np
import pdb
import time
try:
    import cPickle
except:
    import _pickle as cPickle

# In[ ]:

class ReplayMemory(object):
    
    def __init__(self, buffer_size, t_max, Gamma, random_seed=123):
        
        self.count = 0
        self.buf = deque()
        self.buffer_size = buffer_size
        self.t_max = t_max
        self.Gamma = Gamma
        random_seed = random_seed
        
    def add(self, s, a, r, s2, t):
        
        # Current state, Action, Reward, Next State, terminal?
        exp = (s, a, r, s2, t)
        if self.count >= self.buffer_size:
            self.buf.popleft()
            #self.buf.pop(0)
            self.count -= 1
        
        self.buf.append(exp)
        self.count += 1
        
    def size(self):
        return self.count
    
    def sample_batch(self, batch_size):
        #pdb.set_trace()
        if self.count >= (batch_size + self.t_max):
            
            sample_starts = np.random.choice(self.count - self.t_max, batch_size)
            sample_ends = sample_starts + np.array(self.t_max)
            t_batch = []
            r_batch = []
            s_batch = []
            a_batch = []
            samples = []
            start = time.time()
            for i in range(batch_size):
                samples += range(sample_starts[i], sample_ends[i])
            part1 = time.time() -start
            start = time.time()
            xx = np.array(self.buf)[samples,:]
            #pdb.set_trace()
            part2 = time.time() -start
            t_batch = xx[:,4].reshape(batch_size, self.t_max).tolist()
            r_batch = xx[:,2].reshape(batch_size, self.t_max).tolist()
            s_batch = xx[:,0].tolist()
            a_batch = xx[:,1].tolist()
            
            '''
            t_batch = np.array(self.buf)[samples,4].reshape(batch_size, self.t_max).tolist()
            r_batch = np.array(self.buf)[samples,2].reshape(batch_size, self.t_max).tolist()
            s_batch = np.array(self.buf)[samples,0].tolist()
            a_batch = np.array(self.buf)[samples,1].tolist()
            part2 = time.time() -start
            
            for i in range(batch_size):
                t_batch.append(np.array(self.buf)[sample_starts[i]:sample_ends[i],4].tolist())
                r_batch.append(np.array(self.buf)[sample_starts[i]:sample_ends[i],2].tolist())
                s_batch += np.array(self.buf)[sample_starts[i]:sample_ends[i],0].tolist()
                a_batch += np.array(self.buf)[sample_starts[i]:sample_ends[i],1].tolist()
            '''
            steps = 1 - np.array(t_batch)
            for i in range(1, self.t_max):
                steps[:,i] = steps[:,i] * steps[:,i-1]
            self.steps = steps
            self.batch_size = batch_size
            self.t_batch = t_batch
            self.r_batch = r_batch
            self.s_batch = s_batch
            self.a_batch = a_batch
            s2_batch = np.array(self.buf)[sample_ends-1,3].tolist()
            
            return s2_batch, part1, part2
    
    def nstep_calculation(self, last_V):
        
        r_batch = np.array(self.r_batch)
        a_batch = np.array(self.a_batch)
        R_batch = np.zeros([self.batch_size, self.t_max])
        s_batch = np.array(self.s_batch)
        R_batch[:,-1] = r_batch[:,-1] * self.steps[:,-2] + self.Gamma * last_V * self.steps[:,-1]
        for st in reversed(range(2, self.t_max)):
            R_batch[:,st-1] = r_batch[:,st-1] * self.steps[:,st-2]+ self.Gamma * R_batch[:,st] * self.steps[:,st-1]
        R_batch[:,0] = r_batch[:,0] + self.Gamma * R_batch[:,1] * self.steps[:,0]
        R_batch = R_batch.reshape(-1,1)
        mask = np.all(np.equal(R_batch, 0), axis=1)
        
        R_batch = R_batch[~mask].tolist()
        s_batch = s_batch[~mask].tolist()
        a_batch = a_batch[~mask].tolist()
        
        return s_batch, a_batch, R_batch
    
    def clear(self):
        
        self.buf.clear()
        self.count = 0
    
    def save_buf(self, filename):
        with open(filename+'RM_buffer', 'wb') as f:  # Python 3: open(..., 'wb')
            cPickle.dump([self.count, self.buf], f)
    
    def load_buf(self, filename):
        with open(filename+'RM_buffer', 'rb') as f:  # Python 3: open(..., 'wb')
            [self.count, self.buf] = cPickle.load(f)
