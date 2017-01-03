
# coding: utf-8
import numpy as np
import pdb

# In[ ]:

class test_env(object):
    
    def __init__(self):
        self.s = 10 * np.random.choice(2,1)
    def reset(self):
        self.s = 10 * np.random.choice(2,1)
        return np.array(self.s)
    def step(self, A):
        #pdb.set_trace()
        t = 0
        a = A
        if a == 1:
            self.s = self.s + 1
        else:
            self.s -= 1
        r = -1
        if self.s < 0:
            r = -20
            self.s = 0
            t = 1
        elif self.s > 10:   # 3
            r = -20        # -20
            self.s = 10     # 3
            t = 1
        elif self.s == 5:
            r = 100
            t = 1
        elif self.s == 5:
            r = 100
        report = 0
            
        return np.array(self.s), np.array(r), np.array(t), report

