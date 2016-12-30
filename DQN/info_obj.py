import numpy as np
# coding: utf-8
class info_holder:
    def __init__(self):
        self.hid_layers = np.array([32]) #defualt
        self.learning_rate = 1e-3
        self.TAU = 0.01
        self.entropy_beta = 0.01
        

