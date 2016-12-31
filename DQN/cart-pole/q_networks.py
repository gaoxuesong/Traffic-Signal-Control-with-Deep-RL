
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf 
#import tflearn
import pdb
import random


# In[10]:

class Q_Network(object):
    def __init__(self, info):
        self.graph = info.graph
        self.dim_s = info.dim_s
        self.dim_a = info.dim_a
        self.hid_layers = info.hid_layers
        #self.epsilon = info.epsilon
        
        self.learning_rate = tf.placeholder(tf.float32, name="lr")
        self.inputs = tf.placeholder(shape=[None, self.dim_s], dtype=tf.float32, name="inputs")
        with tf.name_scope("model"):
            self.q_values = self.create_net()
            
        with tf.name_scope("target"):
            self.target_q_values = self.create_net()
            
        self.greedy_action = tf.argmax(self.q_values, axis=1)
        self.target_q_max = tf.reduce_max(self.target_q_values, reduction_indices=[1])
        self.R = tf.placeholder(tf.float32, [None], name="R")
        self.taken_action = tf.placeholder(tf.int32, [None], name="taken_actions")
        self.one_hot_action = tf.one_hot(self.taken_action, self.dim_a, dtype=tf.float32)
        
        self.q_a = tf.reduce_sum(self.q_values * self.one_hot_action, [1]) 
        self.loss = tf.reduce_mean(tf.square(self.R - self.q_a))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        # Updating the target net
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model")
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')

        self.update_ops = [] # this list_op must be run to update
        for from_var,to_var in zip(from_vars,to_vars):
            self.update_ops.append(to_var.assign(from_var))
        
        
    def create_net(self):
        
        w_1 = tf.Variable(tf.truncated_normal([self.dim_s, self.hid_layers[0]]))
        b_1 = tf.Variable(tf.zeros([self.hid_layers[0]]))
        inputs_1 = tf.matmul(self.inputs, w_1) + b_1
        net = tf.nn.tanh(inputs_1)
        
        if(len(self.hid_layers) > 1):
            w_2 = tf.Variable(tf.truncated_normal([self.hid_layers[0], self.hid_layers[1]]))
            b_2 = tf.Variable(tf.zeros([self.hid_layers[1]]))
            inputs_2 = tf.matmul(self.out_1, w_2) + b_2
            net = tf.nn.tanh(inputs_2)
        
        w_3 = tf.Variable(tf.truncated_normal([self.hid_layers[-1], self.dim_a]))
        b_3 = tf.Variable(tf.zeros([self.dim_a]))
        q_value = tf.matmul(net, w_3) + b_3
        
        return q_value
    
    def set_up(self, sess):
        self.sess = sess
        
    def target_q(self, inputs):
        
        return self.sess.run([self.target_q_max], feed_dict={
            self.inputs: inputs,
        })
    
    def train(self, inputs, actions, R, lr_rate):
        #pdb.set_trace()
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.taken_action: actions,
            self.R: R,
            self.learning_rate: lr_rate
        })
        
    def take_action(self, inputs, epsilon):
        
        gr_action = self.sess.run(self.greedy_action, feed_dict={
            self.inputs: inputs
        })
        if (epsilon > random.random()):
            action = np.random.choice(self.dim_a)
        else:
            action = gr_action[0]
        
        return action
    
    def update_target(self):
        self.sess.run(self.update_ops)
        
    def q_a_values(self, inputs):
        
        return self.sess.run([self.q_values], feed_dict={
            self.inputs: inputs,
        })
        
    def reset_net(self):
        tf.reset_default_graph()
        

