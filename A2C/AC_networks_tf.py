
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf 
# import tflearn
import pdb


# In[ ]:

class ActorNet(object):
    
    def __init__(self, info):
        self.graph = info.graph
        self.dim_s = info.dim_s
        self.dim_a = info.dim_a
        self.hid_layers = info.act_hid_layers
        #self.learning_rate = info.learning_rate
        self.learning_rate = tf.placeholder(tf.float32)
        
        # Actor Network
        with tf.name_scope("actor"):
            self.inputs = tf.placeholder(shape=[None, self.dim_s], dtype=tf.float32, name="inputs")
            self.policy = self.create_net()
        # self.inputs, self.policy = self.create_net()

        # This gradient will be provided by the critic network
        self.td = tf.placeholder(tf.float32, [None, 1], name="td_for_actor")

        # taken action (input for policy)
        self.taken_action = tf.placeholder(tf.int32, [None], name="taken_action")
        self.one_hot_action = tf.one_hot(self.taken_action, self.dim_a, dtype=tf.float32)
        
        #######################################################
        ######## Probably has to be changed ###################
        ######## With log of pi (look at https://github.com/miyosuda/async_deep_reinforce/blob/master/game_ac_network.py
        #######################################################
        
        # Combine the gradients here 
        self.responsible_outputs = tf.reduce_sum(self.policy * self.one_hot_action, [1])
        self.log_pi = tf.log(self.responsible_outputs)
        self.loss_vector = self.log_pi * tf.reduce_sum(self.td,[1])
        self.actor_loss = - tf.reduce_sum(self.loss_vector)
        self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "actor")
        self.actor_gradients = tf.gradients(self.actor_loss, self.network_vars)
        
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_vars))
        
    def create_net(self):
        scale = 0.01
        w_1 = tf.Variable(tf.truncated_normal([self.dim_s, self.hid_layers[0]])) * scale
        b_1 = tf.Variable(tf.zeros([self.hid_layers[0]]))
        inputs_1 = tf.matmul(self.inputs, w_1) + b_1
        net = tf.nn.tanh(inputs_1)
        
        if(len(self.hid_layers) > 1):
            w_2 = tf.Variable(tf.truncated_normal([self.hid_layers[0], self.hid_layers[1]])) * scale
            b_2 = tf.Variable(tf.zeros([self.hid_layers[1]]))
            inputs_2 = tf.matmul(self.out_1, w_2) + b_2
            net = tf.nn.tanh(inputs_2)
        
        w_3 = tf.Variable(tf.truncated_normal([self.hid_layers[-1], self.dim_a])) * scale
        b_3 = tf.Variable(tf.zeros([self.dim_a]))
        logits = tf.matmul(net, w_3) + b_3
        policy = tf.nn.softmax(logits)

        return policy

    def set_up(self, sess):
        self.sess = sess
        
    def train(self, inputs, actions, td, lr_rate):
        
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.taken_action: actions,
            self.td: td,
            self.learning_rate: lr_rate
        })
        
    def give_policy(self, inputs):

        return self.sess.run(self.policy, feed_dict={
            self.inputs: inputs
        })
    
    def take_action(self, inputs):
        
        pi = self.give_policy(inputs)
        pi = pi/np.sum(pi, axis = 1).reshape(-1,1)
        action = np.zeros(pi.shape[0])
        
        for i in range(pi.shape[0]):
            action = np.random.choice(self.dim_a, 1, p=pi[i,:].reshape(self.dim_a))
        
        return action[0]
    
    def reset_net(self):
        tf.reset_default_graph()


# In[ ]:

class CriticNet(object):
    
    def __init__(self, info):
        self.graph = info.graph
        self.dim_s = info.dim_s
        self.dim_a = info.dim_a
        self.hid_layers = info.crit_hid_layers
        #self.learning_rate = info.learning_rate
        self.learning_rate = tf.placeholder(tf.float32)
        
        #tf.placeholder(tf.float32, [None, self.dim_a])
        
        # Actor Network
        with tf.name_scope("critic"):
            self.inputs = tf.placeholder(shape=[None, self.dim_s], dtype=tf.float32, name="inputs")
            self.value = self.create_net()

        # Network target (y_i) r+ gamma*V_target(s2)
        self.R = tf.placeholder(tf.float32, [None, 1], name="R_for_critic")

        # Define loss and optimization Op
        self.critic_loss = tf.reduce_mean(tf.square(self.R - self.value))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)
        
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
        
        w_3 = tf.Variable(tf.truncated_normal([self.hid_layers[-1], 1]))
        b_3 = tf.Variable(tf.zeros([1]))
        value = tf.matmul(net, w_3) + b_3

        return value
    
    def set_up(self, sess):
        self.sess = sess
        
    def train(self, inputs, R, lr_rate):
        #pdb.set_trace()
        return self.sess.run([self.value, self.critic_loss, self.optimize], feed_dict={
                self.inputs: inputs,
                self.R: R,
                self.learning_rate: lr_rate
        })
    
    def predict(self, inputs):
        return self.sess.run(self.value, feed_dict={
            self.inputs: inputs
        })

    def reset_net(self):
        tf.reset_default_graph()

