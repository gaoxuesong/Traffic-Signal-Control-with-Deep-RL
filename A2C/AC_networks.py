
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf 
import tflearn
import pdb


# In[ ]:

class ActorNet(object):
    
    def __init__(self, info):
        self.graph = info.graph
        self.dim_s = info.dim_s
        self.dim_a = info.dim_a
        self.hid_layers = info.act_hid_layers
        #self.learning_rate = info.learning_rate
        self.tau = info.tau
        self.learning_rate = tf.placeholder(tf.float32)
        
        #tf.reset_default_graph()
        # Actor Network
        self.inputs, self.policy = self.create_net()
        self.network_params = tf.trainable_variables()
        self.no_params = len(tf.trainable_variables())
        

        # This gradient will be provided by the critic network
        self.td = tf.placeholder(tf.float32, [None, 1])

        # taken action (input for policy)
        self.a = tf.placeholder(tf.float32, [None, self.dim_a])
        #######################################################
        ######## Probably has to be changed ###################
        ######## With log of pi (look at https://github.com/miyosuda/async_deep_reinforce/blob/master/game_ac_network.py
        #######################################################
        # Combine the gradients here 
        self.responsible_outputs = tf.reduce_sum(self.policy * self.a, [1])
        self.log_pi = tf.log(self.responsible_outputs)
        
        self.actor_loss = - tf.reduce_sum( self.log_pi * self.td)
        self.actor_gradients = tf.gradients(self.actor_loss, self.network_params)
           
        #self.reset_gradients = tf.assign(self.actor_gradients, tf.zeros_like(self.actor_gradients))
        #self.add_gradients = tf.add(self.actor_gradients, tf.gradients(self.actor_loss, self.network_params, self.td))
        
        #self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = self.no_params
        
    def create_net(self):
        
        inputs = tflearn.input_data(shape=[None, self.dim_s])
        net = tflearn.fully_connected(inputs, self.hid_layers[0], activation='tanh')
        
        if len(self.hid_layers) > 1:
            for h in self.hid_layers[1:]:
                net = tflearn.fully_connected(net, h, activation='tanh')
        
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        policy = tflearn.fully_connected(net, self.dim_a, activation='softmax', weights_init=w_init)
        #pdb.set_trace()
        return inputs, policy

    def set_up(self, sess):
        self.sess = sess
        
    def train(self, inputs, actions, td, lr_rate):
        
        #taken_action = self.take_action(inputs)
        #pdb.set_trace()
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.a: actions,
            self.td: td,
            self.learning_rate: lr_rate
            #self.action_gradient: a_gradient
        })
        
    def apply_grads(self, lr_rate):
        
        #taken_action = self.take_action(inputs)
        #pdb.set_trace()
        self.sess.run(self.reset_gradients)
        self.sess.run(self.optimize, feed_dict={
            #self.inputs: inputs,
            #self.a: actions,
            #self.td: td,
            self.learning_rate: lr_rate
            #self.action_gradient: a_gradient
        })
        
    def give_policy(self, inputs):
        
        return self.sess.run(self.policy, feed_dict={
            self.inputs: inputs
        })
    
    def take_action(self, inputs):
        
        pi = self.give_policy(inputs)
        pi = pi/np.sum(pi, axis = 1).reshape(-1,1)
        taken_action = np.zeros_like(pi)
        #pdb.set_trace()
        for i in xrange(pi.shape[0]):
            taken_action[i, np.random.choice(self.dim_a, 1, p=pi[i,:].reshape(self.dim_a))] = 1
        return taken_action
    
    def target_take_action(self):
        pi = self.sess.run(self.target_policy, feed_dict={
            self.inputs: inputs})
        taken_action = np.zeros(self.dim_a)
        return taken_action[np.random.choice(self.dim_a, 1, p=np.reshape(pi, self.dim_a))]
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        
    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def reset_net(self):
        tf.reset_default_graph()


# In[ ]:

class CriticNet(object):
    
    def __init__(self, info, num_actor_vars):
        self.graph = info.graph
        self.dim_s = info.dim_s
        self.dim_a = info.dim_a
        self.hid_layers = info.crit_hid_layers
        #self.learning_rate = info.learning_rate
        self.tau = info.tau
        self.learning_rate = tf.placeholder(tf.float32)
        
        #tf.placeholder(tf.float32, [None, self.dim_a])
        
        # Actor Network
        self.inputs, self.value  = self.create_net()

        self.no_params = len(tf.trainable_variables()) - num_actor_vars


        # This gradient will be provided by the critic network
        #self.td = tf.placeholder(tf.float32, [None])

        # taken action (input for policy)
        #self.a = tf.placeholder(tf.float32, [None, self.dim_a])

        # Network target (y_i) r+ gamma*V_target(s2)
        self.R = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.critic_loss = tflearn.mean_square(self.R, self.value)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)
        
    def create_net(self):
        
        inputs = tflearn.input_data(shape=[None, self.dim_s])
        # works with V instead of Q, uncomment action for Q
        # action = tflearn.input_data(shape=[None, self.dim_a])
        net = tflearn.fully_connected(inputs, self.hid_layers[0], activation='tanh')
        
        #net = tflearn.fully_connected(net, self.hid_layers[1])
        #t2 = tflearn.fully_connected(action, self.hid_layers[1])
        #net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        for h in self.hid_layers[1:]:
            net = tflearn.fully_connected(net, h, activation='tanh')
        
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        
        return inputs, out #, action
    
    def set_up(self, sess):
        #pdb.set_trace()
        self.sess = sess
        
    def train(self, inputs, R, lr_rate):
        #pdb.set_trace()
        return self.sess.run([self.value, self.optimize], feed_dict={
                self.inputs: inputs,
                self.R: R,
                self.learning_rate: lr_rate
            })
    
    def predict(self, inputs):
        return self.sess.run(self.value, feed_dict={
            self.inputs: inputs
        })
    
    def predict_target(self, inputs):

        return self.sess.run(self.target_value, feed_dict={
            self.target_inputs: inputs
        })
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        
    def reset_net(self):
        tf.reset_default_graph()

