
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf 
import tflearn
import pdb


# In[ ]:

class AC_Net(object):
    
    def __init__(self, info, thread):
        
        self.dim_s = info.dim_s
        self.dim_a = info.dim_a
        self.shared_layers = info.shared_layers
        self.act_hid_layers = info.act_hid_layers
        self.crit_hid_layers = info.crit_hid_layers
        self.thread = thread
        self.entropy_beta = info.entropy_beta
        self.w_shared = []
        self.w_actor = []
        self.w_critic =[]
        self.b_shared = []
        self.b_actor = []
        self.b_critic = []
        
        self.graph = info.graph
        with self.graph.as_default():
            self.learning_rate = tf.placeholder(tf.float32)
        
            # Actor Network
            self.inputs, self.policy, self.value = self.create_net()
            self.network_params = tf.trainable_variables()
            self.no_params = len(tf.trainable_variables())

            # This gradient will be provided by the critic network
            self.R = tf.placeholder(tf.float32, [None, 1])
            self.td = self.R - self.value

            # taken action (input for policy)
            self.a = tf.placeholder(tf.float32, [None, self.dim_a])

            # Log of Policy
            self.log_pi = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))
            
            # policy entropy
            entropy = -tf.reduce_sum(self.policy * self.log_pi, reduction_indices=1)

            # gradients of actor
            self.actor_gradients = tf.gradients(self.log_pi, self.network_params, self.td + self.entropy_beta * entropy)

            # squared td
            self.sq_td = tf.nn.l2_loss(self.td)
            # gradients of critic
            self.actor_gradients = tf.gradients(self.sq_td, self.network_params)
            
            # EXTRA
            self.actor_loss = - tf.reduce_sum( tf.reduce_sum( tf.mul( self.log_pi, self.a ), reduction_indices=1 ) * self.td + entropy * self.entropy_beta)
            self.critic_loss = self.sq_td
            self.loss = self.actor_loss + self.critic_loss
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.num_trainable_vars = self.no_params
        
    def create_net(self):
        
        inputs = tflearn.input_data(shape=[None, self.dim_s])
        
        # if actor and critic share some part of their networks
        if len(self.shared_layers) > 0:
            shared_net = tflearn.fully_connected(inputs, self.shared_layers[0], activation='relu')
            self.w_shared.append(shared_net.W)
            self.b_shared.append(shared_net.b)
            # keep adding layer till the shared network is completely created
            if len(self.shared_layers) > 1:
                for h in self.shared_layers[1:]:
                    shared_net = tflearn.fully_connected(net, h, activation='relu')
                    self.w_shared.append(shared_net.W)
                    self.b_shared.append(shared_net.b)
            # create a beginning of actor and critic continuing the shared net
            actor_net = tflearn.fully_connected(shared_net, self.act_hid_layers[0], activation='relu')
            self.w_actor.append(actor_net.W)
            self.b_actor.append(actor_net.b)
            critic_net = tflearn.fully_connected(shared_net, self.crit_hid_layers[0], activation='relu')
            self.w_critic.append(critic_net.W)
            self.b_critic.append(critic_net.b)
            
        
        # if actor and critic don't share layers
        else:
            actor_net = tflearn.fully_connected(inputs, self.act_hid_layers[0], activation='relu')
            self.w_actor.append(actor_net.W)
            self.b_actor.append(actor_net.b)
            critic_net = tflearn.fully_connected(inputs, self.crit_hid_layers[0], activation='relu')
            self.w_critic.append(critic_net.W)
            self.b_critic.append(critic_net.b)
            
        # rest of actor
        if len(self.act_hid_layers) > 1:
            for h in self.act_hid_layers[1:]:
                actor_net = tflearn.fully_connected(actor_net, h, activation='relu')
                self.w_actor.append(actor_net.W)
                self.b_actor.append(actor_net.b)
        
        # rest of critic
        if len(self.crit_hid_layers) > 1:
            for h in self.crit_hid_layers[1:]:
                critic_net = tflearn.fully_connected(critic_net, h, activation='relu')
                self.w_critic.append(critic_net.W)
                self.b_critic.append(critic_net.b)
                
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.1, maxval=0.1)
        # policy
        policy = tflearn.fully_connected(actor_net, self.dim_a, activation='softmax', weights_init=w_init)
        self.w_actor.append(policy.W)
        self.b_actor.append(policy.b)
        value = tflearn.fully_connected(critic_net, 1, weights_init=w_init)
        self.w_critic.append(value.W)
        self.b_critic.append(value.b)
        return inputs, policy, value

    def set_up(self, sess):
        self.sess = sess
        
    def sync_from_source(self, src_net):
        '''src_vars = src_net.network_params
        dst_vars = self.network_params
        all_vars = tf.trainable_variables()
        floor = len(src_vars)
        for i in xrange(len(src_vars)):
            tf.assign(tf.trainable_variables()[self.thread * floor + i], src_vars[i])
            
        pdb.set_trace()    
        self.w_shared = [src_net.w_shared[i] for i in xrange(len(src_net.w_shared))]
        self.w_actor = [src_net.w_actor[i] for i in xrange(len(src_net.w_actor))]
        self.w_critic = [src_net.w_critic[i] for i in xrange(len(src_net.w_critic))]
        self.b_shared = [src_net.b_shared[i] for i in xrange(len(src_net.b_shared))]
        self.b_actor = [src_net.b_actor[i] for i in xrange(len(src_net.b_actor))]
        self.b_critic = [src_net.b_critic[i] for i in xrange(len(src_net.b_critic))]
        '''
        pdb.set_trace()
        self.update_op = self.w_shared[0].assign(src_net.w_shared[0])
        self.update_op.eval()
        #tf.trainable_variables() = dst_vars
        #dst_vars = [src_vars[i] for i in xrange(len(src_vars))]
        
    def train(self, inputs, actions, R, lr_rate):
        
        # taken_action = self.take_action(inputs)
        #pdb.set_trace()
        return self.sess.run([self.value, self.optimize], feed_dict={
            self.inputs: inputs,
            self.a: actions,
            self.R: R,
            self.learning_rate: lr_rate
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
        pi = pi/np.sum(pi, axis = 1).reshape(-1,1)
        taken_action = np.zeros(self.dim_a)
        for i in xrange(pi.shape[0]):
            taken_action[i, np.random.choice(self.dim_a, 1, p=pi[i,:].reshape(self.dim_a))] = 1
        return taken_action
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        
    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def reset_net(self):
        tf.reset_default_graph()
      
    def predict(self, inputs):
        return self.sess.run(self.value, feed_dict={
            self.inputs: inputs
        })
        


# In[ ]:
'''
class CriticNet(object):
    
    def __init__(self, info, num_actor_vars):
        self.graph = info.graph
        self.dim_s = info.dim_s
        self.dim_a = info.dim_a
        self.hid_layers = info.crit_hid_layers
        #self.learning_rate = info.learning_rate
        self.tau = info.tau
        self.learning_rate = tf.placeholder(tf.float32)
        
        # Actor Network
        self.inputs, self.value  = self.create_net()

        self.no_params = len(tf.trainable_variables()) - num_actor_vars

        # Target Network
        self.target_inputs, self.target_value = self.create_net()
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]
        self.target_no_params = len(tf.trainable_variables()) - self.no_params - num_actor_vars

        #pdb.set_trace()
        # Op for periodically updating target network with online network weights
        #self.update_target_network_params =\
        #    [self.network_params[self.no_params + i].assign(tf.mul(self.network_params[i], self.tau) +\
        #                                                    tf.mul(self.network_params[self.no_params + i], 1. - self.tau))\
        #        for i in range(self.target_no_params)]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
        
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
        net = tflearn.fully_connected(inputs, self.hid_layers[0], activation='relu')
        
        #net = tflearn.fully_connected(net, self.hid_layers[1])
        #t2 = tflearn.fully_connected(action, self.hid_layers[1])
        #net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        for h in self.hid_layers[1:]:
            net = tflearn.fully_connected(net, h, activation='relu')
        
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        
        return inputs, out #, action
    
    def set_up(self, sess):
        self.sess = sess
        
    def train(self, inputs, R, lr_rate):
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
'''
