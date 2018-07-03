
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy


# In[90]:
train=pd.read_csv('train_matrix.csv',index_col=0)
test=pd.read_csv('test_matrix.csv',index_col=0)

nb_users,nb_business=train.shape
train_set=np.array(train)
test_set=np.array(test)


# In[91]:


class LFM:
    
    def __init__(self,train_set,nb_users,nb_business,factor,batch_size,iter_num = 20,alpha = 0.1,Lambda = 0.1,epsilon = 0.01):
        '''
        initiate parameters
        '''
        self.train_set = train_set
        self.nb_users=nb_users
        self.nb_business=nb_business
        self.factor = factor
        self.batch_size=batch_size
        self.batchs=nb_users//batch_size
        self.alpha = alpha
        self.iter_num = iter_num
        self.Lambda = Lambda
        self.epsilon = epsilon
        
    
    def build(self,train):
        

        #initiate latent factor matrix
        self.decompose_p = tf.Variable(tf.random_uniform([self.nb_business,self.factor]))
        self.decompose_q = tf.Variable(tf.random_uniform([self.factor,self.nb_users]))    
        
        # Placeholders

        self.u_ind = tf.placeholder('int32', shape=[self.batch_size])
        #i = tf.placeholder('int32')
        self.y_rate = tf.placeholder('float32', shape=[self.batch_size, self.nb_business])
        self.y_mask = tf.placeholder('float32', shape=[self.batch_size, self.nb_business])
        #batch_q = tf.placeholder('float32', shape=[self.batch_size, self.factor])

        batch_q=tf.gather(tf.transpose(self.decompose_q),self.u_ind)
        self.pred_y_rate = tf.matmul(batch_q, tf.transpose(self.decompose_p))
        loss_m = tf.squared_difference(self.y_rate, self.pred_y_rate)
        self.loss = tf.reduce_sum(loss_m * self.y_mask)+self.Lambda*(tf.reduce_sum(tf.square(self.decompose_p))+tf.reduce_sum(tf.square(batch_q)))
        optimizer = tf.train.AdamOptimizer(self.alpha)
        train_op = optimizer.minimize(self.loss)
                                                      
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        

        
        tot_loss = 0.0
        self.delta_error=[]
        for epoch in range(self.iter_num):
            t = time.time()           
            for i in range(self.batchs):
                u_ind=list(range(i*self.batch_size,(i+1)*self.batch_size))
                U=train_set[i*self.batch_size:(i+1)*self.batch_size]
                mask=deepcopy(U)
                mask[mask!=0]=1
                feed={self.y_rate:U,self.y_mask:mask,self.u_ind:u_ind}
                _, loss = self.sess.run([train_op, self.loss], feed_dict = feed)
                tot_loss += loss
                #print (i)
                avg_loss = tot_loss / (epoch*self.batchs+i+1)
                if i%10==0:
                    print ("Epoch %d\tLoss\t%.2f\tTime %dmin"                     % (epoch, avg_loss, (time.time()-t)))
                    self.delta_error.append(avg_loss)

        print ("Recommender is built!")
        

        


# In[92]:


import time
tf.set_random_seed(123)
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

lfm=LFM(train_set,nb_users,nb_business,factor=20,batch_size=500,iter_num = 500,alpha = 0.01)

lfm.build(train)
p=lfm.sess.run(tf.Print(lfm.decompose_p,[lfm.decompose_p]))
q=lfm.sess.run(tf.Print(lfm.decompose_q,[lfm.decompose_q]))
print (p)
print (q)
pd.DataFrame(p).to_csv('p.csv')
pd.DataFrame(q).to_csv('q.csv')
ex = range(len(lfm.delta_error))
plt.figure(1)
plt.plot(ex,lfm.delta_error)
plt.show()

pred=np.dot(p,q).T
pd.DataFrame(pred).to_csv('lfmPred.csv')
mask=deepcopy(test_set)
mask[mask!=0]=1

loss=(abs(test_set-pred)*mask).sum()/mask.sum()
print(loss)


