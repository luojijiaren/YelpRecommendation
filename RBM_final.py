
# coding: utf-8

# In[19]:


# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt

# In[20]:


# Importing the dataset
training_set0=pd.read_csv('train_matrix.csv',index_col=0)
test_set0=pd.read_csv('test_matrix.csv',index_col=0)
nb_users,nb_business=training_set0.shape


# In[14]:


training_set1=np.array(training_set0)
test_set1=np.array(test_set0)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set1)
test_set = torch.FloatTensor(test_set1)


# In[15]:




# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set ==1] = 0
test_set[test_set ==2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.Tensor(nh, nv).uniform_(-0.1, 0.1)
        #torch.randn(nh, nv)
        self.a = 0.1*torch.ones(1, nh)
        self.b = 0.1*torch.ones(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk,learning_rate):
        self.W += learning_rate*(torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk))
        self.b += learning_rate*(torch.sum((v0 - vk), 0))
        self.a += learning_rate*(torch.sum((ph0 - phk), 0))


# In[16]:
torch.manual_seed(8)

nv = len(training_set[0])
nh = 20
batch_size = 100
learning_rate=0.05
rbm = RBM(nv, nh)
# Training the RBM
nb_epoch = 100
delta_error=[]
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]   #no data stay the same
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk,learning_rate)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    delta_error.append(train_loss/s)

ex = range(len(delta_error))
plt.figure(1)
plt.plot(ex,delta_error)
plt.xlabel('epoch')
plt.ylabel('mean training loss')
plt.title('RBM')
plt.show()
# In[18]:


# Testing the RBM
test_loss = 0
s = 0.
pred=np.zeros(training_set.shape)
weight=1.25*np.ones(nb_users)
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    for k in range(10):
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
    pred[id_user:id_user+1]=v.numpy()
    if len(vt[vt>=0]) > 0:#if there is data about test sample
        user_loss = torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        test_loss += user_loss
        s += 1.
        weight[id_user]=np.exp(user_loss)

weight=weight/weight.sum()
        
    #if id_user==0:
    #    res=v.numpy()
    #else:
    #    res=np.vstack((res,v.numpy()))
print('test loss: '+str(test_loss/s))
pd.DataFrame(pred).to_csv('rbmPred.csv')
print(pred)

