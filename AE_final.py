
# coding: utf-8

# In[2]:


# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt


# In[3]:


# Importing the dataset
training_set0=pd.read_csv('/my_data/train_matrix.csv',index_col=0)
test_set0=pd.read_csv('/my_data/test_matrix.csv',index_col=0)
nb_users,nb_business=training_set0.shape
training_set1=np.array(training_set0)
test_set1=np.array(test_set0)


# In[4]:


# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set1)
test_set = torch.FloatTensor(test_set1)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__() #inheritage
        self.fc1 = nn.Linear(nb_business, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 50)
        self.fc4 = nn.Linear(50, nb_business)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


# In[7]:


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 20
delta_loss=[]
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.   #float
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)  #add an additional batch dimension at o column
        target = input.clone()   #similar to copy
        if torch.sum(target.data > 0) > 0: #to save space, just include users have rating.
            output = sae(input)
            target.require_grad = False #target will not change, save calculation
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_business/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
        if s%500==0:
            print(train_loss/s)
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    delta_loss.append(train_loss/s)


# In[10]:


# Testing the SAE
test_loss = 0
s = 0.
pred=np.zeros(training_set.shape)
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)  #use the movies a user has watched(training set)
    target = Variable(test_set[id_user])
    output = sae(input)
    pred[id_user]=output.data.numpy()
    if torch.sum(target.data > 0) > 0:
        
        
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_business/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))


# In[ ]:


pd.DataFrame(pred).to_csv('aePred.csv')
print(pred)


# In[16]:


#delta_loss=[3.09,2.30,2.11,1.90,1.85,1.71,1.67,1.55,1.52,1.42,1.39,1.31,1.30,1.24,1.23,1.19,1.18,1.15,1.15,1.12]
#import matplotlib.pyplot as plt


# In[9]:



ex = range(len(delta_loss))
plt.figure(1)
plt.plot(ex,delta_loss)
plt.xlabel('epoch')
plt.ylabel('mean training loss')
plt.title('AutoEncoder')
plt.axis([0, 20, 0.9, 3])
plt.show()

