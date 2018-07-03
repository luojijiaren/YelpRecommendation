# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:03:43 2018

@author: fzhan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

data=pd.read_csv('CF_data.csv',index_col=0)
data0=pd.read_csv('yelp_business.csv')
data0=data0.iloc[:,[0,12]]
data1=data0[list(map(lambda x: 'Restaurants' in x,data0['categories']))]
data=data.merge(data1,on='business_id').iloc[:,:3]


def present(idn):
    d={}
    for i,idn in enumerate(idn):
        d[idn]=i
    return d
ud=present(data['user_id'])
data['user_id']=list(map(lambda x: ud[x],data['user_id']))
bd=present(data['business_id'])
data['business_id']=list(map(lambda x: bd[x],data['business_id']))

random.seed(86)
business=random.sample(list(set(data['business_id'])),5000)
data2=data[data['business_id'].isin(business)]
#exploratory analysis
user_num=len(set(data2['user_id']))
business_num=len(set(data2['business_id']))

business_popu=data2.iloc[:,:2].groupby('business_id').count()
business_popu.columns=['review_count']

plt.hist(business_popu.iloc[:,0].tolist(),range=(0,400))
plt.title('restaurant reveiw number distribution')
user_popu=data2.iloc[:,:2].groupby('user_id').count()
user_popu.columns=['review_count']
plt.hist(user_popu.iloc[:,0].tolist(),range=(0,40))
plt.title('user reveiw number distribution')

#filter out users with less than three reviews 
data3=data2.set_index('user_id')
data4=data3[user_popu['review_count']>2] 
#split 
train,validate=train_test_split(data4, test_size=0.33,random_state=42,stratify=data4.index)

train.sort_index().to_csv('train.csv')
validate.sort_index().to_csv('validate.csv')

train=pd.read_csv('train.csv')
validate=pd.read_csv('validate.csv')

users=sorted(list(set(train['user_id']).union(set(validate['user_id']))))
nb_users=len(users)
business=sorted(list(set(train['business_id']).union(set(validate['business_id']))))
nb_business=len(business)
dic=present(business)

# Converting the data into an array with users in lines and movies in columns
    
def convert(data):
    new_data = []
    for id_users in enumerate(users):
        id_business = data['business_id'][data['user_id'] == id_users]
        ind=list(map(lambda x:dic[x],id_business))
        id_ratings = data['stars'][data['user_id'] == id_users]
        ratings = np.zeros(nb_business)
        ratings[ind] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(train)
test_set = convert(validate)

pd.DataFrame(training_set).to_csv('train_matrix.csv')
pd.DataFrame(test_set).to_csv('test_matrix.csv')




    
    

