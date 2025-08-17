#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR


# In[2]:



#raw_data = pd.read_excel('raw data.xlsx')
#raw_data = raw_data.drop(['times'], axis=1)
train_dataset = pd.read_excel('20230406_train set.xlsx')


# In[3]:



#data=[]
#for i in range(len(raw_data)):
    #if raw_data.iloc[i,25] == 0:
        #pass
    #else:
        #data.append(raw_data.iloc[i])

#data = pd.DataFrame(data)



# In[4]:




#train_dataset = data.sample(frac=0.8,random_state=0)


train_x = train_dataset.copy()
train_y= train_x.pop('Grid  Power')
train_y=pd.DataFrame(train_y)



train_stats = train_dataset.describe()
train_stats.pop("Grid  Power")
train_stats = train_stats.transpose() 


# In[11]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_x = norm(train_x)





# In[14]:


# svr=SVR(C=1.0, epsilon=0.2)

svr=SVR(C=1.0, epsilon=0.1)

svr.fit(normed_train_x,train_y)


# In[16]:


# score = svr.score(normed_labeled_train_data,labeled_data_labels)
# score

# =====================================================================================

test_x = pd.read_excel('20230406 test x.xlsx')


# In[18]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_test_x = norm(test_x)


# In[19]:


predictions = svr.predict(normed_test_x)
predictions = pd.DataFrame(predictions)




# predictions.to_csv('20230311_60 svr predictions.csv')
predictions.to_excel('20230406_svr predictions.xlsx')



test_y = pd.read_excel('20230406 test y(actual label).xlsx')



from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_y,predictions)

from sklearn.metrics import mean_squared_error
mean_squared_error(test_y,predictions)






