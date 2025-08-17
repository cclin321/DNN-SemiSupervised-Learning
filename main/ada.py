# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:49:05 2022

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:34:31 2022

@author: admin
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR





#raw_data = pd.read_excel('raw data.xlsx')
#raw_data = raw_data.drop(['times'], axis=1)

train_dataset = pd.read_excel('20230406_train set.xlsx')

# In[25]:



#data=[]
#for i in range(len(raw_data)):
    #if raw_data.iloc[i,25] == 0:
        #pass
    #else:
        #data.append(raw_data.iloc[i])

#data = pd.DataFrame(data)




#train_dataset = data.sample(frac=0.8,random_state=0)



# In[28]:



# train_x = train_dataset.sample(frac=0.6,random_state=0)

train_x = train_dataset.copy()
train_y= train_x.pop('Grid  Power')
train_y=pd.DataFrame(train_y)
# unlabeled_train_x = train_dataset.drop(labeled_train_data.index)


# In[29]:









train_stats = train_dataset.describe()
train_stats.pop("Grid  Power")
train_stats = train_stats.transpose() 



def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_x = norm(train_x)





# adb = AdaBoostRegressor(base_estimator=SVR())

adb = AdaBoostRegressor(base_estimator=SVR(),n_estimators=10, learning_rate=0.8)

adb.fit(normed_train_x, train_y)


========================================================================

# In[36]:


test_x = pd.read_excel('20230406 test x.xlsx')





# In[37]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_test_x = norm(test_x)


# In[38]:


predictions = adb.predict(normed_test_x)
predictions = pd.DataFrame(predictions)




predictions.to_excel('20230407_ada predictions.xlsx')



test_y = pd.read_excel('20230406 test y(actual label).xlsx')

# In[20]:

from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_y,predictions)

from sklearn.metrics import mean_squared_error
mean_squared_error(test_y,predictions)
























