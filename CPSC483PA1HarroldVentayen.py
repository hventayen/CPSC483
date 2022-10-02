#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("/home/harrold/Desktop/HappinessData-1.csv")


# In[3]:


print(df)


# In[4]:


first = df.pop('Unhappy/Happy')


# In[5]:


df.insert(6, 'Unhappy/Happy', first)


# In[6]:


print(df)


# In[11]:


df.loc[[5]]


# In[12]:


df = df.fillna(0)


# In[13]:


df.loc[[5]]


# In[17]:


import sklearn


# In[20]:


import sklearn.model_selection


# In[21]:


from sklearn.model_selection import train_test_split


# In[72]:


X = np.array(df.iloc[:, 0:5])


# In[74]:


y = np.array(df['Unhappy/Happy'])


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)


# In[76]:


from sklearn.neighbors import KNeighborsClassifier


# In[77]:


from sklearn.metrics import accuracy_score


# In[78]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[79]:


knn.fit(X_train, y_train)


# In[80]:


pred = knn.predict(X_test)


# In[81]:


print("accuracy: {}".format(accuracy_score(y_test, pred)))


# In[ ]:




