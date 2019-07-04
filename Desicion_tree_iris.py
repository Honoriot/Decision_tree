#!/usr/bin/env python
# coding: utf-8

# In[24]:


import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import train_test_split


# In[28]:


iris = load_iris()
print('Keys: {}'.format(iris.keys()))


# In[29]:


print('data:{}'.format(iris['data']))


# In[30]:


print('features_names:{}'.format(iris['feature_names']))


# In[31]:


print('target:{}'.format(iris['target']))
print('target_names:{}'.format(iris['target_names']))


# In[32]:


clf1 = tree.DecisionTreeClassifier()
clf1.fit(iris['data'] , iris['target'])


# In[33]:


ans = clf1.predict([[0.2, 2, 4.5, 1]])
print(ans)


# In[34]:


#X_train, y_train, X_test, y_test = train_test_split(iris['data'], iris['target'], random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)


# In[35]:


clf2 = tree.DecisionTreeClassifier()
clf2.fit(X_train, y_train)


# In[36]:


res = clf2.predict(X_test)
print(res)


# In[23]:


clf2.score(X_test, y_test)


# In[ ]:




