#!/usr/bin/env python
# coding: utf-8

# In[14]:


import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[19]:


# height weight size
X = [[123,454,234],[75,66,556],[34,56,343],[34,53,644,],[78,75,574],[89,45,453],[45,44,454],[56,78,987],[98,76,890],[76,43,123]]
y = ['male','female','female','male','female','female','male','male', 'male','male']

clf = tree.DecisionTreeClassifier()
clf.fit(X , y)


# In[23]:


new = clf.predict([[23,56,765],[98,76,800]])


# In[24]:


print(new)


# In[ ]:




