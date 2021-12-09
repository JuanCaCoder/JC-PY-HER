#!/usr/bin/env python
# coding: utf-8

# # Proyecto Integrador

# In[90]:


from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tkinter
matplotlib.use('TkAgg')
from sklearn.linear_model import LinearRegression


# # Linear regression

# ## Hallazgo #1

# In[91]:


df = pd.read_csv("datosc.csv")
df


# In[92]:


x= np.array(df['TEMPERATURA'])
y= np.array(df['ALTITUD'])


# In[93]:


plt.scatter(x,y)
plt.show()


# In[94]:


model= LinearRegression()

#learning step
model.fit(x.reshape(-1,1),y)


# In[95]:


model.coef_


# In[96]:


model.intercept_


# In[97]:


#make prediction
test = np.array([22]).reshape(-1,1)
prediction = model.predict(test)


# In[98]:


prediction


# Esto quiere decir que a la altura de 1596.0458m sobre el nivel del mar la temperatura será de 22 grados, de acuerdo a este análisis de datos

# In[99]:


plt.scatter(x,y)
test=np.arange(24,25.5,0.2).reshape(-1,1)
plt.plot(test,model.predict(test), c="r")

#testing with one sample 
heightp = np.array([24]).reshape(-1,1)
tempp = model.predict(heightp)
plt.scatter(heightp,tempp,c="y")

plt.xlabel("Temperatura (C)")
plt.ylabel("Altura (M)")
plt.show()


# Aqui se puede notar la relación que existe entre temperatura y altura en la que a menos altura hay más temperatura, el punto amarillo es la predicción de altura para 24 grados

# ## Hallazgo #2

# In[108]:


df.columns 


# In[109]:


x= np.array(df['CALIDAD'])
y= np.array(df['ALTITUD'])


# In[110]:


plt.scatter(x,y)
plt.show()


# In[111]:


model2= LinearRegression()

#learning step
model2.fit(x.reshape(-1,1),y)


# In[112]:


model2.coef_


# In[113]:


model2.intercept_


# In[114]:


#make prediction
test = np.array([160]).reshape(-1,1)
prediction = model2.predict(test)


# In[115]:


prediction


# In[123]:


plt.scatter(x,y)
test=np.arange(100,160,10).reshape(-1,1)
plt.plot(test,model2.predict(test), c="r")

#testing with one sample 
#heightp = np.array([24]).reshape(-1,1)
tempp = model.predict(heightp)
#plt.scatter(heightp,tempp,c="y")

plt.xlabel("Calidad del aire)")
plt.ylabel("Altura (M)")
plt.show()


# Aqui podemos ver la relación entre calidad del aire y altura, a más altura mejor es la calidad del aire 
