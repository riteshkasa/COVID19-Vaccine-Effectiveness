#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
data = pd.read_csv('washington-history.csv',index_col=0,parse_dates=True)
data.sort_values('date', inplace=True)
data.fillna(0, inplace=True)
print(data.shape)
data.head()


# In[3]:


plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.scatter(data['positive'], data['positive'])
plt.title('COVID-19 Cases')
plt.ylabel('# of Cases')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[4]:


# data['positive'] = np.log(data['positive'])
# data['positive'] = data['positive'].diff()
# data = data.drop(data.index[0])
# data.fillna(0, inplace=True)
# data.head()


# In[5]:


plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.plot(data['positive'])
plt.title("Number of Cases of COVID-19 (California)")
plt.ylabel('Total Number of COVID-19 Cases in Millions')
plt.xlabel('Date')
plt.show()


# In[6]:


ad_fuller_result = adfuller(data['positive'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')


# In[7]:


train_data = data['positive'][:len(data)-8]
test_data = data['positive'][len(data)-8:]


# In[8]:


ar_model = AutoReg(train_data, lags=2).fit()


# In[9]:


pred = ar_model.predict(start=len(train_data), end=(len(data)-1), dynamic=False)

plt.figure(figsize=[15, 7.5]);
plt.title('New York COVID-19 Total Cases Prediction: 2-28 to 3-07')
plt.ylabel('Total Number of COVID-19 Cases in Millions')
plt.xlabel('Date')
# plt.plot(train_data, color='black', label = 'Real-time Data')
plt.plot(test_data, color='red', label = 'Real-time Data when Vaccines Came')
plt.plot(pred, label = 'Prediction if there were no Vaccines')
plt.legend(prop={'size': 20})


# In[10]:


from sklearn.metrics import mean_squared_error
import math
math.sqrt(mean_squared_error(test_data, pred))


# In[11]:


r2_score(test_data, pred)

