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
data = pd.read_csv('new-york-history.csv',index_col=0,parse_dates=True)
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
plt.title("Number of Cases of COVID-19")
plt.show()


# In[6]:


ad_fuller_result = adfuller(data['positive'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')


# In[7]:


train_data = data['positive'][:len(data)]
novaccine_data = data['positive'][:len(data)-58]


# In[8]:


ar_model = AutoReg(train_data, lags=2).fit()
ar2_model = AutoReg(novaccine_data, lags=2).fit()


# In[11]:


pred = ar_model.predict(start=len(train_data), end=(len(data)+7), dynamic=False)
pred2 = ar2_model.predict(start=len(novaccine_data), end=(len(data)+7), dynamic=False)

plt.figure(figsize=[12, 6]);
plt.title('New York COVID-19 Future Total Cases Prediction: With and Without Vaccines')
plt.ylabel('Total Number of COVID-19 Cases in Millions')
plt.xlabel('Date')
# plt.plot(train_data, color='black', label = 'Real-time Data')
# plt.plot(test_data, color='red', label = 'Real-time Data when Vaccines Came')
# plt.plot(pred2, label = 'Total Future Cases Prediction (Without Vaccines)')
plt.plot(pred, label = 'Total Future Cases Prediction (With Vaccines)')
plt.legend()

