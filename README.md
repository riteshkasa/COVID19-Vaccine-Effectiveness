# COVID19-Vaccine-Effectiveness

CSV Files: 
- LSTM Neural Network uses the state-flipped csv files
- Autoregressive Model uses the state-history csv files

# Abstract
The COVID-19 pandemic has been one of the most devastating events in recent history, resulting in
millions of deaths and destruction on the global economy. While vaccines have been developed to slow the
spread of the virus and achieve global herd immunity, millions of US citizens refuse to take them due to
speculation regarding their safety and effectiveness. Our project goal is to reassure people that COVID vaccines
are effective at controlling the spread of the virus and reducing the number of people infected. To demonstrate
this, we use an autoregressive (AR) model and long short-term memory (LSTM) network to represent the spread
of COVID over time. Using data from various US states, we display COVID trends over the last year and make
predictions on how the disease will spread in the future (beyond the scope of our data set) with and without
vaccines. In the end, our predictions show that vaccines are effective at reducing cases and slowing the spread
of the disease. By comparing results from both models for each state, we were able to choose the more accurate
model and use it for our graphs and predictions. After comparing sources of error in our models (root mean
square error and coefficient of determination), our results indicated that the LSTM neural network was much
more accurate than the autoregressive model.

# Autoregressive (AR) Model Predictions:

States: California New York Washington Kansas Connecticut

RMSE: 30311.5033 12630.3163 3178.8517 5891.6085 2576.2446

R2: 0.8153 0.8168 0.7327 -0.3837 0.7889

# Long Short Term Memory (LSTM) Recurrent Neural Network Predictions

States: California New York Washington Kansas Connecticut

RMSE: 34688.1765 5803.3196 2031.2507 2022.4028 2403.7648

R2: 0.75818 0.9613 0.8908 0.8369 0.8162
