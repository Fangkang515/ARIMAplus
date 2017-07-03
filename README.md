# ARIMAplus
Small project aimed to find the best model for predicting values for a given set of data. 

###### What is going on:
Chosen CSV file is loaded. The file is expected to contain numbers one-number-for-a-line.
A population of several entities is created, based on random 'DNA' code generation, and on
preseted codes. Each entity's purpose is to predict values of a given data set and learn 
during the process, finally to predict future values. Entity's 'DNA' describes used model: 
combination of regression forecast, ARIMA, and neural networks (with varying topologies 
and neuron types).
Entity fitness is based on RMSE of produced forecast.

###### Despite the project per se, second aim is to learn Python, thus avoiding importing external libraries wherever it can be.

###### Yet, there is much to be done
