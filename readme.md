## Accurate Load prediction using Historical Data
The need to reduce the carbon footprint drastically over the next decade requires a switch to Electric power instead of carbon fuels. Such a dramatic switch will require energy efficient energy systems in all aspects of life. This work explores classical machine learning methods to achieve accurate load prediction ahead of time.

## Methods
### Classical Methods
- Linear regression with regularization
- Support Vector Regression

### Neural Network Methods
- Linear regression using FC feedforward network
- Linear regression using RNN variants (LSTM)

## Dataset
I. First Dataset :
PJM Hourly Energy Consumption Data - https://www.kaggle.com/robikscube/hourly-energy-consumption
PJM Interconnection is an  based in US. From the snippet below, it is easy to spot the daily and seasonal trends. That's just one region. There are few more regions. That's why I think it's a good data. The drawback of this data is that it has only two attributes - data and load. Some of the things that could be done with the data is
Building a model to predict energy consumption

![Dataset 1](https://github.com/wasimusu/load_forecasting/blob/master/Images/AEP%20dataset.png)

II. Second Dataset :
It has total household power consumption with a 30 min interval  from 6031 residential houses from 2013. The data has been cleaned and missing values have been filled.
![Dataset 2](https://github.com/wasimusu/load_forecasting/blob/master/Images/dataset2.PNG)

## Motivations
- Building a model to predict energy consumption
- Find trends in energy consumption around hours of the day, holidays, or long term trends.
- Understand how daily trends change depending of the time of year.
