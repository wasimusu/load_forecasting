## Accurate Load prediction using Historical Data
The need to reduce the carbon footprint drastically over the next decade requires a switch to Electric power instead of carbon fuels. Such a dramatic switch will require energy efficient energy systems in all aspects of life. This work explores classical machine learning methods to achieve accurate load prediction ahead of time.

## Methods
### Classical Methods
- Random Forests using xgboost (https://github.com/wasimusu/load_forecasting/blob/master/boosting.py)
```
python boosting.py
```

- Support Vector Regression (https://github.com/wasimusu/load_forecasting/blob/master/svr.py)
```
python svr.py
```

### Neural Network Methods
- Linear regression using FC feedforward network
- Linear regression using RNN variants (LSTM)
- Autoencoders using LSTMs

## Dataset
I. First Dataset :
PJM Hourly Energy Consumption Data - https://www.kaggle.com/robikscube/hourly-energy-consumption
PJM Interconnection is an  based in US. From the snippet below, it is easy to spot the daily and seasonal trends. That's just one region. There are few more regions. That's why I think it's a good data. The drawback of this data is that it has only two attributes - data and load. Some of the things that could be done with the data is
Building a model to predict energy consumption

<img src="https://github.com/wasimusu/load_forecasting/blob/master/Images/AEP%20dataset.png" width= "600">

II. Second Dataset :
It has total household power consumption with a 30 min interval  from 6031 residential houses from 2013. The data has been cleaned and missing values have been filled.
<img src="https://github.com/wasimusu/load_forecasting/blob/master/Images/dataset2.PNG" width="600">

## Motivations
- Building a model to predict energy consumption
- Find trends in energy consumption around hours of the day, holidays, or long term trends.
- Understand how daily trends change depending of the time of year.

## Dependencies
* python3.7
* xgboost
* statsmodels
* holidays
* sklearn
* matplotlib
* numpy
* pandas
* holidays

## Installations
```
pip3 install -r requirements.txt
```

## Running exectuables
* fname : filename of dataset. Change fname to change dataset. For each executable fname is inside the __main__.

## Files explained
#### All codes are work of the authors unless referred or cited.
* datareader.py : Implements data reader
* datarepr.py : Implements functions relating to data representation. Date can be represented as sin, cos, (sin, cos) pair or without any encoding.
* utils.py : Implements timer to time performance of various functions
* process.py : Implements data aggregator functions which aggregates hourly or so data into daily data.
* networks.py : Implements different types of Neural Network Architectures.
* models.py : Implements training module for above architectures.
* \data : contains datasets containing hourly samples or hourly samples aggregated into daily samples.
* uber\networks.py and uber\models.py : Similar to above networks.py and models.py

### Requirements for Neural Networks
* https://pytorch.org/
```
pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
#### Running neural network models
```
python models.py
```

### Results
<img src="https://github.com/wasimusu/load_forecasting/blob/master/Images/aep%20daily%20svr%20scatter.png" width = "500">
<img src="https://github.com/wasimusu/load_forecasting/blob/master/Images/aep%20daily%20svr%20line.png" width ="500">
                                                                                                                   
### Summary
In this case, classical methods outperformed neural networks while also simultaneously being easier to train, debug and explain. Using Sine, Cosine and Pair representation of dates also produced similar results. Even without explicit usage of date and time, the prediction preformed just as good meaning that dates were not as important in this case.

### Conclusion
SVR with polynomial kernel and Boosting (Random Forest) can be used to do load forecasting. Using windows of previous N-1 loads to augment the input features helps to significantly improve the prediction results. The size of window plays critical role in being able to accurately predict lower and higher extremes. 

### Computational Study
We have conducted a study comparing the performance of SVR on each dataset, AEP and Household for hourly prediction. In each case the model has been trained on the last 7 hours. Prediction on the  household data performs much better than the AEP data.

<img src="https://github.com/wasimusu/load_forecasting/blob/master/Images/compare%20dataset%201.png" width = "500">
<img src="https://github.com/wasimusu/load_forecasting/blob/master/Images/compare%20dataset%202.png" width = "500">
