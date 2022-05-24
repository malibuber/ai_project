from keras.models import model_from_json

import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date
import math
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras
data = web.get_data_yahoo('BTC-USD', start=datetime.datetime(2022, 1, 2),
                          end=date.today())
data = data[['Adj Close']]
data.columns = ['Price']
print('There are {} number of days in the dataset.'.format(data.shape[0]))
plt.figure(figsize=(28, 12))  # , dpi=100)
plt.plot(data.index, data, label='BTC-USD Price')
plt.xlabel('Date')
plt.ylabel('Rs')
plt.title('BTC-USD Price')
plt.savefig('eee.png')


def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Price'].rolling(window=21).mean()

    # Create MACD
    dataset['26ema'] = dataset['Price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Price'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema']-dataset['26ema']

    # Create Bollinger Bands
    dataset['20sd'] = dataset['Price'].rolling(window=21).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)

    # Create Exponential moving average
    dataset['ema'] = dataset['Price'].ewm(com=0.5).mean()

    # Create Momentum
    dataset['momentum'] = dataset['Price']-1
    dataset['log_momentum'] = np.log(dataset['momentum'])
    return dataset


df = get_technical_indicators(data)
df = df.dropna()
print(df.head())


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days

    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)

    plt.figure(figsize=(30, 20))
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['Price'], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'],
                     dataset['upper_band'], alpha=0.35)
    plt.title(
        'Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot

    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.plot(dataset['log_momentum'], label='Momentum',
             color='b', linestyle='-')

    plt.legend()
    plt.savefig('ddd.png')


plot_technical_indicators(df, 1000)

plt.figure(figsize=(28, 12))
sns.set_context('poster', font_scale=1)
heat_map = sns.heatmap(df.corr(), annot=True).set_title('Params')
figure = heat_map.get_figure()
figure.savefig('svm_conf.png', dpi=400)
print('Total dataset has {} samples, and {} features.'.format(df.shape[0],
                                                              df.shape[1]))

print(df.columns)

data_training = df[df.index < '2019-01-31'].copy()
data_testing = df[df.index >= '2019-01-31'].copy()

scalar = MinMaxScaler()

data_training_scaled = scalar.fit_transform(data_training)
print(data_training_scaled.shape)

X_train = []
y_train = []
for i in range(60, data_training.shape[0]):
    X_train.append(data_training_scaled[i-60: i])
    y_train.append(data_training_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


past_60 = data_training.tail(60)

dt = past_60.append(data_testing, ignore_index=True)

inputs = scalar.fit_transform(dt)
print(inputs.shape)

X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
loaded_model.compile(optimizer='adam', loss='mean_squared_error')
y_pred = loaded_model.predict(X_test)
scale = 1/scalar.scale_[0]

y_pred = y_pred*scale
print(y_pred)
