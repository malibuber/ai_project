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
import pickle
import numpy

data = web.get_data_yahoo('BTC-USD', start=datetime.datetime(2015, 9, 1),
                          end=datetime.datetime(2020, 12, 17))
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

data_training = df[df.index < '2019-04-20'].copy()
data_testing = df[df.index >= '2019-04-20'].copy()

scalar = MinMaxScaler()

data_training_scaled = scalar.fit_transform(data_training)
print(data_training_scaled.shape)

X_train = []
y_train = []
for i in range(60, data_training.shape[0]):
    X_train.append(data_training_scaled[i-60: i])
    y_train.append(data_training_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

regressor = Sequential()

regressor.add(LSTM(units=50, activation='relu',
              return_sequences=True, input_shape=(X_train.shape[1], 12)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units=120, activation='relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units=1))

regressor.summary()

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=100, batch_size=64)

# serialize model to JSON
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")

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
X_test.shape, y_test.shape

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape

y_pred = regressor.predict(X_test)

scale = 1/scalar.scale_[0]

y_pred = y_pred*scale
y_test = y_test*scale
numpy.savetxt("sa.csv", y_pred, delimiter=",")
numpy.savetxt("as.csv", y_test, delimiter=",")
print(y_pred)
print("---")
print(y_test)
# Visualising the results
plt.figure(figsize=(28, 12))
plt.plot(y_test, color='red', label='Real BTC-USD Price')
plt.plot(y_pred, color='blue', label='Predicted BTC-USD Price')
plt.title('BTC-USD Price Prediction-After 100 epochs and Batch Size=64')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('ccc.png')

### ----
y_pred_train = regressor.predict(X_train)

scale = 1/scalar.scale_[0]

y_pred_train = y_pred_train*scale
y_train = y_train*scale
numpy.savetxt("saa.csv", y_pred, delimiter=",")
numpy.savetxt("asa.csv", y_test, delimiter=",")

# Visualising the results
plt.figure(figsize=(28, 12))
plt.plot(y_train, color='red', label='Real BTC-USD Price')
plt.plot(y_pred_train, color='blue', label='Predicted BTC-USD Price')
plt.title('BTC-USD Price Prediction-After 100 epochs and Batch Size=64')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('aaa.png')








