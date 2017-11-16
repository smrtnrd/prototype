from math import sqrt
from numpy import concatenate
from matplotlib import pyplot

import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def prepare_data(data, normalize=True):
    """
    Return a dataframe of that stock and normalize all the values. 
    (Optional: create moving average)
    """
    dataset = read_csv(data, names=['statename','YEAR','MONTH','WEEK','weekend','ili_activity_label','a_2009_h1n1','Latitude','Longitude','week_TEMP','week_MAX','week_MIN'])
    print(dataset.head(5))
    # manually specify column names
    dataset.index.name = 'date'

    

    if normalize:
        # integer encoder
        label_encoder = LabelEncoder()
        ili_values = dataset['ili_activity_label'].reshape(-1,1)
        dataset['ili_activity_label'] = label_encoder.fit_transform(ili_values)
        min_max_scaler = MinMaxScaler()
        dataset['ili_activity_label'] = min_max_scaler.fit_transform(ili_values)
        dataset['Latitude'] = min_max_scaler.fit_transform(dataset.Latitude.reshape(-1,1))
        dataset['Longitude'] = min_max_scaler.fit_transform(dataset.Longitude.reshape(-1,1))
    
    # Move Adj Close to the rightmost for the ease of training
    # activity_level = dataset['ili_activity_label']
    # dataset.drop(labels=['activity_level'], axis=1, inplace=True)
    # dataset = pd.concat([dataset, activity_level], axis=1)
    dataset.to_csv("../data/ili.csv", sep=',', encoding='utf-8')
    return dataset


def denormalize(data, normalized_value):
    dataset = read_csv(data, names=['statename','YEAR','MONTH','WEEK','weekend','ili_activity_label','a_2009_h1n1','Latitude','Longitude','week_TEMP','week_MAX','week_MIN'])
    print(dataset.head(5))
    # manually specify column names
    dataset.index.name = 'date'

    dataset = dataset['activity_level'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)

    #return df.shape, p.shape
    min_max_scaler = MinMaxScaler()
    a = min_max_scaler.fit_transform(dataset)
    new = min_max_scaler.inverse_transform(normalized_value)
      
    return new

# load dataset
data = '../data/2010-2015_ili_sub_climate.csv'

scaled = prepare_data(data)

# frame as supervised learning
reframed = series_to_supervised(scaled, 8, 4)
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())
 
# split into train and test sets
values = reframed.values
n_train_weeks = round(0.8*len(values))
train = values[:n_train_weeks, :]

print('train data :')
print(train)
test = values[n_train_weeks:, :]
print('test data :')
print(test)
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
newp = denormalize(data, yhat)
print(newp)
newy_test =denormalize(data, test_y) 
print(newy_test)
# calculate RMSE
rmse = sqrt(mean_squared_error(newy_test, newp))
print('Test RMSE: %.3f' % rmse)
