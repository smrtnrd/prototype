
from math import sqrt
from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from pandas import DataFrame
from pandas import concat
from pandas import read_csv

from model import helpers


class Model(object):

    def __init__(self, data, n_lag,n_seq, n_test, n_epochs, n_batch, n_neurons):
        self.data = data
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_test = n_test
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
    
    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    
    @staticmethod
    # transform series into train and test sets for supervised learning
    def prepare_data(data):
        """
        Return a dataframe of that stock and normalize all the values. 
        (Optional: create moving average)
        """
        scaled = data

        # frame as supervised learning
        reframed = Model.series_to_supervised(scaled, 8, 4)
        # drop columns we don't want to predict
        # reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
 
        # split into train and test sets
        values = reframed.values
        n_train_weeks = round(0.8*len(values))
        train = values[:n_train_weeks, :]
        test = values[n_train_weeks:, :]

        # split into train and test sets
        values = reframed.values
        n_train_weeks = round(0.8*len(values))
        train = values[:n_train_weeks, :]
        test = values[n_train_weeks:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        

        return train_X, train_y, test_X, test_y


    def denormalize(data, normalized_value):
        dataset = read_csv(data, index_col=3, header=0)
        # manually specify column names
        dataset.columns = ['statename','activity_level','activity_level_label','season','weeknumber','Latitude','Longitude']
        dataset.index.name = 'date'
            
        # convert index to datetime
        dataset.index = pd.to_datetime(dataset.index, format='%b-%d-%Y')
            
        # manually remove the feature we don;t want to evaluate 
        dataset.drop(['statename', 'season', 'weeknumber','activity_level_label'], axis=1, inplace=True)

        dataset = dataset['activity_level'].values.reshape(-1,1)
        normalized_value = normalized_value.reshape(-1,1)

        #return df.shape, p.shape
        min_max_scaler = MinMaxScaler()
        a = min_max_scaler.fit_transform(dataset)
        new = min_max_scaler.inverse_transform(normalized_value)
        
        return new


    # def build_model(self):
    #     model = Sequential()

    #     model.add(LSTM(self.neurons[0], input_shape=(self.shape[0], self.shape[1]), return_sequences=True))
    #     model.add(Dropout(self.dropout))

    #     model.add(LSTM(self.neurons[1], input_shape=(self.shape[0], self.shape[1]), return_sequences=False))
    #     model.add(Dropout(self.dropout))

    #     model.add(Dense(self.neurons[2],kernel_initializer="uniform",activation='relu'))
    #     model.add(Dense(self.neurons[3],kernel_initializer="uniform",activation='linear'))
    #     # model = load_model('my_LSTM_stock_model1000.h5')

    #     return model

    # fit an LSTM network to training data
    @staticmethod
    def fit_lstm(train_X, train_y,test_X, test_y, nb_epoch = 5, n_batch=8 ):
        
        #train, n_lag, n_seq, n_batch, n_epochs, n_neurons
        # prepare data
       
        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        model.save('LSTM-test-201710.h5')
        # fit network
        
                model.fit(train_X, train_y, 
                    epochs=1, 
                    batch_size=n_batch,
                    validation_data=(test_X, test_y), 
                    verbose=2, 
                    shuffle=False)
                model.reset_states()
        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        
        return model

    # make one forecast with an LSTM,
    def forecast_lstm(model, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    def train_model(self):
        series = self.data
        n_lag = self.n_lag
        n_seq = self.n_seq
        n_test = self.n_test
        n_epochs = self.n_epochs
        n_batch = self.n_batch
        n_neurons = self.n_neurons
        
        #prepare the data
        train_X, train_y, test_X, test_y = self.prepare_data(series)

        # fit model
        model = Model.fit_lstm(train_X, train_y, test_X, test_y)
        # make a prediction
        yhat = model.predict(test_X)
        newp = denormalize(data, yhat)
        newy_test =denormalize(data, test_y) 
        # calculate RMSE
        rmse = sqrt(mean_squared_error(newy_test, newp))
        print('Test RMSE: %.3f' % rmse)


