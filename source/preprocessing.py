import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

class preprocessor:
    
    def __init__(self, dataframe):

        self.df = dataframe

    def trainTestSplit(self):
        
        # Dependent Variable
        output_var = pd.DataFrame(self.df['Adj Close'])

        # Indepenent Variables
        features = ['Open', 'High', 'Low', 'Volume']
        
        # Scale the Features
        scaler = MinMaxScaler()
        feature_transform = scaler.fit_transform(self.df[features])
        feature_transform = pd.DataFrame(columns = features, data = feature_transform, index = self.df.index)
        
        # Split into train and test data.
        timesplit = TimeSeriesSplit(n_splits=10)
        for train_index, test_index in timesplit.split(feature_transform):
        
            X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
            y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index) + len(test_index))].values.ravel()

        # Process the data for LSTM.
        trainX = np.array(X_train)
        testX = np.array(X_test)
        X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        return X_train, y_train, X_test, y_test