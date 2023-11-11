import numpy as np

import statsmodels.formula.api as smf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class LogisticRegression:
    def __init__(self, features, target, data):
        self.formula = target + " ~ " + " + ".join(features)
        self.data = data

    def fit_model(self):
        logit_model = smf.logit(self.formula, data=self.data)
        return logit_model.fit(disp=False)


class NeuralNetwork:

    def __init__(self, target, data):

        self.data = data
        self.loss_func = "binary_crossentropy"
        self.target = target
        pass

    def __network(self, input_dim):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),  # Input layer
            keras.layers.Dense(128, activation='relu'),  # Hidden layer with ReLU activation
            keras.layers.Dense(64, activation='relu'),  # Another hidden layer
            keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for probability
        ])

        model.compile(optimizer='adam', loss=self.loss_func, metrics=['accuracy'])

        return model

    def fit_model(self, epochs=5):

        scaler = StandardScaler()
        x = scaler.fit_transform(self.data.drop(self.target))
        y = self.data[self.target]
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        weights = dict(zip(np.unique(y), weights))

        net = self.__network(x.shape[0])
        net.fit(x, y, epochs=epochs, batch_size=32, class_weight=weights)
        return net
