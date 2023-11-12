import pickle
import numpy as np

import statsmodels.formula.api as smf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier, Booster, DMatrix

import xgboost as xgb

from config import XGB_SAVE_PATH, NN_SAVE_PATH, LR_SAVE_PATH


class LogisticRegression:
    def __init__(self):
        self.model = None

    def fit_model(self, data, target, features):
        formula = target + " ~ " + " + ".join(features)
        data = data
        self.model = smf.logit(formula, data=data).fit(disp=False)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def save_model(self):
        with open(LR_SAVE_PATH, 'wb') as file:
            pickle.dump(self.model, file)

    def load_trained_model(self):
        with open(LR_SAVE_PATH, 'rb') as file:
            self.model = pickle.load(file)


class NeuralNetwork:

    def __init__(self):
        # self.data = data
        self.loss_func = "binary_crossentropy"
        # self.target = target

        self.scaler = StandardScaler()
        self.model = None

    def __network(self, input_dim):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),  # Input layer
            keras.layers.Dense(128, activation='relu'),  # Hidden layer with ReLU activation
            keras.layers.Dense(64, activation='relu'),  # Another hidden layer
            keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for probability
        ])

        model.compile(optimizer='adam', loss=self.loss_func, metrics=['accuracy'])

        return model

    def fit_model(self, data, target, features, epochs=5):
        x = self.scaler.fit_transform(data[features])
        y = data[target]

        weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        weights = dict(zip(np.unique(y), weights))

        self.model = self.__network(x.shape[1])
        self.model.fit(x, y, epochs=epochs, batch_size=32, class_weight=weights)

    def predict(self, test_data):
        x = self.scaler.fit_transform(test_data)
        return self.model.predict(x)

    def save_model(self):
        self.model.save(NN_SAVE_PATH)

    def load_trained_model(self):
        self.model = keras.models.load_model(NN_SAVE_PATH)


class XGBoost:

    def __init__(self):
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': 0.013
        }
        self.model = None

    def fit_model(self, data, target, features):
        dtrain = xgb.DMatrix(data[features], label=data[target], enable_categorical=True)
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)

    def predict(self, test_data):
        dtest = xgb.DMatrix(test_data, enable_categorical=True)
        return self.model.predict(dtest)

    def save_model(self):
        # with open(XGB_SAVE_PATH, 'wb') as file:
        #     pickle.dump(self.model, file)
        self.model.save_model(XGB_SAVE_PATH)

    def load_trained_model(self):
        # with open(XGB_SAVE_PATH, 'rb') as file:
        #     self.model = pickle.load(file)
        self.model = xgb.Booster(model_file=XGB_SAVE_PATH)

