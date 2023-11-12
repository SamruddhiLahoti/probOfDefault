import pandas as pd
import numpy as np

from config import *
from utils import load_dataset, extract_feature_values
from preprocess import Preprocessor
from models import XGBoost


def define_target_var(dataset):
    preprocessor = Preprocessor()

    dataset = preprocessor.to_datetime(dataset)
    dataset[TARGET] = dataset.apply(preprocessor.default_logic, axis=1)
    # dropping records for which def_date is within 6 months of statement date
    dataset.drop(dataset[dataset[TARGET] == -1].index, inplace=True)
    dataset.reset_index(inplace=True, drop=True)

    return dataset


def create_test_file():
    dataset = load_dataset(DATA_PATH)
    dataset = define_target_var(dataset)

    sample_size = 100
    default_rate = 0.2

    columns_to_nan = ['asst_tot', 'prof_operations', "eqty_tot", "debt_st"]
    nan_percentage = 0.05

    non_defaults = dataset[dataset[TARGET] == 0].sample(int(sample_size * (1 - default_rate)))
    defaults = dataset[dataset[TARGET] == 1].sample(int(sample_size * default_rate))

    final_df = pd.concat([non_defaults, defaults]).sample(frac=1).reset_index(drop=True)

    for column in columns_to_nan:
        nan_mask = np.random.rand(len(final_df)) < nan_percentage
        final_df.loc[nan_mask, column] = np.nan

    final_df.to_csv("/Users/lahosa/Documents/NYU/Fall 2023/ML in Finance/Project/test.csv")


def simulate_harness():

    dataset = load_dataset(DATA_PATH)
    dataset = define_target_var(dataset)
    dataset = extract_feature_values(dataset)

    model = XGBoost()
    model.load_trained_model()

    dataset["predicted_pd"] = model.predict(dataset[FEATURE_SET])
    # dataset["predicted"] = dataset["predicted_pd"].apply(lambda x: int(x > 0.5))

    # default_rate = dataset[dataset["predicted"] == 1].shape[0] / dataset.shape[0]
    # dataset.loc[dataset[FEATURE_SET].isnull().any(axis=1), "predicted_pd"] = np.nan

    print(dataset[["is_default", "predicted_pd"]])


if __name__ == "__main__":
    simulate_harness()
