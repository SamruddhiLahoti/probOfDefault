import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import FEATURE_SET, CATEGORICAL_FT, NORM_STD, NORM_MEAN, TARGET


def load_dataset(data_path):
    dataset = pd.read_csv(data_path)
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop("Unnamed: 0", axis=1).reset_index(drop=True)
    return dataset


# -------------------------------------- Preprocessor Util Functions --------------------------------------------------#

def to_categorical(df):
    df[CATEGORICAL_FT] = df[CATEGORICAL_FT].astype("category")
    return df


def extract_feature_values(df):
    """
    Extracting required features from the dataset
    """

    df = to_categorical(df)

    # Existing features
    df["rev_operating"].fillna(df["prof_operations"] + df["COGS"], inplace=True)

    # New features
    df["cash_roa"] = df["cf_operations"] / df["asst_tot"]
    df["debt_assets_lev"] = (df["asst_tot"] - df["eqty_tot"]) / df["asst_tot"]
    df["cash_ratio"] = df["cash_and_equiv"] / df["debt_st"]
    df["asset_turnover"] = df["rev_operating"] / df["asst_tot"]
    df["norm_asst_tot"] = (df["asst_tot"] - NORM_MEAN) / NORM_STD

    df = df[FEATURE_SET + [TARGET]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ----------------------------------- Feature Extraction Util Functions -----------------------------------------------#

def assert_low_VIF(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    assert sum(vif_data["VIF"] > 2) == 0


