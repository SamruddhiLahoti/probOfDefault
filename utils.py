import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import DAYS_IN_YEAR, NORM_STD, NORM_MEAN


def load_dataset(data_path):
    dataset = pd.read_csv(data_path)
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop("Unnamed: 0", axis=1).reset_index(drop=True)
    return dataset


# -------------------------------------- Preprocessor Util Functions --------------------------------------------------#

# FEATURE_SET = ["wc_net", "roa", "debt_st", "working_capital_turnover", "asst_tot", "defensive_interval",
#                "debt_assets_lev", "current_ratio", "debt_equity_lev", "cash_ratio", "receivable_turnover",
#                "avg_receivables_collection_day", "asset_turnover", "net_profit_margin_on_sales"]


def extract_feature_values(df):
    """
    Fills in missing values for a few relevant features based on formulas derived from sanity checks
    """
    # No formulae: wc_net, debt_st, asst_tot, prof_operations, COGS, cash_and_equiv, AR, eqty_tot, asst_current,
    #       profit

    df = df[["wc_net", "debt_st", "asst_tot", "prof_operations", "COGS", "cash_and_equiv", "AR", "eqty_tot",
             "asst_current", "profit", "roa", "rev_operating"]]

    # Existing features
    df["roa"].fillna(df["prof_operations"] / df["asst_tot"] * 100, inplace=True)
    df["rev_operating"].fillna(df["prof_operations"] + df["COGS"], inplace=True)

    # New features
    df["working_capital_turnover"] = df["rev_operating"] / df["wc_net"]
    df["defensive_interval"] = (df["cash_and_equiv"] + df["AR"]) * DAYS_IN_YEAR / \
                               (df["COGS"] + df["rev_operating"] - df["prof_operations"])
    df["debt_assets_lev"] = (df["asst_tot"] - df["eqty_tot"]) / df["asst_tot"]
    df["current_ratio"] = df["asst_current"] / df["debt_st"]
    df["debt_equity_lev"] = (df["asst_tot"] - df["eqty_tot"]) / df["eqty_tot"]
    df["cash_ratio"] = df["cash_and_equiv"] / df["debt_st"]
    df["receivable_turnover"] = df["rev_operating"] / df["AR"]
    df["avg_receivables_collection_day"] = DAYS_IN_YEAR / df["receivable_turnover"]
    df["asset_turnover"] = df["rev_operating"] / df["asst_tot"]
    df["net_profit_margin_on_sales"] = df["profit"] / df["rev_operating"]

    df["norm_asst_tot"] = (df["asst_tot"] - NORM_MEAN) / NORM_STD

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ----------------------------------- Feature Extraction Util Functions -----------------------------------------------#

def assert_low_VIF(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    assert sum(vif_data["VIF"] > 2) == 0
    # return vif_data


