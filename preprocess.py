# import key packages
import pandas as pd
import numpy as np

from config import TARGET


class Preprocessor:
    def __init__(self):
        # self.ateco_sector_map = {}

        self.categorical_feat = ["legal_struct", "ateco_sector", "HQ_city"]

        self.stmt_delay = 0.5
        self.days_in_year = 365

        self.target = TARGET

        self.features = ["stmt_date", "HQ_city", "legal_struct", "ateco_sector", "def_date", "fs_year",
                         "asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin", "asst_current", "AR",
                         "cash_and_equiv", "asst_tot", "eqty_tot", "debt_st", "rev_operating", "COGS",
                         "prof_operations", "goodwill", "exp_financing", "profit", "roa", "roe", "wc_net",
                         "margin_fin", "cf_operations"]

        self.new_features = ["working_capital_turnover", "defensive_interval", "debt_assets_lev", "current_ratio",
                             "cash_roa", "debt_equity_lev", "cash_ratio", "receivable_turnover",
                             "avg_receivables_collection_day", "asset_turnover", "net_profit_margin_on_sales"]

    def __default_logic(self, row):
        """
        Define the binary target variable for default.
        If def_date is non-empty, it is assumed that the firm defaults at some point in our training data

        Args:
            row: the dataframe record
        Returns:
            1, if the firm"s default date is within 6 to 18 months of the stmt_date
            0, if the firm"s default date null or is outside the valid range
        """
        if pd.notna(row["def_date"]):
            days = (row["def_date"] - row["stmt_date"]).days
            # def_date is before statement date
            if days < 0:
                return 0
            # def_date is within 6 months of statement date, it isn"t valid
            elif days <= self.days_in_year * self.stmt_delay:
                return -1
            return int(self.days_in_year * self.stmt_delay < days < self.days_in_year * (1 + self.stmt_delay))
        return 0

    def __convert_data_types(self, df):
        """

        Args:
            df:
        Returns:

        """
        df["def_date"] = pd.to_datetime(df["def_date"], dayfirst=True)
        df["stmt_date"] = pd.to_datetime(df["stmt_date"])

        df[self.categorical_feat] = df[self.categorical_feat].astype("category")
        return df

    def __fill_missing_values(self, df):

        df["roa"].fillna(df["prof_operations"] / df["asst_tot"] * 100, inplace=True)
        df["margin_fin"].fillna(
            df["eqty_tot"] - df["asst_intang_fixed"] - df["asst_tang_fixed"] - df["asst_fixed_fin"],
            inplace=True
        )
        df["rev_operating"].fillna(df["prof_operations"] + df["COGS"], inplace=True)
        df["roe"].fillna(df["profit"] / df["eqty_tot"], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    def __calc_financial_ratios(self, df):
        
        # liquidity
        df["current_ratio"] = df["asst_current"] / df["debt_st"]
        df["cash_ratio"] = df["cash_and_equiv"] / df["debt_st"]
        df["defensive_interval"] = (df["cash_and_equiv"] + df["AR"]) * self.days_in_year / (df["COGS"] + df["rev_operating"] - df["prof_operations"])
        # +wc_net

        # profitability
        df["gross_profit_margin_on_sales"] = df["prof_operations"] / df["rev_operating"]
        df["net_profit_margin_on_sales"] = df["profit"] / df["rev_operating"]
        df["cash_roa"] = df["cf_operations"] / df["asst_tot"]
        # FIXME: assumption
        # df = df[df["cash_roa"] <= 50]
        # +roe
        # +roa

        # leverage
        df["debt_assets_lev"] = (df["asst_tot"] - df["eqty_tot"]) / df["asst_tot"]
        df["debt_equity_lev"] = (df["asst_tot"] - df["eqty_tot"]) / df["eqty_tot"]
        df["financial_leverage"] = df["roe"] - df["roa"]

        # efficiency
        # df.drop(df[(df["rev_operating"] == 0.0) & (df["AR"] == 0.0)].index, inplace=True)
        df["receivable_turnover"] = df["rev_operating"] / df["AR"]
        df["avg_receivables_collection_day"] = self.days_in_year / df["receivable_turnover"]
        df["asset_turnover"] = df["rev_operating"] / df["asst_tot"]
        df["working_capital_turnover"] = df["rev_operating"] / df["wc_net"]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    def preprocess(self, df):
        """

        Args:
            df:

        Returns:

        """

        df = df[self.features].copy().reset_index(drop=True)

        df_shape = df.shape[0]
        print(f"Initial number of rows: {df_shape}")

        df = self.__convert_data_types(df)

        df[self.target] = df.apply(self.__default_logic, axis=1)
        # dropping records for which def_date is within 6 months of statement date
        df.drop(df[df[self.target] == -1].index, inplace=True)
        df.reset_index(inplace=True, drop=True)

        rec_dropped = (1 - df.shape[0] / df_shape) * 100
        print(f"After dropping based on target: {df.shape[0]} => {rec_dropped:.3f}% dropped")

        df = self.__fill_missing_values(df)
        df = self.__calc_financial_ratios(df)

        # no longer needed for analysis
        df.drop("def_date", axis=1, inplace=True)

        # dropping na rows
        df.dropna(
            subset=["HQ_city", "wc_net", "roa", "debt_st", "working_capital_turnover", "asst_tot", "defensive_interval",
                    "debt_assets_lev", "current_ratio", "cash_roa", "AR", "rev_operating", "prof_operations",
                    "debt_equity_lev", "cash_ratio", "receivable_turnover", "avg_receivables_collection_day", "goodwill",
                    "asset_turnover", "net_profit_margin_on_sales", "cf_operations", "cash_and_equiv", "margin_fin"],
            how="any",
            inplace=True
        )

        rec_dropped = (1 - df.shape[0] / df_shape) * 100
        print(f"After dropping NaNs: {df.shape[0]} => {rec_dropped:.3f}% dropped (of original)")

        df["norm_asst_tot"] = (df["asst_tot"] - df["asst_tot"].mean()) / df["asst_tot"].std()

        # print(f"After dropping na, num NaNs in df:\n{df.isna().sum()}")

        return df
