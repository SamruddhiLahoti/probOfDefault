TARGET = "is_default"

ROOT_PATH = "/Users/lahosa/Documents/NYU/Fall 2023/ML in Finance/Project/"

DATA_PATH = ROOT_PATH + "train.csv"

LR_SAVE_PATH = ROOT_PATH + "lr_model.pkl"
NN_SAVE_PATH = ROOT_PATH + "neural_net"
XGB_SAVE_PATH = ROOT_PATH + "xgb_model.json"

NORM_MEAN = 11163910.359835
NORM_STD = 182824801.73946536

DAYS_IN_YEAR = 365

# FEATURE_SET = ["wc_net", "roa", "debt_st", "working_capital_turnover", "defensive_interval",
#                "debt_assets_lev", "current_ratio", "debt_equity_lev", "cash_ratio", "receivable_turnover",
#                "avg_receivables_collection_day", "asset_turnover", "net_profit_margin_on_sales", "norm_asst_tot"]

FEATURE_SET = ["cash_ratio", "cash_roa", "debt_assets_lev", "asset_turnover", "norm_asst_tot",
               "legal_struct", "ateco_sector", "HQ_city"]
CATEGORICAL_FT = ["legal_struct", "ateco_sector", "HQ_city"]


FINAL_MODEL = "xgb"
