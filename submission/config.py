TARGET = "is_default"

DATA_PATH = "/Users/lahosa/Documents/NYU/Fall 2023/ML in Finance/Project/train.csv"

LR_SAVE_PATH = "/Users/lahosa/Documents/NYU/Fall 2023/ML in Finance/Project/lr_model.pkl"
NN_SAVE_PATH = "/Users/lahosa/Documents/NYU/Fall 2023/ML in Finance/Project/neural_net"
XGB_SAVE_PATH = "/Users/lahosa/Documents/NYU/Fall 2023/ML in Finance/Project/xgb_model.json"

NORM_MEAN = 11163910.359835
NORM_STD = 182824801.73946536

DAYS_IN_YEAR = 365

FEATURE_SET = ["cash_ratio", "cash_roa", "debt_assets_lev", "asset_turnover", "norm_asst_tot",
               "legal_struct", "ateco_sector", "HQ_city"]
CATEGORICAL_FT = ["legal_struct", "ateco_sector", "HQ_city"]


FINAL_MODEL = "xgb"
