import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler

from preprocess import Preprocessor
from features import FeatureSelection


def assert_low_VIF(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    assert sum(vif_data["VIF"] > 2) == 0
    # return vif_data


def predictor(model, test_df, features, target='default'):
    new_df = pd.DataFrame()

    if not test_df.empty:
        scaler = StandardScaler()
        x = scaler.fit_transform(test_df[features])

        new_df['y_true'] = test_df[target].copy()
        new_df['y_pred_prob'] = model.predict(x)

    return new_df


if __name__ == "__main__":
    file_path = "/Users/lahosa/Documents/NYU/Fall 2023/ML in Finance/Project/train.csv"

    df = pd.read_csv(file_path).drop("Unnamed: 0", axis=1).reset_index(drop=True)

    preprocessor = Preprocessor()
    fs = FeatureSelection()

    df = preprocessor.preprocess(df)

    univariate_features = fs.univariate_analysis(df)
    assert_low_VIF(df[univariate_features])

    rfe_features = fs.rfe_analysis(df)
    assert_low_VIF(df[rfe_features])
