import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

from config import *
from features import FeatureSelection
from models import *
from preprocess import Preprocessor


def plot(pred_list, est_name):
    # Combine predictions and true labels from all walk-forward steps
    pred_df = pd.concat(pred_list)

    all_predictions = pred_df['y_pred_prob']
    all_true_labels = pred_df['y_true']

    # Calculate the ROC curve
    fpr, tpr, _ = metrics.roc_curve(all_true_labels, all_predictions)

    # Calculate the AUC
    roc_auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{est_name} ROC")
    plt.legend(loc='lower right')
    plt.show()

    # Display the AUC value
    print(f'\nAUC: {roc_auc:.6f}')


class WalkForward:

    def __init__(self, dataset):

        self.dataset = dataset
        self.target = TARGET

        self.model = None
        self.train_mean, self.train_std = None, None

    def __save_params(self):
        # FIXME
        pass

    def __normalize_assets(self, train_df, test_df):
        """
        The total asset values in both train and test datasets are normalized
        using mean and std from ONLY the train dataset
        Args:
            train_df: the train slice
            test_df: the test slice
        Returns:
            train and test slices with asst_tot normalized
        """
        self.train_mean = train_df["asst_tot"].mean()
        self.train_std = train_df["asst_tot"].std()

        train_df["norm_asst_tot"] = (train_df["asst_tot"] - self.train_mean) / self.train_std
        test_df["norm_asst_tot"] = (test_df["asst_tot"] - self.train_mean) / self.train_std
        return train_df, test_df

    def __predictor(self, test_df, features):
        pred_df = pd.DataFrame()

        if not test_df.empty:
            pred_df['y_true'] = test_df[self.target].copy()
            pred_df['y_pred_prob'] = self.model.predict(test_df[features])

        return pred_df

    def __estimator(self, est_name, features, data, epochs):
        if est_name == "logistic":
            self.model = LogisticRegression()
            self.model.fit_model(data, self.target, features)
            return self.model

        elif est_name == "NN":
            self.model = NeuralNetwork()
            self.model.fit_model(data, self.target, features, epochs)
            return self.model

        elif est_name == "xgb":
            self.model = XGBoost()
            self.model.fit_model(data, self.target, features)
            return self.model

    def __walk_forward(self, est_name, features, epochs, start_year):

        # df = self.preprocessor.preprocess(df)
        max_year = self.dataset['fs_year'].max()

        model_list, pred_list = [], []

        for year in range(start_year, max_year+1):

            train_df = self.dataset[self.dataset['fs_year'] <= year].copy()
            test_df = self.dataset[self.dataset['fs_year'] == year+1].copy()
            train_df, test_df = self.__normalize_assets(train_df, test_df)

            default_rate = train_df[train_df[self.target] == 1].shape[0] / train_df.shape[0]
            print(f"\nYear: {year} | Sample default rate: {default_rate * 100:.3f}%")

            model = self.__estimator(est_name, features, train_df, epochs)
            pred = self.__predictor(test_df, features)

            model_list.append(model)
            pred_list.append(pred)

            del train_df, test_df

        return model_list, pred_list

    def run(self, est_name, features, epochs=5, start_year=2008):
        model_list, pred_list = self.__walk_forward(est_name, features, epochs, start_year)

        self.__save_params()
        plot(pred_list, est_name)


if __name__ == "__main__":

    dataset = pd.read_csv(DATA_PATH).drop("Unnamed: 0", axis=1).reset_index(drop=True)

    preprocessor = Preprocessor()
    fs = FeatureSelection()

    dataset = preprocessor.preprocess(dataset)

    # univariate_features = fs.univariate_analysis(dataset)
    # assert_low_VIF(dataset[univariate_features])
    #
    # rfe_features = fs.rfe_analysis(dataset)
    # assert_low_VIF(dataset[rfe_features])

    wf = WalkForward(dataset)
    wf.run("xgb", FEATURE_SET)

    # print(f"Final mean: {wf.train_mean} and std: {wf.train_std}")


