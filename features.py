import numpy as np
import statsmodels.formula.api as smf

from sklearn.feature_selection import RFE
import sklearn.metrics as metrics
from sklearn import tree

import matplotlib.pyplot as plt

from config import TARGET

import warnings
warnings.filterwarnings('ignore')


class FeatureSelection:

    def __init__(self):

        self.feature_sets = {
            "liquidity": ["current_ratio", "cash_ratio", "defensive_interval", "wc_net"],
            "profitability": ["roe", "roa", "gross_profit_margin_on_sales", "net_profit_margin_on_sales", "cash_roa"],
            "leverage": ["debt_assets_lev", "debt_equity_lev", "financial_leverage"],
            "efficiency": ["receivable_turnover", "avg_receivables_collection_day", "asset_turnover",
                           "working_capital_turnover"]
        }
        self.target = TARGET

    def univariate_analysis(self, df, plot=False):

        features_identified = {}
        for feat_set in self.feature_sets:

            print(f"\n{feat_set} Feature Set:\n")

            if plot:
                plt.title(f"{feat_set} ROC")
                plt.plot([0, 1], [0, 1], "r--")
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel("True Positive Rate")
                plt.xlabel("False Positive Rate")

            best_auc = -np.inf
            for feat in self.feature_sets[feat_set]:
                # fit simple logit model
                logit_model_sm = smf.logit(f"{self.target} ~ {feat}", data=df).fit(disp=False)
                pred_def_prob = logit_model_sm.predict(df)

                # calculate the fpr and tpr for all thresholds of the classification
                fpr, tpr, _ = metrics.roc_curve(df[self.target], pred_def_prob)
                feat_auc = metrics.auc(fpr, tpr)

                if best_auc < feat_auc:
                    features_identified[feat_set] = feat
                    best_auc = feat_auc

                print(f"{feat} p_val:", logit_model_sm.pvalues[1])
                if plot:
                    plt.plot(fpr, tpr, label=f"{feat}: AUC {feat_auc:.3f}")

            if plot:
                plt.legend()
                plt.show()

        return list(features_identified.values())

    def rfe_analysis(self, df):
        print()
        features_identified = {}
        for feat_set in self.feature_sets:
            features = self.feature_sets[feat_set]

            rfe_method = RFE(
                tree.DecisionTreeClassifier(max_depth=4),
                n_features_to_select=1,
            )
            rfe_method.fit(df[features], df[self.target])

            features_identified[feat_set] = df[features].columns[(rfe_method.get_support())][0]
            print(f"Most relevant {feat_set} measure: {features_identified[feat_set]}")

        return list(features_identified.values())
