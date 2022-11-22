from sklearn.base import BaseEstimator, TransformerMixin


class CreditScoreCategoryFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X.loc[:, "credit_score_category"] = X.apply(
            self.credit_score_categorization, axis=1
        )
        return X

    def credit_score_categorization(self, row):
        credit_score = row["credit_score"]

        # Worst => 1
        # Best => 7
        if credit_score < 300:
            return 1
        elif credit_score <= 579:
            return 2
        elif credit_score <= 669:
            return 3
        elif credit_score <= 739:
            return 4
        elif credit_score < 799:
            return 5
        elif credit_score <= 850:
            return 6
        else:
            return 7
