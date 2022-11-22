from sklearn.base import BaseEstimator, TransformerMixin


class EstimatedMonthlySalaryFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[:, "estimated_monthly_salary"] = X.apply(
            self.estimated_monthly_salary, axis=1
        )
        return X

    def estimated_monthly_salary(self, row):
        return round(row["estimated_salary"] / 12, 2)
