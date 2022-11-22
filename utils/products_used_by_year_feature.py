from sklearn.base import BaseEstimator, TransformerMixin


class ProductsUsedByYearFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[:, "products_used_year"] = X.apply(self.products_used_by_year, axis=1)

        return X

    def products_used_by_year(self, row):
        if row["num_of_products"] <= 0:
            return 0
        elif row["tenure"] <= 0:
            return row["num_of_products"]
        else:
            return round(row["num_of_products"] / row["tenure"], 2)
