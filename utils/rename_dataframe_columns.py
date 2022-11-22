import inflection
from sklearn.base import BaseEstimator, TransformerMixin


class RenameDataframeColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.rename_columns(X)

        return X

    def rename_columns(self, dataframe):
        # Copy DataFrame
        df = dataframe.copy()

        # Columns Names
        cols_old = list(df.columns)

        # Function to convert columns names to snake_case pattern
        snakecase = lambda x: inflection.underscore(x)

        # New columns names
        cols_new = list(map(snakecase, cols_old))

        # Apply new columns names to DataFrame
        df.columns = cols_new

        return df
