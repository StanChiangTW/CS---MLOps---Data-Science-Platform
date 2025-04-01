import logging
import pandas as pd
from sklearn.base import ClassifierMixin
from dsba.preprocessing import preprocess_dataframe


def predict(
    model: ClassifierMixin, X_test: pd.DataFrame
) -> pd.Series:
    return model.predict(X_test)
