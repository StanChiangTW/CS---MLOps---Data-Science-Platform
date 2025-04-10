import logging
import pandas as pd
from sklearn.base import ClassifierMixin


def predict(
    model: ClassifierMixin, X_test: pd.DataFrame
) -> np.ndarray:
    return model.predict(X_test)
