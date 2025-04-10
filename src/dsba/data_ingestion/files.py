from io import StringIO
from pathlib import Path
from typing import Union, Optional 
import requests
import pandas as pd


def load_csv_from_path(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a CSV file on the local filesystem into a pandas DataFrame
    Since it loads it all in memory, it is only suitable for datasets small enough to fit in memory
    """
    return pd.read_csv(filepath)


def load_csv_from_url(url: str) -> pd.DataFrame:
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def write_csv_to_path(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    df.to_csv(filepath, index=False)
