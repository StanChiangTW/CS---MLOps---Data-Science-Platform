import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
    """function to load a CSV file and give a data frame"""
    data = pd.read_csv(file_path)
    return data
