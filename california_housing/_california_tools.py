__all__ = [
    'load_data',
]

import os

import pandas as pd

_CSV_NAME = 'housing.csv'


def load_data(base_path: str, csv_name: str = _CSV_NAME) -> pd.DataFrame:
    path = os.path.join(base_path, csv_name)
    return pd.read_csv(path)
