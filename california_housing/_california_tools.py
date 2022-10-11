__all__ = [
    'CombinedAttributesAdder',
    'display_scores',
    'load_data',
]

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

_CSV_NAME = 'housing.csv'
_ROOMS_INDEX, _BEDROOMS_INDEX, _POPULATION_INDEX, _HOUSEHOLDS_INDEX = 3, 4, 5, 6


def load_data(base_path: str, csv_name: str = _CSV_NAME) -> pd.DataFrame:
    path = os.path.join(base_path, csv_name)
    return pd.read_csv(path)


def display_scores(scores: np.ndarray) -> None:
    print(f'Scores: {scores}\nMean: {scores.mean()}\nSTD: {scores.std()}')


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room: bool = True) -> None:
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, _: np.ndarray, __: Optional[np.ndarray] = None) -> 'CombinedAttributesAdder':
        return self
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        rooms_per_household = x[:, _ROOMS_INDEX] / x[:, _HOUSEHOLDS_INDEX]
        population_per_household = x[:, _POPULATION_INDEX] / x[:, _HOUSEHOLDS_INDEX]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, _BEDROOMS_INDEX] / x[:, _ROOMS_INDEX]
            return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
        
        return np.c_[x, rooms_per_household, population_per_household]
