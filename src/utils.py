import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_dataset() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame
