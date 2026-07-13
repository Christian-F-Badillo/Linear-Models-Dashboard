from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame


def get_pca_dfs(
    df: pd.DataFrame,
    target: str = "MedHouseVal",
    test_prop: float = 0.15,
    val_prop: float = 0.15,
    n_components: int = 4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    X = df.drop(columns=[target])
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_prop, shuffle=True, random_state=random_state
    )

    val_size_adjusted = val_prop / (1.0 - test_prop)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size_adjusted,
        shuffle=True,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    pca_train = pca.fit_transform(X_train_scaled)
    pca_val = pca.transform(X_val_scaled)
    pca_test = pca.transform(X_test_scaled)

    pc_columns = [f"PC{i + 1}" for i in range(n_components)]

    # Train
    pca_df_train = pd.DataFrame(data=pca_train, columns=pc_columns)
    pca_df_train["target"] = y_train.values

    # Val
    pca_df_val = pd.DataFrame(data=pca_val, columns=pc_columns)
    pca_df_val["target"] = y_val.values

    # Test
    pca_df_test = pd.DataFrame(data=pca_test, columns=pc_columns)
    pca_df_test["target"] = y_test.values

    return pca_df_train, pca_df_val, pca_df_test
