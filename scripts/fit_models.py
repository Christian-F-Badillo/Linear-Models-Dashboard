# This script fits the models for the dashboard and saves the results.
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.knn import fit_knn_cv
from src.models.linear import fit_lr_ridge_cv
from src.models.poly import fit_poly_ridge_cv
from src.utils import get_pca_dfs, load_dataset, remove_outliers

# Carga y procesamiento de datos
df = load_dataset()
df = remove_outliers(df)
df_pca_train, df_pca_val, df_pca_test = get_pca_dfs(df)

# Definición estricta de rutas y modelos
DATA_PATH = Path("./data/model_data/")
MODELS = {"linear": fit_lr_ridge_cv, "poly": fit_poly_ridge_cv, "knn": fit_knn_cv}


def cast_to_float32(data):
    if isinstance(data, dict):
        return {k: cast_to_float32(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [cast_to_float32(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(cast_to_float32(v) for v in data)
    elif isinstance(data, np.ndarray) and data.dtype == np.float64:
        return data.astype(np.float32)
    return data


def precompute_and_save(model_name: str, results: tuple, base_path: Path):
    target_dir = base_path / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    optimized_results = cast_to_float32(results)

    file_path = target_dir / "data.joblib.lzma"
    joblib.dump(optimized_results, file_path, compress=("lzma", 3))
    print(f"[{model_name.upper()}] Data saved at {file_path}")


def main(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    force_compute: bool = False,
):
    """
    Pipeline for Models Fits

    Args:
        train, val, test: DataFrames of Data PCA.
        force_compute: If True recompute the fits.
    """
    for model_name, fit_func in MODELS.items():
        expected_file = DATA_PATH / model_name / "data.joblib.lzma"

        # Comprobación de estado persistente
        if expected_file.exists() and not force_compute:
            print(
                f"[{model_name.upper()}] Model is fitted, ignorig it. If you want to force the fit set the flag --force."
            )
            continue

        print(f"[{model_name.upper()}] Fitting the model ...")
        results = fit_func(train, val, test)
        precompute_and_save(model_name=model_name, results=results, base_path=DATA_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute the Models Fits for the dashboard."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force the fit of previously fitted models.",
    )
    args = parser.parse_args()

    main(df_pca_train, df_pca_val, df_pca_test, force_compute=args.force)
