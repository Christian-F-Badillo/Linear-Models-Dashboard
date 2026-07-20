# This script fits the models for the dashboard and saves the results.
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.knn import fit_knn_cv
from src.models.linear import fit_lr_ridge_cv
from src.models.poly import fit_poly_ridge_cv

from .utils import get_pca_dfs, load_dataset, remove_outliers

# Definición estricta de rutas y modelos
DATA_PATH = Path("./data/model_data/")
RAW_DATA_PATH = Path("./data/raw/")

MODELS = {
    "linear": fit_lr_ridge_cv,
    "poly": fit_poly_ridge_cv,
    "knn": fit_knn_cv,
}
VALID_MODELS = list(MODELS.keys())


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


def get_or_create_data(force: bool = False):
    """
    Gestiona el ciclo de vida de los datos. Si existen y no se fuerza su recreación,
    los lee de disco. De lo contrario, ejecuta el pipeline PCA y los guarda.
    """
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DATA_PATH / "raw_data.parquet"
    train_path = RAW_DATA_PATH / "df_pca_train.parquet"
    val_path = RAW_DATA_PATH / "df_pca_val.parquet"
    test_path = RAW_DATA_PATH / "df_pca_test.parquet"

    # Lectura de caché estático
    if (
        not force
        and train_path.exists()
        and val_path.exists()
        and test_path.exists()
        and raw_path.exists()
    ):
        print("[DATA] Loading preprocessed Parquet data from disk...")
        return (
            pd.read_parquet(train_path),
            pd.read_parquet(val_path),
            pd.read_parquet(test_path),
        )

    # Cómputo de tensores y almacenamiento
    print("[DATA] Computing PCA transformations and serializing to Parquet...")
    df = load_dataset()
    df = remove_outliers(df)
    train, val, test = get_pca_dfs(df)

    df.to_parquet(raw_path)
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    test.to_parquet(test_path)

    return train, val, test


def main(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    force_compute: bool = False,
    target_model: str = None,
):
    """
    Pipeline for Models Fits

    Args:
        train, val, test: DataFrames of Data PCA.
        force_compute: If True recompute the fits.
        target_model: Optional specific model key to execute exclusively.
    """
    # Filtro topológico: Ejecutar todo el diccionario o aislar una llave
    models_to_run = {target_model: MODELS[target_model]} if target_model else MODELS

    for model_name, fit_func in models_to_run.items():
        expected_file = DATA_PATH / model_name / "data.joblib.lzma"

        # Comprobación de estado persistente
        if expected_file.exists() and not force_compute:
            print(
                f"[{model_name.upper()}] Model is fitted, ignoring it. "
                "If you want to force the fit set the flag --force."
            )
            continue

        print(f"[{model_name.upper()}] Fitting the model ...")
        results = fit_func(train, val, test)
        precompute_and_save(model_name=model_name, results=results, base_path=DATA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute the Models Fits and PCA transformations for the dashboard."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force the fit of previously fitted models and data regeneration.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=VALID_MODELS,
        help=f"Execute a single specific model. Valid options: {VALID_MODELS}",
    )
    args = parser.parse_args()

    # 1. Resolución de tensores (Caché vs Cómputo)
    df_pca_train, df_pca_val, df_pca_test = get_or_create_data(force=args.force)

    # 2. Resolución de ajuste algorítmico
    main(
        train=df_pca_train,
        val=df_pca_val,
        test=df_pca_test,
        force_compute=args.force,
        target_model=args.model,
    )
