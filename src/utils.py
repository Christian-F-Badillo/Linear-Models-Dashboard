import os
from pathlib import Path

import joblib


def load_precomputed_model(model_name: str, base_path: str):

    base_path = Path(base_path)

    if not os.path.exists(base_path):
        raise FileExistsError(f"{base_path} does not exists.")

    file_path = base_path / model_name / "data.joblib.lzma"

    if not file_path.exists():
        raise FileNotFoundError(f"No model fit data found for model {model_name}")

    return joblib.load(file_path)
