from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def fit_knn_cv(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[Any, ...]:

    feature_cols = [c for c in df_train.columns if c != "target"]
    X_train, y_train = df_train[feature_cols].values, df_train["target"].values
    X_val, y_val = df_val[feature_cols].values, df_val["target"].values
    X_test, y_test = df_test[feature_cols].values, df_test["target"].values

    # Param Grid
    num_neighbors = np.unique(np.geomspace(2, 500, num=20, dtype=int))
    dist_metrics = np.array(["manhattan", "euclidean", "chebyshev"])
    grid_param = {"n_neighbors": num_neighbors, "metrics": dist_metrics}

    # Metrics
    train_errors = np.zeros((len(num_neighbors), len(dist_metrics)))
    val_errors = np.zeros((len(num_neighbors), len(dist_metrics)))

    # CV
    for j, metric in enumerate(dist_metrics):
        for i, k in enumerate(num_neighbors):
            model_knn = KNeighborsRegressor(
                n_neighbors=k, metric=metric, algorithm="auto"
            )
            model_knn.fit(X_train, y_train)

            train_errors[i, j] = np.sqrt(
                mean_squared_error(y_train, model_knn.predict(X_train))
            )
            val_errors[i, j] = np.sqrt(
                mean_squared_error(y_val, model_knn.predict(X_val))
            )

    # Best params
    best_params = get_best_params(grid_param, val_errors)

    # Best Model
    best_knn = KNeighborsRegressor(
        n_neighbors=best_params["n_neighbors"], metric=best_params["metrics"]
    )
    best_knn.fit(X_train, y_train)

    # Preds
    y_pred_test = best_knn.predict(X_test)

    # Metrics
    metric_dict = evaluate_model(best_knn, X_val, y_val, X_test, y_test)

    info_plots = gen_plot_info(best_knn, X_test, X_val, y_val, feature_cols)

    coefs = None

    return (
        train_errors,
        val_errors,
        coefs,
        grid_param,
        metric_dict,
        best_params,
        y_pred_test,
        info_plots,
    )


def get_best_params(grid_params: Dict[str, Any], errors: np.ndarray) -> Dict[str, Any]:

    idx_k_opt, idx_m_opt = np.unravel_index(np.argmin(errors), errors.shape)
    best_params = {
        "n_neighbors": grid_params["n_neighbors"][idx_k_opt],
        "metrics": grid_params["metrics"][idx_m_opt],
    }

    return best_params


def gen_plot_info(model, X_test: np.ndarray, X_val, y_val, features):
    # Plot info
    info_plots = {}

    # PDP 1D
    for i in range(X_test.shape[1]):
        pd_results = partial_dependence(
            estimator=model,
            X=X_test,
            features=[i],
            grid_resolution=25,
            percentiles=(0, 1),
        )
        info_plots[f"PC{i + 1}"] = {
            "x": pd_results["grid_values"][0],
            "y": pd_results["average"][0],
        }

    # Permutation Importance
    vi_results = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    # Shift sign to better interpretation
    info_plots["permutation_importance"] = {
        "features": features,
        "importances_mean": vi_results.importances_mean * -1,
        "importances_std": vi_results.importances_std,
    }

    return info_plots


def evaluate_model(
    model, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    metric_dict = {"rmse": {}, "mae": {}}

    # Predict
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Metrics
    metric_dict["rmse"]["val"] = np.sqrt(mean_squared_error(y_val, y_pred_val))
    metric_dict["rmse"]["test"] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    metric_dict["mae"]["val"] = mean_absolute_error(y_val, y_pred_val)
    metric_dict["mae"]["test"] = mean_absolute_error(y_test, y_pred_test)

    return metric_dict
