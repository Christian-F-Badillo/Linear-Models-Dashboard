from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def fit_lr_ridge_cv(
    df_train: pd.DataFrame, df_val: np.ndarray, df_test: pd.DataFrame
) -> Tuple[List[Any]]:

    feature_cols = [c for c in df_train.columns if c != "target"]

    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values

    X_val = df_val[feature_cols].values
    y_val = df_val["target"].values

    X_test = df_test[feature_cols].values
    y_test = df_test["target"].values

    alphas = np.logspace(-3, 3, 20)
    train_errors = np.zeros(shape=(alphas.size,))
    val_errors = np.zeros(shape=(alphas.size,))
    coefs = np.zeros(shape=((alphas.size, X_train.shape[1] + 1)))
    grid_param = {"alpha": alphas}

    for i, alpha in enumerate(alphas):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

        train_errors[i] = train_rmse
        val_errors[i] = val_rmse
        coefs[i, 0] = model.intercept_
        coefs[i, 1:] = model.coef_

    # Best Params
    best_params = get_best_params(grid_param, val_errors)

    best_model = Ridge(alpha=best_params["alpha"])
    best_model.fit(X_train, y_train)

    y_pred_test = best_model.predict(X_test)
    metric_dict = evaluate_model(best_model, X_val, y_val, X_test, y_test)

    info_plots = gen_plot_info()

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

    best_alpha_arg = np.argmin(errors)
    best_params = {"alpha": grid_params["alpha"][best_alpha_arg]}

    return best_params


def gen_plot_info():

    info_plots = {}

    return info_plots


def evaluate_model(
    model, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:

    metric_dict = {"r2": {}, "rmse": {}, "mae": {}}

    y_pred_val = model.predict(X_val)
    r2_val = r2_score(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae_val = mean_absolute_error(y_val, y_pred_val)

    y_pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    metric_dict["r2"]["val"] = r2_val
    metric_dict["r2"]["test"] = r2_test
    metric_dict["mae"]["val"] = mae_val
    metric_dict["mae"]["test"] = mae_test
    metric_dict["rmse"]["val"] = rmse_val
    metric_dict["rmse"]["test"] = rmse_test

    return metric_dict
