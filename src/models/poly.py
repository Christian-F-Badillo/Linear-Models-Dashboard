from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def polynomial_no_interactions(X, degree=2):
    """
    Generate a Polynomial feature matrix with no interaction terms for GAM models.
    X: array-like of shape (n_samples, n_features)
    Return: array (n_samples, n_features * degree)
    """
    return np.hstack([X**d for d in range(1, degree + 1)])


def fit_poly_ridge_cv(
    df_train: pd.DataFrame, df_val: np.ndarray, df_test: pd.DataFrame
) -> Tuple[List[Any]]:

    # Data preparation for Model Fit
    feature_cols = [c for c in df_train.columns if c != "target"]
    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["target"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["target"].values

    # Parameter Grid
    alphas = np.logspace(-3, 3, 20)
    degrees_k = np.array([1, 2, 3, 4])
    grid_param = {"alpha": alphas, "k": degrees_k}

    # Evaluation Arrays Initialization
    train_errors = np.zeros(shape=(degrees_k.size, alphas.size))
    val_errors = np.zeros(shape=(degrees_k.size, alphas.size))

    # Parameter Tensor
    coefs = np.full((degrees_k.size, alphas.size, 17), np.nan)

    # Feature Matrix Creation
    poly_transf = FunctionTransformer(
        func=polynomial_no_interactions, kw_args={"degree": 2}, validate=True
    )

    # Model Definition with Pipelines
    model_poly = Pipeline(
        [
            ("poly_feat_mat", poly_transf),
            ("scaler", StandardScaler()),
            ("ridge_reg", Ridge()),
        ]
    )

    for i, k in enumerate(degrees_k):
        # Set Parameter for CV
        model_poly.set_params(poly_feat_mat__kw_args={"degree": k})

        for j, alpha in enumerate(alphas):
            # Set parameter for CV
            model_poly.set_params(ridge_reg__alpha=alpha)

            # Model Fit
            model_poly.fit(X_train, y_train)

            # Model Evaluation
            train_rmse = np.sqrt(
                mean_squared_error(y_train, model_poly.predict(X_train))
            )
            val_rmse = np.sqrt(mean_squared_error(y_val, model_poly.predict(X_val)))

            # Error Storing
            train_errors[i, j] = train_rmse
            val_errors[i, j] = val_rmse

            # Parameter Extraction
            ridge_model = model_poly.named_steps["ridge_reg"]
            coefs[i, j, 0] = ridge_model.intercept_

            num_coefs = k * len(feature_cols)
            coefs[i, j, 1 : (num_coefs + 1)] = ridge_model.coef_

    best_params = get_best_params(grid_param, val_errors)

    # Train the best model
    model_poly.set_params(poly_feat_mat__kw_args={"degree": best_params["k"]})
    model_poly.set_params(ridge_reg__alpha=best_params["alpha"])
    model_poly.fit(X_train, y_train)

    # Get model predictions
    y_pred_test = model_poly.predict(X_test)

    # Evaluate the best model
    metric_dict = evaluate_model(model_poly, X_val, y_val, X_test, y_test)

    # Get extra plotting info
    info_plots = gen_plot_info(X_train, best_params, grid_param, coefs)

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

    idx_k_opt, idx_alpha_opt = np.unravel_index(np.argmin(errors), errors.shape)
    k_opt = grid_params["k"][idx_k_opt]
    alpha_opt = grid_params["alpha"][idx_alpha_opt]
    best_params = {"k": k_opt, "alpha": alpha_opt}

    return best_params


def gen_plot_info(
    X_train: np.ndarray, best_params: dict, grid_param: dict, coefs: np.ndarray
):
    x_support = [
        np.linspace(np.min(X_train[:, i]), np.max(X_train[:, i]), 100) for i in range(4)
    ]
    K = best_params["k"]

    index_k = np.where(grid_param["k"] == best_params["k"])[0]
    index_alpha = np.where(grid_param["alpha"] == best_params["alpha"])[0]
    coefs_model = coefs[index_k.item(), index_alpha.item(), :]

    intercept = coefs_model[0]
    betas_full = coefs_model[1:]

    info_plots = {"pc_preds": []}

    for i in range(4):
        X_poly_local = np.vstack([x_support[i] ** d for d in range(1, K + 1)]).T
        betas_pc = betas_full[i::4][:K]
        preds: np.ndarray = X_poly_local @ betas_pc + intercept
        info_plots["pc_preds"].append(preds)

    return info_plots


def evaluate_model(
    model, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    metric_dict = {"rmse": {}, "mae": {}}

    # Best Model Evaluation
    y_pred_val = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae_val = mean_absolute_error(y_val, y_pred_val)

    # Test the Best Model against the Test Set
    y_pred_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Store the Metrics of the best Model
    metric_dict["mae"]["val"] = mae_val
    metric_dict["mae"]["test"] = mae_test
    metric_dict["rmse"]["val"] = rmse_val
    metric_dict["rmse"]["test"] = rmse_test

    return metric_dict
