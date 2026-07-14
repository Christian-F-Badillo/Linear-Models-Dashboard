from operator import index
from typing import Any, Dict, List, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    # Metric Dict
    metric_dict = {"rmse": {}, "mae": {}}

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

    # Get the optimal parameters
    idx_k_opt, idx_alpha_opt = np.unravel_index(np.argmin(val_errors), val_errors.shape)
    k_opt = degrees_k[idx_k_opt]
    alpha_opt = alphas[idx_alpha_opt]
    best_params = {"k": k_opt, "alpha": alpha_opt}

    # Fit the best model
    model_poly.set_params(poly_feat_mat__kw_args={"degree": k_opt})
    model_poly.set_params(ridge_reg__alpha=alpha_opt)
    model_poly.fit(X_train, y_train)

    # Best Model Evaluation
    y_pred_val = model_poly.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae_val = mean_absolute_error(y_val, y_pred_val)

    # Test the Best Model against the Test Set
    y_pred_test = model_poly.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Store the Metrics of the best Model
    metric_dict["mae"]["val"] = mae_val
    metric_dict["mae"]["test"] = mae_test
    metric_dict["rmse"]["val"] = rmse_val
    metric_dict["rmse"]["test"] = rmse_test

    return (
        train_errors,
        val_errors,
        coefs,
        grid_param,
        metric_dict,
        best_params,
        y_pred_test,
    )


def plot_fit(
    df: pd.DataFrame, x_support: List[Any], coefs: np.ndarray, hyper_params: dict
):
    y = df["target"].to_numpy()

    K = hyper_params["k"]
    intercept = coefs[0]

    betas_full = coefs[1:]

    combinations = ["PC1", "PC2", "PC3", "PC4"]

    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False)

    for i, x_axis in enumerate(combinations):
        col = i % 2 + 1
        row = i // 2 + 1

        X_poly_local = np.vstack([x_support[i] ** d for d in range(1, K + 1)]).T

        betas_pc = betas_full[i::4][:K]

        preds = X_poly_local @ betas_pc + intercept

        fig.add_trace(
            go.Scattergl(
                x=df[x_axis],
                y=y,
                mode="markers",
                legendgroup="data",
                showlegend=True,
                name=f"PC{i + 1} projection data",
                marker=dict(size=6, opacity=0.7),
            ),
            col=col,
            row=row,
        )
        fig.add_trace(
            go.Scattergl(
                x=x_support[i],
                y=preds,
                legendgroup="model",
                showlegend=(i == 0),
                name="Best Model Fit",
                marker=dict(size=14, opacity=1, color="red"),
            ),
            col=col,
            row=row,
        )
        fig.update_xaxes(title_text=x_axis, row=row, col=col)
        fig.update_yaxes(title_text="MedHouseVal", row=row, col=col, range=[-0.1, 5.1])

    fig.update_layout(
        title_text="Test Data vs Best Model Fit", height=650, showlegend=True
    )

    return fig


def plot_cv_score(
    score_train: np.ndarray, score_val: np.ndarray, param_grid: Dict[str, Any]
):
    alphas_str = [f"{a:.2e}" for a in param_grid["alpha"]]
    k_str = [f"{k}" for k in param_grid["k"]]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Train Error CV", "Val Error CV"),
        shared_xaxes=True,
        shared_yaxes=True,
    )

    fig.add_trace(
        go.Heatmap(
            z=score_train,
            y=k_str,
            x=alphas_str,
            colorscale="Viridis",
            colorbar=dict(title="RMSE"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=score_val,
            y=k_str,
            x=alphas_str,
            colorscale="Viridis",
            colorbar=dict(title="RMSE"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Cross Validation Analysis",
        height=650,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Alpha")
    fig.update_yaxes(title_text="Degree")

    return fig


def plot_test_predictions(y_true: np.ndarray, y_pred: np.ndarray):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=y_true,
            mode="markers",
            name="",
            marker=dict(size=7, opacity=0.7, symbol="diamond"),
        )
    )

    fig.update_layout(
        title_text="Test vs Model Prediction",
        height=650,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Model Prediction")
    fig.update_yaxes(title_text="MedHouseVal")

    return fig


def plot_coef_cv(coefs: np.ndarray, alphas: np.ndarray, best_k: int):

    num_traces, num_vars = coefs.shape
    coefs = coefs.T

    fig = go.Figure()

    for i in range(1, num_vars):
        fig.add_trace(
            go.Scatter(
                y=coefs[i,],
                x=alphas,
                mode="markers+lines",
                name=f"Coefficient {i}",
                marker=dict(size=7, opacity=0.7, symbol="circle"),
            )
        )

    fig.update_layout(
        title_text=f"Coefficients vs Alpha (Degree {best_k})",
        height=650,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Alpha")
    fig.update_yaxes(title_text="Value")

    return fig


def make_metrics_table(metrics: Dict[str, Dict[str, float]]):

    rmse_val, mae_val = (
        metrics["rmse"]["val"],
        metrics["mae"]["val"],
    )
    rmse_test, mae_test = (
        metrics["rmse"]["test"],
        metrics["mae"]["test"],
    )

    header = html.Thead(
        html.Tr(
            [
                html.Th("Metric", className="text-start"),
                html.Th("Validation Set (CV)", className="text-center"),
                html.Th("Test Set (Generalization)", className="text-center"),
            ],
            className="table-dark",
        )
    )

    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(html.B("RMSE"), className="text-start"),
                    html.Td(f"{rmse_val:.4f}", className="text-center text-muted"),
                    html.Td(html.B(f"{rmse_test:.4f}"), className="text-center"),
                ]
            ),
            html.Tr(
                [
                    html.Td(html.B("MAE"), className="text-start"),
                    html.Td(f"{mae_val:.4f}", className="text-center text-muted"),
                    html.Td(html.B(f"{mae_test:.4f}"), className="text-center"),
                ]
            ),
        ]
    )

    return dbc.Table(
        [header, body],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        size="sm",
        className="shadow-sm mt-3",
    )


def make_polynomial_regression_layout(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> dbc.Container:
    X = df_train.drop(columns=["target"]).to_numpy()

    train_errors, val_errors, coefs, grid_params, metrics, best_params, test_preds = (
        fit_poly_ridge_cv(df_train=df_train, df_val=df_val, df_test=df_test)
    )

    index_k = np.where(grid_params["k"] == best_params["k"])[0]
    index_alpha = np.where(grid_params["alpha"] == best_params["alpha"])[0]

    x_support = [np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100) for i in range(4)]
    coefs_model = coefs[index_k.item(), index_alpha.item(), :]

    fig1 = plot_fit(
        df_test, x_support=x_support, coefs=coefs_model, hyper_params=best_params
    )
    fig2 = plot_cv_score(
        score_train=train_errors, score_val=val_errors, param_grid=grid_params
    )
    fig3 = plot_test_predictions(y_true=df_test["target"].to_numpy(), y_pred=test_preds)
    fig4 = plot_coef_cv(
        coefs=coefs[index_k.item(), :, :],
        alphas=grid_params["alpha"],
        best_k=index_k.item() + 1,
    )
    table_html = make_metrics_table(metrics=metrics)

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(figure=fig1)], md=6),
                    dbc.Col([dcc.Graph(figure=fig2)], md=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(figure=fig4)], md=5),
                    dbc.Col([dcc.Graph(figure=fig3)], md=4),
                    dbc.Col([table_html], md=3),
                ]
            ),
        ],
        fluid=True,
    )
