from typing import Any, Dict, List, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots


def plot_fit(df: pd.DataFrame, x_support: List[Any], preds: np.ndarray):
    y = df["target"].to_numpy()

    combinations = ["PC1", "PC2", "PC3", "PC4"]

    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False)

    for i, x_axis in enumerate(combinations):
        col = i % 2 + 1
        row = i // 2 + 1

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
                y=preds[i],
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

    line_plot = np.arange(0, 5, 0.5)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=y_pred,
            y=y_true,
            mode="markers",
            name="",
            showlegend=False,
            marker=dict(size=7, opacity=0.7, symbol="diamond"),
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=line_plot,
            y=line_plot,
            mode="markers+lines",
            name="Ideal Fit",
            showlegend=True,
            marker=dict(size=7, opacity=0.7, symbol="diamond"),
        )
    )

    fig.update_layout(
        title_text="Test vs Model Prediction",
        height=650,
        showlegend=True,
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

    train_errors, val_errors, coefs, grid_params, metrics, best_params, test_preds = (
        fit_poly_ridge_cv(df_train=df_train, df_val=df_val, df_test=df_test)
    )

    index_k = np.where(grid_params["k"] == best_params["k"])[1]

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
