from typing import Any, Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots


def plot_fit(df: pd.DataFrame, x_support: List[Any], preds: List[Any]):
    y = df["target"].to_numpy()

    combinations = [
        "PC1",
        "PC2",
        "PC3",
        "PC4",
    ]

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
            go.Scatter(
                x=x_support[i],
                y=preds[i],
                legendgroup="model",
                showlegend=not bool(i + 1 // 4),
                name="Best Model Fit",
                marker=dict(size=14, opacity=1, color="red"),
            ),
            col=col,
            row=row,
        )
        fig.update_xaxes(title_text=x_axis, row=row, col=col)
        fig.update_yaxes(title_text="MedHouseVal", row=row, col=col)

    fig.update_layout(
        title_text="Test Data vs Best Model Fit", height=650, showlegend=True
    )

    return fig


def plot_cv_score(
    score_train: np.ndarray, score_val: np.ndarray, alphas: np.ndarray, score: str
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=alphas,
            y=score_train,
            mode="markers+lines",
            name=f"{score} Train",
            marker=dict(size=8, opacity=1, symbol="x"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=alphas,
            y=score_val,
            mode="markers+lines",
            name=f"{score} Validation",
            marker=dict(size=8, opacity=1, symbol="diamond"),
        )
    )

    fig.update_layout(
        title_text="Cross Validation Analysis",
        height=450,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Alpha")
    fig.update_yaxes(title_text=f"{score}")

    return fig


def plot_test_predictions(df_test: pd.DataFrame, coefs: np.ndarray):
    X = df_test.drop(columns=["target"]).to_numpy()
    y_true = df_test["target"]
    preds = coefs[0] + (X @ coefs[1:])

    line_plot = np.arange(0, 5, 0.5)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preds,
            y=y_true,
            mode="markers",
            name="",
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
        height=450,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Model Prediction")
    fig.update_yaxes(title_text="MedHouseVal")

    return fig


def plot_coef_cv(coefs: np.ndarray, alphas: np.ndarray):

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
        title_text="Parameters vs Alpha",
        height=650,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Alpha")
    fig.update_yaxes(title_text="Parameter Value")

    return fig


def make_metrics_table(metrics: Dict[str, Dict[str, float]]):

    r2_val, rmse_val, mae_val = (
        metrics["r2"]["val"],
        metrics["rmse"]["val"],
        metrics["mae"]["val"],
    )
    r2_test, rmse_test, mae_test = (
        metrics["r2"]["test"],
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
                    html.Td(html.B("R²"), className="text-start"),
                    html.Td(f"{r2_val:.4f}", className="text-center text-muted"),
                    html.Td(
                        html.B(f"{r2_test:.4f}"), className="text-center"
                    ),  # Negrita para resaltar el test real
                ]
            ),
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


def make_linear_regression_layout(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> dbc.Container:
    X = df_train.drop(columns=["target"]).to_numpy()

    train_errors, val_errors, coef_by_alpha, alphas, metrics = fit_ridge_cv(
        df_train=df_train, df_val=df_val, df_test=df_test
    )

    best_fit = np.argmin(val_errors)

    x_support = [np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100) for i in range(4)]
    preds = [
        coef_by_alpha[best_fit, 0] + coef_by_alpha[best_fit, i + 1] * x_support[i]
        for i in range(4)
    ]

    fig1 = plot_fit(df_test, x_support=x_support, preds=preds)
    fig2 = plot_cv_score(
        score_train=train_errors, score_val=val_errors, alphas=alphas, score="RMSE"
    )
    fig3 = plot_test_predictions(df_test=df_test, coefs=coef_by_alpha[best_fit, :])
    fig4 = plot_coef_cv(coefs=coef_by_alpha, alphas=alphas)
    table_html = make_metrics_table(metrics=metrics)

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(figure=fig1)], md=6),
                    dbc.Col([dcc.Graph(figure=fig4)], md=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(figure=fig2)], md=4),
                    dbc.Col([dcc.Graph(figure=fig3)], md=4),
                    dbc.Col([table_html], md=4),
                ]
            ),
        ],
        fluid=True,
    )
