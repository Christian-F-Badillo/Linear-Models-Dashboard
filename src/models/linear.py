from typing import Any, List, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc
from dash.html import Data
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def fit_ridge_cv(df_train: pd.DataFrame, df_val: np.ndarray) -> Tuple[List[Any]]:
    feature_cols = [c for c in df_train.columns if c != "target"]

    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values

    X_val = df_val[feature_cols].values
    y_val = df_val["target"].values

    alphas = np.logspace(-3, 3, 20)
    train_errors = np.zeros(shape=(alphas.size,))
    val_errors = np.zeros(shape=(alphas.size,))
    coefs_by_alpha = np.zeros(shape=((alphas.size, X_train.shape[1] + 1)))

    for i, alpha in enumerate(alphas):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

        train_errors[i] = train_rmse
        val_errors[i] = val_rmse
        coefs_by_alpha[i, 0] = model.intercept_
        coefs_by_alpha[i, 1:] = model.coef_

    return train_errors, val_errors, coefs_by_alpha, alphas


def plot_fit_train(df: pd.DataFrame, x_support: List[Any], preds: List[Any]):
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

    fig.update_layout(title_text="Training Model Fit", height=650, showlegend=True)

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
    fig.update_yaxes(title_text=f"{score}", range=[0.5, 1])

    return fig


def plot_test_predictions(df_test: pd.DataFrame, coefs: np.ndarray):
    X = df_test.drop(columns=["target"]).to_numpy()
    y_true = df_test["target"]
    preds = coefs[0] + (X @ coefs[1:])

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

    fig.update_layout(
        title_text="Test vs Model Prediction",
        height=450,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Model Prediction")
    fig.update_yaxes(title_text="MedHouseVal")

    return fig


def make_linear_regression_layout(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> dbc.Container:
    X = df_train.drop(columns=["target"]).to_numpy()

    train_errors, val_errors, coef_by_alpha, alphas = fit_ridge_cv(
        df_train=df_train, df_val=df_val
    )

    best_fit = np.argmin(val_errors)

    x_support = [np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100) for i in range(4)]
    preds = [
        coef_by_alpha[best_fit, 0] + coef_by_alpha[best_fit, i + 1] * x_support[i]
        for i in range(4)
    ]

    fig1 = plot_fit_train(df_train, x_support=x_support, preds=preds)
    fig2 = plot_cv_score(
        score_train=train_errors, score_val=val_errors, alphas=alphas, score="RMSE"
    )
    fig3 = plot_test_predictions(df_test=df_test, coefs=coef_by_alpha[best_fit, :])

    return dbc.Container(
        [
            dbc.Row([dcc.Graph(figure=fig1)]),
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(figure=fig2)], md=4),
                    dbc.Col([dcc.Graph(figure=fig3)], md=4),
                ]
            ),
        ],
        fluid=True,
    )
