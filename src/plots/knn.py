from typing import Any, Dict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots


def plot_fit(df: pd.DataFrame, data: Dict[str, Any]):
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
                x=data[x_axis]["x"],
                y=data[x_axis]["y"],
                legendgroup="model",
                showlegend=(i == 0),
                mode="lines",
                name="PDP (Conditional Expectectation of KNN)",
                marker=dict(size=14, opacity=1, color="red"),
            ),
            col=col,
            row=row,
        )
        fig.update_xaxes(title_text=x_axis, row=row, col=col)
        fig.update_yaxes(title_text="MedHouseVal", row=row, col=col, range=[-0.1, 5.1])

    fig.update_layout(
        title_text="Test Data vs Marginal Best Model Fit", height=650, showlegend=True
    )

    return fig


def plot_cv_score(
    score_train: np.ndarray, score_val: np.ndarray, param_grid: Dict[str, Any]
):
    knn_k_str = [str(a) for a in param_grid["n_neighbors"]]
    z_min = min(score_train.min(), score_val.min())
    z_max = max(score_train.max(), score_val.max())

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Train Error CV", "Val Error CV"),
        shared_xaxes=False,
        shared_yaxes=False,
    )

    fig.add_trace(
        go.Heatmap(
            z=score_train.T,
            y=param_grid["metrics"],
            x=knn_k_str,
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=score_val.T, y=param_grid["metrics"], x=knn_k_str, coloraxis="coloraxis"
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Cross Validation Analysis",
        height=650,
        showlegend=True,
        coloraxis=dict(
            colorscale="Viridis", cmin=z_min, cmax=z_max, colorbar_title="RMSE"
        ),
    )

    fig.update_xaxes(title_text="Number of Nearest-Neighbors")
    fig.update_yaxes(title_text="Metrics")

    return fig


def plot_test_predictions(y_true: np.ndarray, y_pred: np.ndarray):

    line_plot = [0, 1, 2, 3, 4, 5]

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
            mode="lines",
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


def plot_tradeoff(train: np.ndarray, val: np.ndarray, param: np.ndarray):

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            y=train,
            x=1 / param,
            mode="markers+lines",
            name="Train",
            marker=dict(size=7, opacity=0.7, symbol="circle"),
        )
    )

    fig.add_trace(
        go.Scattergl(
            y=val,
            x=1 / param,
            mode="markers+lines",
            name="Validation",
            marker=dict(size=7, opacity=0.7, symbol="diamond"),
        )
    )

    fig.update_layout(
        title_text="Model Error vs Model Complexity (Bias-Variance Tradeoff)",
        height=650,
        showlegend=True,
    )

    fig.update_yaxes(title_text="RMSE")
    fig.update_xaxes(
        title_text="Model Complexity (1/k)",
        tickmode="array",
        type="log",
        tickvals=1 / param,
        ticktext=param,
        tickformat=".3f",
    )

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


def plot_importance_permutation(data=Dict[str, Any]):
    features = data["permutation_importance"]["features"]
    means = data["permutation_importance"]["importances_mean"]
    stds = data["permutation_importance"]["importances_std"]

    fig_permutation = go.Figure()

    fig_permutation.add_trace(
        go.Bar(
            y=features,
            x=-means,
            orientation="h",
            name="Marginal Importance",
            marker=dict(color="rgba(50, 171, 96, 0.6)"),
            error_x=dict(
                type="data",
                array=stds,
                visible=True,
                color="black",
                thickness=1.5,
                width=5,
            ),
        )
    )

    fig_permutation.update_layout(
        title="Permutation Importance (delta RMSE)",
        yaxis=dict(autorange="reversed"),
        xaxis_title="Increment on RMSE in Permutation",
        yaxis_title="Principal Component",
    )

    return fig_permutation


def make_knn_regression_layout(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> dbc.Container:
    # X = df_train.drop(columns=["target"]).to_numpy()

    (
        train_errors,
        val_errors,
        grid_params,
        metrics,
        best_params,
        test_preds,
        info,
    ) = fit_cv(df_train=df_train, df_val=df_val, df_test=df_test)

    index_metric = np.where(grid_params["metrics"] == best_params["metrics"])[0]

    fig1 = plot_fit(df=df_test, data=info)
    fig2 = plot_cv_score(
        score_train=train_errors, score_val=val_errors, param_grid=grid_params
    )
    fig3 = plot_test_predictions(y_true=df_test["target"].to_numpy(), y_pred=test_preds)
    fig4 = plot_tradeoff(
        train=train_errors.T[index_metric.item(), :],
        val=val_errors.T[index_metric.item(), :],
        param=grid_params["n_neighbors"],
    )
    fig5 = plot_importance_permutation(data=info)
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
                    dbc.Col(
                        [dbc.Row([table_html]), dbc.Row([dcc.Graph(figure=fig5)])], md=3
                    ),
                ]
            ),
        ],
        fluid=True,
    )
