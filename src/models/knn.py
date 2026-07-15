from typing import Any, Dict, List, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def fit_cv(
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
    metric_dict = {"rmse": {}, "mae": {}}

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

    # Best Params
    idx_k_opt, idx_m_opt = np.unravel_index(np.argmin(val_errors), val_errors.shape)
    best_params = {
        "n_neighbors": num_neighbors[idx_k_opt],
        "metrics": dist_metrics[idx_m_opt],
    }

    # Best Model
    best_knn = KNeighborsRegressor(
        n_neighbors=best_params["n_neighbors"], metric=best_params["metrics"]
    )
    best_knn.fit(X_train, y_train)

    # Predict
    y_pred_val = best_knn.predict(X_val)
    y_pred_test = best_knn.predict(X_test)

    # Metrics
    metric_dict["rmse"]["val"] = np.sqrt(mean_squared_error(y_val, y_pred_val))
    metric_dict["rmse"]["test"] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    metric_dict["mae"]["val"] = mean_absolute_error(y_val, y_pred_val)
    metric_dict["mae"]["test"] = mean_absolute_error(y_test, y_pred_test)

    # Plot info
    info_plots = {}

    # PDP 1D
    for i in range(X_test.shape[1]):
        pd_results = partial_dependence(
            estimator=best_knn,
            X=X_test,
            features=[i],
            grid_resolution=25,
            percentiles=(0.01, 0.99),
        )
        info_plots[f"PC{i + 1}"] = {
            "x": pd_results["grid_values"][0],
            "y": pd_results["average"][0],
        }

    # Permutation Importance
    vi_results = permutation_importance(
        best_knn,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    # Shift sign to better interpretation
    info_plots["permutation_importance"] = {
        "features": feature_cols,
        "importances_mean": vi_results.importances_mean * -1,
        "importances_std": vi_results.importances_std,
    }

    return (
        train_errors,
        val_errors,
        grid_param,
        metric_dict,
        best_params,
        y_pred_test,
        info_plots,
    )


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
