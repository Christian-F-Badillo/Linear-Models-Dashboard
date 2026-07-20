from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src.plots.knn import make_knn_regression_layout
from src.plots.linear import make_linear_regression_layout
from src.plots.poly import make_polynomial_regression_layout

DATA_DIR = Path("data/raw")

df_train = pd.read_parquet(DATA_DIR / "df_pca_train.parquet")
df_val = pd.read_parquet(DATA_DIR / "df_pca_val.parquet")
df_test = pd.read_parquet(DATA_DIR / "df_pca_test.parquet")

dash.register_page(__name__, path="/linear-models", name="Linear Model Estimation")

layout = html.Div(
    [
        html.H1(
            "Linear Model Estimation",
            style={"text-weight": "bold", "text-align": "center"},
            className="text-2xl font-bold mb-4",
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dcc.Dropdown(
                            options=[
                                {"label": "Linear Regression (Ridge)", "value": 0},
                                {"label": "K Nearest-Neighbor Regression", "value": 1},
                                {"label": "Polynomial Regression (GAM)", "value": 2},
                            ],
                            value=0,
                            placeholder="Select a Model ...",
                            id="dropdown-model-selector",
                        )
                    ],
                    style={"align": "center"},
                ),
            ],
            fluid=True,
        ),
        html.Div(id="layout-model-fit"),
    ]
)


@callback(
    Output("layout-model-fit", "children"),
    [
        Input("dropdown-model-selector", "value"),
    ],
)
def fit_model(model_type):

    match model_type:
        case 0:
            return make_linear_regression_layout(df_train, df_val, df_test)
        case 1:
            return make_knn_regression_layout(df_train, df_val, df_test)
        case 2:
            return make_polynomial_regression_layout(df_train, df_val, df_test)
        case _:
            return dbc.Container([html.P("")])
