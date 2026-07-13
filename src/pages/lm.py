import json

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

dash.register_page(__name__, path="/linear-models", name="Linear Model EStimation")

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
                            options=["Linear Regression", "Bayesian Linear Regression"],
                            value="Linear Regression",
                            placeholder="Select a Model ...",
                            id="dropdown-model-selector",
                        )
                    ],
                    style={"align": "center"},
                ),
                dbc.Row([dbc.Col([dcc.Graph(id="model-fit-data")], md=12)]),
            ],
            fluid=True,
        ),
    ]
)


@callback(
    Output("model-fit-data", "figure"),
    [Input("global-pca-train", "data"), Input("dropdown-model-selector", "value")],
)
def fit_model(df_json_train, model_type):

    match model_type:
        case "Linear Regression":
            model = LinearRegression()
        case _:
            return go.Figure()

    data_dict = json.loads(df_json_train)

    df = pd.DataFrame(
        data=data_dict["data"], columns=data_dict["columns"], index=data_dict["index"]
    )

    X = df.drop(columns=["target"])
    y = df["target"]

    model.fit(X, y)

    combinations = [
        ("PC1", "PC2"),
        ("PC1", "PC3"),
        ("PC1", "PC4"),
        ("PC2", "PC3"),
        ("PC2", "PC4"),
        ("PC3", "PC4"),
    ]

    fig = make_subplots(rows=2, cols=3, shared_xaxes=False, shared_yaxes=False)

    for i, (x_axis, y_axis) in enumerate(combinations):
        col = i % 3 + 1
        row = i // 3 + 1
        fig.add_trace(
            go.Scattergl(
                x=df[x_axis],
                y=df[y_axis],
                mode="markers",
                marker=dict(size=6, opacity=0.7),
            ),
            col=col,
            row=row,
        )
        fig.update_xaxes(title_text=x_axis, row=row, col=col)
        fig.update_yaxes(title_text=y_axis, row=row, col=col)

    fig.update_layout(title_text="", height=650, showlegend=False)

    return fig
