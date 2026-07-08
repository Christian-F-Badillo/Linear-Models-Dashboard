import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

dash.register_page(__name__, path="/eda", name="Exploratory Data Analysis (EDA)")

layout = html.Div(
    [
        html.H2("Exploratory Data Analysis", className="text-2xl font-bold mb-4"),
        html.Div(id="table-eda-container"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="plot-pairplot",
                                    responsive=True,
                                    style={"width": "100%", "height": "650px"},
                                )
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="plot-corr-matrix",
                                    responsive=True,
                                    style={"width": "100%", "height": "650px"},
                                )
                            ],
                            md=6,
                        ),
                    ]
                )
            ],
            fluid=True,
        ),
    ]
)


@callback(Output("table-eda-container", "children"), Input("global-df-store", "data"))
def render_table_dataset(json_data):
    if json_data is None:
        return html.P("Loading data...")

    data_dict = json.loads(json_data)

    df = pd.DataFrame(
        data=data_dict["data"], columns=data_dict["columns"], index=data_dict["index"]
    )

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=5,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_data={"color": "black", "backgroundColor": "white"},
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "rgb(220, 220, 220)",
            }
        ],
        style_header={
            "backgroundColor": "rgb(210, 210, 210)",
            "color": "black",
            "fontWeight": "bold",
            "border": "1px solid black",
        },
        style_cell={"border": "1px solid grey"},
    )


@callback(Output("plot-pairplot", "figure"), Input("global-df-store", "data"))
def plot_pairplot(json_data):

    data_dict = json.loads(json_data)

    df = pd.DataFrame(
        data=data_dict["data"], columns=data_dict["columns"], index=data_dict["index"]
    )

    regressors = list(df.columns)
    regressors.remove("MedHouseVal")

    fig = go.Figure(
        data=go.Splom(
            dimensions=[{"label": i, "values": df[i]} for i in regressors],
            showupperhalf=False,
            diagonal=dict(visible=False),
            text=df["MedHouseVal"],
            marker=dict(
                color=df["MedHouseVal"],
                showscale=True,
                line_color="white",
                line_width=0.5,
            ),
        )
    )

    fig.update_layout(
        yaxis=dict(tickangle=45),
    )

    return fig


@callback(Output("plot-corr-matrix", "figure"), Input("global-df-store", "data"))
def plot_corr_mat(json_data):

    data_dict = json.loads(json_data)

    df = pd.DataFrame(
        data=data_dict["data"], columns=data_dict["columns"], index=data_dict["index"]
    )

    df = df.corr(numeric_only=True)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=df.columns,
            y=df.index,
            z=np.array(df),
            text=df.values,
            texttemplate="%{text:.2f}",
            colorscale=px.colors.diverging.Picnic_r,
            zmid=0,
            zmax=1,
            zmin=-1,
        )
    )

    return fig
