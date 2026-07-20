from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme, Trim

DATA_DIR = Path("data/raw")

df = pd.read_parquet(DATA_DIR / "raw_data.parquet")

dash.register_page(__name__, path="/eda", name="Exploratory Data Analysis (EDA)")

layout = html.Div(
    [
        html.H1(
            "Exploratory Data Analysis",
            style={"text-weight": "bold", "text-align": "center"},
            className="text-2xl font-bold mb-4",
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col([html.Div(id="table-eda-raw-data")], md=6),
                        dbc.Col([html.Div(id="table-eda-stats")], md=6),
                        dbc.Col([], md=1),
                    ]
                ),
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
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dcc.Dropdown(
                                            placeholder="Select a Variable ...",
                                            id="dropdown-scatterplot",
                                        )
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dcc.Graph(
                                            id="plot-scatterplot",
                                            responsive=True,
                                            style=dict(width="100%", height="650px"),
                                        )
                                    ]
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Dropdown(
                                                    options=[
                                                        "PC1",
                                                        "PC2",
                                                        "PC3",
                                                        "PC4",
                                                    ],
                                                    value="PC1",
                                                    placeholder="PCA dimension",
                                                    id="dropdown-pca-x",
                                                )
                                            ],
                                            md=3,
                                        ),
                                        dbc.Col(
                                            [
                                                dcc.Dropdown(
                                                    options=[
                                                        "PC1",
                                                        "PC2",
                                                        "PC3",
                                                        "PC4",
                                                    ],
                                                    value="PC2",
                                                    placeholder="PCA dimension",
                                                    id="dropdown-pca-y",
                                                )
                                            ],
                                            md=3,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dcc.Graph(
                                            id="plot-pca",
                                            responsive=True,
                                            style=dict(width="100%", height="650px"),
                                        )
                                    ]
                                ),
                            ],
                            md=6,
                        ),
                    ]
                ),
            ],
            fluid=True,
        ),
    ]
)


# ---------------------------------------------------------------------------------------
# TABLE RAW DATA
# ---------------------------------------------------------------------------------------
@callback(Output("table-eda-raw-data", "children"))
def render_table_dataset():
    df_table = df.copy()
    return dash_table.DataTable(
        data=df_table.to_dict("records"),
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.fixed, trim=Trim.yes),
            }
            for i in df_table.columns
        ],
        page_size=8,
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
        style_cell={"border": "1px solid grey", "fontSize": 13},
    )


# ---------------------------------------------------------------------------------------
# TABLE SUMMARY STATISTICS
# ---------------------------------------------------------------------------------------
@callback(Output("table-eda-stats", "children"))
def render_summary_stats():

    df_sum = df.describe()
    df_sum = df_sum.reset_index()
    df_sum = df_sum.rename(columns={"index": "Stat"})
    df_sum.Stat = df_sum.Stat.replace(["25%", "50%", "75%"], ["q25", "median", "q75"])
    df_sum.Stat = df_sum.Stat.str.upper()

    return dash_table.DataTable(
        data=df_sum.to_dict("records"),
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.fixed, trim=Trim.yes),
            }
            for i in df_sum.columns
        ],
        page_size=8,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_data={"color": "black", "backgroundColor": "white"},
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "rgb(220, 220, 220)",
            },
            {
                "if": {"column_id": "Stat"},
                "backgroundColor": "#99FFFF",
                "color": "black",
                "font-weight": "bold",
                "textAlign": "left",
            },
        ],
        style_header={
            "backgroundColor": "rgb(210, 210, 210)",
            "color": "black",
            "fontWeight": "bold",
            "border": "1px solid black",
        },
        style_cell={"border": "1px solid grey", "fontSize": 13},
    )


# ---------------------------------------------------------------------------------------
# DROPDOWN DATA INJECTION
# ---------------------------------------------------------------------------------------
@callback(Output("dropdown-scatterplot", "options"))
def get_columns_dropdown():

    regressors = list(df.columns)
    regressors.remove("MedHouseVal")

    return regressors


# ---------------------------------------------------------------------------------------
# SCATTER PLOT REGRESSOR VS TARGET
# ---------------------------------------------------------------------------------------
@callback(Output("plot-scatterplot", "figure"), Input("dropdown-scatterplot", "value"))
def plot_scatter(column):

    if column is None:
        column = "MedInc"

    fig = px.scatter(
        data_frame=df,
        y=column,
        x="MedHouseVal",
        color="MedHouseVal",
        marginal_x="violin",
        marginal_y="box",
        trendline="rolling",
        trendline_options=dict(function="median", window=500),
        trendline_color_override="black",
    )

    return fig


# ---------------------------------------------------------------------------------------
# PCA PLOT
# ---------------------------------------------------------------------------------------
@callback(
    Output("plot-pca", "figure"),
    [
        Input("dropdown-pca-x", "value"),
        Input("dropdown-pca-y", "value"),
    ],
)
def plot_pca(pc_x, pc_y):

    df_pca = pd.read_parquet(DATA_DIR / "df_pca_train.parquet")

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=df_pca[pc_x],
            y=df_pca[pc_y],
            mode="markers",
            marker=dict(
                color=df["MedHouseVal"],
                colorscale="viridis",
                size=4,
                opacity=0.7,
                showscale=True,
                colorbar=dict(title="MedHouseVal", thickness=15, len=0.8),
            ),
            # Inyectamos customdata y hovertemplate de forma nativa
            customdata=np.stack([df["MedHouseVal"]], axis=-1),
            hovertemplate=(
                f"<b>{pc_x}:</b> %{{x:.2f}}<br>"
                f"<b>{pc_y}:</b> %{{y:.2f}}<br>"
                "<b>MedHouseVal:</b> %{customdata[0]:.2f}$<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        autosize=True,
        xaxis_title=f"{pc_x}",
        yaxis_title=f"{pc_y}",
        margin=dict(l=40, r=10, t=40, b=40),
        height=450,
        font=dict(family="monospace", size=10),
        plot_bgcolor="#f8f9fa",
    )

    return fig


# ---------------------------------------------------------------------------------------
# PAIR PLOT
# ---------------------------------------------------------------------------------------
@callback(Output("plot-pairplot", "figure"))
def plot_pairplot():

    df_mod = df.copy()
    df_mod["MedHouseVal"] = df_mod["MedHouseVal"] * 100000

    regressors = list(df_mod.columns)
    regressors.remove("MedHouseVal")

    splom_hover = (
        "<b>%{xaxis.title.text}:</b> %{x:.2f}<br>"
        "<b>%{yaxis.title.text}:</b> %{y:.2f}<br>"
        "<b>MedHouseVal:</b> $%{text:.2s}<extra></extra>"
    )

    fig = go.Figure(
        data=go.Splom(
            dimensions=[{"label": i, "values": df_mod[i]} for i in regressors],
            showupperhalf=False,
            diagonal=dict(visible=False),
            text=df_mod["MedHouseVal"],
            hovertemplate=splom_hover,
            marker=dict(
                color=df_mod["MedHouseVal"],
                showscale=True,
                colorscale="plasma",
                colorbar=dict(title="MedVal", thickness=15, len=0.7),
                line_color="white",
                line_width=0.3,
                opacity=0.8,
            ),
        )
    )

    fig.update_layout(
        yaxis=dict(tickangle=45),
        autosize=True,
        margin=dict(l=80, r=20, t=60, b=20),
        height=450,
        font=dict(family="monospace", size=10),
    )

    return fig


# ---------------------------------------------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------------------------------------------
@callback(Output("plot-corr-matrix", "figure"))
def plot_corr_mat():

    df_corr = df.corr()

    matrix_hover = (
        "<b>%{y}</b><br><b>%{x}</b><br><b>Pearson Corr:</b> %{z:.3f}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=df_corr.columns,
            y=df_corr.index,
            z=df_corr.to_numpy(),
            text=df_corr.values,
            texttemplate="%{text:.2f}",
            hovertemplate=matrix_hover,
            colorscale=px.colors.diverging.Picnic_r,
            zmid=0,
            zmax=1,
            zmin=-1,
        )
    )

    fig.update_xaxes(side="top")
    fig.update_traces(showscale=False)

    fig.update_layout(
        autosize=True,
        margin=dict(l=80, r=20, t=60, b=20),
        height=450,
        font=dict(family="monospace", size=10),
    )

    return fig
