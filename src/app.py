import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from utils import get_pca_dfs, load_dataset

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

df = load_dataset()
df_pca_train, df_pca_val, df_pca_test = get_pca_dfs(df)

df = df.to_json(orient="split")
df_pca_train = df_pca_train.to_json(orient="split")
df_pca_val = df_pca_val.to_json(orient="split")
df_pca_test = df_pca_test.to_json(orient="split")


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(f"{page['name']}", href=page["relative_path"], active="exact")
        )
        for page in dash.page_registry.values()
    ],
    brand="Linear Models Dashboard",
    brand_href="/",
    color="dark",
    dark=True,
    fluid=True,
    className="mb-4",
)

app.layout = html.Div(
    [
        dcc.Store(id="global-df-store", data=df, storage_type="memory"),
        dcc.Store(id="global-pca-train", data=df_pca_train, storage_type="memory"),
        dcc.Store(id="global-pca-val", data=df_pca_val, storage_type="memory"),
        dcc.Store(id="global-pca-test", data=df_pca_test, storage_type="memory"),
        navbar,
        dbc.Container([dash.page_container], fluid=True),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
