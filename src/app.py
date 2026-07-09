import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from utils import load_dataset

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

df = load_dataset()
df_json = df.to_json(orient="split")

app.layout = html.Div(
    [
        dcc.Store(id="global-df-store", data=df_json, storage_type="memory"),
        html.Header(
            [
                html.Nav(
                    [
                        dcc.Link(
                            f"{page['name']}",
                            href=page["relative_path"],
                            className="nav-link",
                        )
                        for page in dash.page_registry.values()
                    ],
                    className="navbar",
                )
            ]
        ),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
