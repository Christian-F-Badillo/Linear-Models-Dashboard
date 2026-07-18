import os

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)


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
        navbar,
        dbc.Container([dash.page_container], fluid=True),
    ]
)

if __name__ == "__main__":
    HOST = os.environ.get("DASH_HOST", "127.0.0.1")
    PORT = int(os.environ.get("DASH_PORT", 8050))

    DEBUG = os.environ.get("DASH_DEBUG", "True").lower() == "true"

    app.run(host=HOST, port=PORT, debug=DEBUG)
