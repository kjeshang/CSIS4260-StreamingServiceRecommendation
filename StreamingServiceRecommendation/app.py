from dash import Dash, html
import dash_bootstrap_components as dbc

import page

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX]);

app.layout = html.Div([page.pageLayout]);

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)