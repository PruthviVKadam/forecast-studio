"""Forecasting Studio — a Plotly Dash time-series model bake-off.

Pick a series, choose models, and the app runs walk-forward backtesting, ranks the
models by error (MASE/MAPE/RMSE), and overlays each model's forecast against a held-out
tail. Naive baselines are always included — the honest takeaway ("a baseline wins") is
often the point.

Run locally:  python app.py   (http://localhost:8050)
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

from core.backtest import aggregate, walk_forward
from core.data import PRESETS, get_series
from core.models import BASELINE_NAMES, default_models

SEASON = 5
MODELS = default_models(SEASON)
OPTIONAL_MODELS = [name for name in MODELS if name not in BASELINE_NAMES]

app = Dash(__name__, title="Forecasting Studio")
server = app.server  # WSGI entrypoint for gunicorn / HF Spaces / Render


def run_study(ticker, optional_selected, horizon, n_folds):
    series = get_series(ticker)
    names = BASELINE_NAMES + [n for n in optional_selected if n in MODELS]

    rows, forecasts = [], {}
    for name in names:
        factory, _ = MODELS[name]
        agg = aggregate(walk_forward(series, factory, horizon=horizon, n_folds=n_folds, season=SEASON))
        rows.append(
            {
                "Model": name,
                "MASE": round(agg["mase"], 3),
                "MAPE %": round(agg["mape"], 2),
                "RMSE": round(agg["rmse"], 3),
            }
        )
        model = factory()
        model.fit(series.iloc[:-horizon])
        forecasts[name] = model.predict(horizon)

    rows.sort(key=lambda r: (float("inf") if r["MASE"] != r["MASE"] else r["MASE"]))
    return series, rows, forecasts


def build_figure(series, forecasts, horizon, ticker):
    tail = series.iloc[-(horizon * 8):]
    test_idx = series.index[-horizon:]
    fig = go.Figure()
    fig.add_scatter(x=tail.index, y=tail.to_numpy(), name="Actual", line=dict(color="#e6e8ee", width=2))
    for name, fc in forecasts.items():
        fig.add_scatter(x=test_idx, y=fc, name=name, mode="lines+markers")
    fig.add_vline(x=test_idx[0], line_dash="dot", line_color="#8b949e")
    fig.update_layout(
        title=f"{ticker}: forecasts vs. held-out actuals (last {horizon} steps)",
        template="plotly_dark",
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.18),
        paper_bgcolor="#171a23",
        plot_bgcolor="#171a23",
    )
    return fig


def takeaway(rows):
    best = rows[0]
    if best["Model"] in BASELINE_NAMES:
        return (f"🏅 A baseline wins: **{best['Model']}** (MASE {best['MASE']}). "
                "The fancier models don't beat it here — a classic reminder to always ship the baseline.")
    verdict = "beats" if best["MASE"] < 1 else "is best but still doesn't beat the in-sample naive (MASE ≥ 1)"
    return f"🏅 **{best['Model']}** {verdict} the baselines (MASE {best['MASE']}, MAPE {best['MAPE %']}%)."


# ----------------------------- layout ----------------------------- #
CONTROL = {"marginBottom": "16px"}

app.layout = html.Div(
    className="wrap",
    children=[
        html.H1("📈 Forecasting Studio"),
        html.P(
            "Walk-forward model bake-off — ARIMA, Prophet, and gradient boosting vs. naive "
            "baselines, scored honestly with leakage-free backtesting.",
            className="sub",
        ),
        html.Div(
            className="grid",
            children=[
                html.Div(
                    className="panel",
                    children=[
                        html.H3("Series"),
                        dcc.Dropdown(
                            id="ticker",
                            options=[{"label": v, "value": k} for k, v in PRESETS.items()],
                            value="SPY",
                            clearable=False,
                            style=CONTROL,
                        ),
                        dcc.Input(id="custom-ticker", type="text", placeholder="…or type any ticker",
                                  debounce=True, style={"width": "100%", **CONTROL}),
                        html.H3("Optional models"),
                        dcc.Checklist(
                            id="models",
                            options=[{"label": " " + m, "value": m} for m in OPTIONAL_MODELS],
                            value=[m for m in OPTIONAL_MODELS if "Prophet" not in m],
                            style=CONTROL,
                        ),
                        html.P(f"Always included: {', '.join(BASELINE_NAMES)}", className="muted small"),
                        html.H3("Forecast horizon"),
                        dcc.Slider(id="horizon", min=3, max=20, step=1, value=5,
                                   marks={3: "3", 10: "10", 20: "20"}),
                        html.H3("Backtest folds"),
                        dcc.Slider(id="folds", min=2, max=6, step=1, value=4,
                                   marks={2: "2", 4: "4", 6: "6"}),
                        html.Button("Run backtest", id="run", n_clicks=0, className="run-btn"),
                        html.P("Prophet adds a few seconds.", className="muted small"),
                    ],
                ),
                html.Div(
                    className="panel",
                    children=[
                        dcc.Loading(
                            children=[
                                html.Div(id="takeaway", className="takeaway"),
                                dcc.Graph(id="forecast-plot"),
                                html.H3("Leaderboard — ranked by MASE (lower is better; <1 beats naive)"),
                                dash_table.DataTable(
                                    id="leaderboard",
                                    columns=[{"name": c, "id": c} for c in ["Model", "MASE", "MAPE %", "RMSE"]],
                                    style_as_list_view=True,
                                    style_header={"backgroundColor": "#1d212d", "color": "#a2a8b8", "fontWeight": "600"},
                                    style_cell={"backgroundColor": "#171a23", "color": "#e6e8ee",
                                                "border": "1px solid #262b38", "padding": "8px 12px",
                                                "fontFamily": "system-ui"},
                                    style_data_conditional=[
                                        {"if": {"row_index": 0}, "backgroundColor": "rgba(63,185,80,0.12)"}
                                    ],
                                ),
                            ]
                        )
                    ],
                ),
            ],
        ),
        html.P("Educational demo — not investment advice. Data via yfinance (cached).",
               className="muted small footer"),
    ],
)


@app.callback(
    Output("forecast-plot", "figure"),
    Output("leaderboard", "data"),
    Output("takeaway", "children"),
    Input("run", "n_clicks"),
    State("ticker", "value"),
    State("custom-ticker", "value"),
    State("models", "value"),
    State("horizon", "value"),
    State("folds", "value"),
)
def on_run(_n_clicks, ticker, custom_ticker, models, horizon, folds):
    ticker = (custom_ticker or ticker or "SPY").strip().upper()
    try:
        series, rows, forecasts = run_study(ticker, models or [], int(horizon), int(folds))
    except Exception as exc:  # noqa: BLE001 - surface any data/backtest error to the UI
        return go.Figure(), [], f"⚠️ {exc}"
    return build_figure(series, forecasts, int(horizon), ticker), rows, takeaway(rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
