from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import plotly.graph_objects as go
import json
import traceback

app = Flask(__name__, template_folder="templates")

# ASX 200 精选股票（10只流动性最好的蓝筹）
ASX_TICKERS = [
    "BHP.AX",   # BHP Group - 矿业
    "CBA.AX",   # Commonwealth Bank - 银行
    "CSL.AX",   # CSL Limited - 生物科技
    "NAB.AX",   # National Australia Bank - 银行
    "WBC.AX",   # Westpac - 银行
    "ANZ.AX",   # ANZ Bank - 银行
    "WES.AX",   # Wesfarmers - 零售
    "MQG.AX",   # Macquarie Group - 金融
    "RIO.AX",   # Rio Tinto - 矿业
    "TLS.AX",   # Telstra - 电信
]


def fetch_price_data():
    """从 Yahoo Finance 获取2年历史收盘价"""
    raw = yf.download(
        ASX_TICKERS,
        period="2y",
        auto_adjust=True,
        progress=False
    )

    # yfinance 多股票返回 MultiIndex DataFrame，取 Close 层
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    # 删掉全为 NaN 的行和列
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # 向前填充少量缺失值（公假日等）
    prices = prices.ffill().dropna()

    if prices.empty or prices.shape[1] < 2:
        raise ValueError("获取到的价格数据不足，请检查网络连接")

    return prices


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/frontier")
def frontier():
    try:
        # ── 1. 获取数据 ──────────────────────────────────────────
        prices = fetch_price_data()
        tickers_used = list(prices.columns)

        # ── 2. 计算预期收益率 & 协方差矩阵 ──────────────────────
        mu = expected_returns.mean_historical_return(prices)
        S  = risk_models.sample_cov(prices)

        # ── 3. 生成有效前沿曲线（30个点）────────────────────────
        target_returns = np.linspace(float(mu.min()) + 0.001,
                                     float(mu.max()) - 0.001, 30)
        frontier_vols, frontier_rets = [], []

        for r in target_returns:
            try:
                ef_tmp = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                ef_tmp.efficient_return(r)
                perf = ef_tmp.portfolio_performance(verbose=False)
                frontier_rets.append(perf[0])
                frontier_vols.append(perf[1])
            except Exception:
                continue

        # ── 4. 最优夏普比率组合 ──────────────────────────────────
        ef_sharpe = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef_sharpe.max_sharpe(risk_free_rate=0.04)   # 澳大利亚无风险利率约4%
        sharpe_weights_raw = ef_sharpe.clean_weights()
        sharpe_perf = ef_sharpe.portfolio_performance(
            verbose=False, risk_free_rate=0.04
        )

        # ── 5. 最小方差组合 ──────────────────────────────────────
        ef_minvol = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef_minvol.min_volatility()
        minvol_perf = ef_minvol.portfolio_performance(verbose=False)

        # ── 6. 各股票单独风险收益（散点用）─────────────────────
        annual_ret  = {k: float(v) for k, v in mu.to_dict().items()}
        annual_vol  = {t: float(np.sqrt(S.loc[t, t])) for t in S.columns}

        # ── 7. 构建 Plotly 图表 ──────────────────────────────────
        fig = go.Figure()

        # 有效前沿曲线
        fig.add_trace(go.Scatter(
            x=frontier_vols,
            y=frontier_rets,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#185FA5", width=2.5),
            hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
        ))

        # 个股散点
        fig.add_trace(go.Scatter(
            x=[annual_vol[t] for t in tickers_used],
            y=[annual_ret[t] for t in tickers_used],
            mode="markers+text",
            name="Individual stocks",
            text=tickers_used,
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(color="#B4B2A9", size=8, symbol="circle"),
            hovertemplate="%{text}<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
        ))

        # 最小方差组合
        fig.add_trace(go.Scatter(
            x=[minvol_perf[1]],
            y=[minvol_perf[0]],
            mode="markers",
            name="Min Variance",
            marker=dict(color="#1D9E75", size=14, symbol="diamond"),
            hovertemplate="Min Variance<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
        ))

        # 最大夏普比率组合
        fig.add_trace(go.Scatter(
            x=[sharpe_perf[1]],
            y=[sharpe_perf[0]],
            mode="markers",
            name="Max Sharpe",
            marker=dict(color="#E85D24", size=16, symbol="star"),
            hovertemplate="Max Sharpe<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(
                text="ASX 200 Portfolio — Efficient Frontier",
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Annualised Volatility (Risk)",
                tickformat=".0%",
                gridcolor="#f0f0f0"
            ),
            yaxis=dict(
                title="Annualised Expected Return",
                tickformat=".0%",
                gridcolor="#f0f0f0"
            ),
            legend=dict(x=0.02, y=0.98),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=520,
            margin=dict(l=60, r=40, t=60, b=60),
            hovermode="closest"
        )

        # ── 8. 权重表（只显示 >0.5% 的持仓）────────────────────
        weights_display = {
            k: round(v, 4)
            for k, v in sharpe_weights_raw.items()
            if v > 0.005
        }

        return jsonify({
            "status": "ok",
            "chart": json.loads(fig.to_json()),
            "weights": weights_display,
            "metrics": {
                "expected_return": round(sharpe_perf[0], 4),
                "volatility":      round(sharpe_perf[1], 4),
                "sharpe_ratio":    round(sharpe_perf[2], 4),
            },
            "tickers_used": tickers_used,
            "data_points":  len(prices),
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "detail": traceback.format_exc()
        }), 500


@app.route("/api/health")
def health():
    """健康检查端点，Azure 会用到"""
    return jsonify({"status": "ok", "app": "ASX Portfolio Optimizer"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
