# ASX 200 Portfolio Optimizer

> 🌐 Live demo: https://asx-portfolio-dashboard.onrender.com

A web application that computes the **Efficient Frontier** for ASX 200 stocks using Modern Portfolio Theory, deployed on Render.

## Results
- Expected Annual Return: 23.8%
- Annual Volatility: 10.8%
- Sharpe Ratio: 1.84 (risk-free rate: 4%)

## Tech Stack
- **Backend**: Python, Flask, PyPortfolioOpt, yfinance
- **Frontend**: Plotly.js interactive charts
- **Deployment**: Render (CI/CD via GitHub)
- **Theory**: Markowitz MPT, Mean-Variance Optimisation, Sharpe Ratio maximisation

## Local Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
