# Indian Stock Market Predictor (NSE)

A machine learning–based web application that predicts **next-day price direction (UP/DOWN)** for NSE-listed stocks using **XGBoost** and historical market data.

---

## Key Features
- Next trading day direction prediction (UP / DOWN)
- Estimated next-day price based on recent volatility
- Technical indicators: SMA, EMA, daily returns
- Interactive web app built with Streamlit

---

## Tech Stack
- Python, Pandas, NumPy
- XGBoost, scikit-learn
- yfinance (Yahoo Finance)
- Streamlit

---

## How It Works
Historical NSE data is fetched using Yahoo Finance, technical indicators are engineered, and an XGBoost classifier predicts short-term price movement with a confidence score.

---

## How to Run
```bash
git clone https://github.com/shreeshtjagga/indian-stock-predictor.git
cd indian-stock-predictor
pip install -r requirements.txt
streamlit run app.py
