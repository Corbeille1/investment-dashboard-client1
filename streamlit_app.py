import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime
from fpdf import FPDF
from io import BytesIO

# -------------------------------
# SETUP & CONFIG
# -------------------------------
st.set_page_config(page_title="Your Investment Dashboard", layout="wide")
st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# -------------------------------
# LANGUAGE SETUP
# -------------------------------
lang = st.sidebar.selectbox("Language", ["English"])
texts = {
    "English": {
        "login": "Login to Your Investment Dashboard",
        "email": "Email",
        "password": "Password",
        "title": "Investment Portfolio Tracker",
        "tickers": "Enter tickers (comma separated)",
        "shares": "Enter number of shares (same order)",
        "buy_prices": "Enter buy prices (same order)",
        "track": "Track Portfolio",
        "summary": "Portfolio Summary",
        "allocation": "Asset Allocation",
        "compare": "Portfolio vs. S&P 500",
        "metrics": "Performance Metrics",
        "history": "📊 Historical Portfolio Performance",
    },
}
t = texts[lang]

# -------------------------------
# LOGIN SYSTEM
# -------------------------------
EMAIL = st.secrets.get("EMAIL", "amahali.we@gmail.com")
PASSWORD = st.secrets.get("PASSWORD", "changeme")

if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

if "performance_history" not in st.session_state:
    if os.path.exists("performance_history.json"):
        with open("performance_history.json", "r") as f:
            try:
                st.session_state.performance_history = json.load(f)
            except:
                st.session_state.performance_history = []
    else:
        st.session_state.performance_history = []

# -------------------------------
# LOGIN FORM
# -------------------------------
st.title(f"🔒 {t['login']}")
email_input = st.text_input(t["email"])
password_input = st.text_input(t["password"], type="password")
if st.button("Access Dashboard"):
    if email_input == EMAIL and password_input == PASSWORD:
        st.session_state.show_dashboard = True
    else:
        st.error("❌ Invalid credentials.")

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
if st.session_state.show_dashboard:
    st.title(f"📊 {t['title']}")
    uploaded_file = st.file_uploader("📂 Load portfolio (JSON or CSV)", type=["json", "csv"])
    tickers = st.text_input(t["tickers"], st.session_state.get("tickers", "AAPL, TSLA, VOO"))
    shares = st.text_input(t["shares"], st.session_state.get("shares", "10, 5, 7"))
    buy_prices = st.text_input(t["buy_prices"], st.session_state.get("buy_prices", "145, 700, 380"))

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".json"):
                data = json.load(uploaded_file)
            else:
                df_uploaded = pd.read_csv(uploaded_file)
                data = df_uploaded.to_dict(orient="records")

            # Fill fields
            tickers = ", ".join([x["ticker"] for x in data])
            shares = ", ".join([str(x["shares"]) for x in data])
            buy_prices = ", ".join([str(x["buy_price"]) for x in data])

            st.session_state["tickers"] = tickers
            st.session_state["shares"] = shares
            st.session_state["buy_prices"] = buy_prices
            st.success("✅ Portfolio loaded!")
        except Exception as e:
            st.error(f"❌ Failed to load portfolio: {e}")

    if st.button(t["track"]):
        try:
            tickers = [x.strip().upper() for x in tickers.split(",")]
            shares = [int(x.strip()) for x in shares.split(",")]
            buy_prices = [float(x.strip()) for x in buy_prices.split(",")]

            if not (len(tickers) == len(shares) == len(buy_prices)):
                st.error("❌ Tickers, shares, and buy prices count mismatch.")
                st.stop()

            portfolio = [{"ticker": t, "shares": s, "buy_price": b} for t, s, b in zip(tickers, shares, buy_prices)]

            # Fetch price data
            price_data = yf.download(tickers, period="5d", group_by="ticker", auto_adjust=True)

            # Build price dict
            price_row = {}
            if isinstance(price_data.columns, pd.MultiIndex):
                for t in tickers:
                    if "Close" in price_data[t]:
                        price_row[t] = price_data[t]["Close"].dropna().iloc[-1]
                    else:
                        st.warning(f"⚠️ No 'Close' data for {t}")
            else:
                price_row[tickers[0]] = price_data["Close"].dropna().iloc[-1]

            # Build results table
            rows = []
            total_value = total_cost = 0
            for item in portfolio:
                price = price_row.get(item["ticker"], 0)
                value = item["shares"] * price
                cost = item["shares"] * item["buy_price"]
                pnl = value - cost
                rows.append({
                    "Ticker": item["ticker"],
                    "Shares": item["shares"],
                    "Buy Price": item["buy_price"],
                    "Current Price": round(price, 2),
                    "Current Value": round(value, 2),
                    "P&L": round(pnl, 2),
                    "Return %": round((pnl / cost) * 100 if cost else 0, 2)
                })
                total_value += value
                total_cost += cost

            df = pd.DataFrame(rows)
            st.subheader(t["summary"])
            st.dataframe(df)
            st.markdown(f"**Total Cost:** ${total_cost:,.2f}")
            st.markdown(f"**Total Value:** ${total_value:,.2f}")
            st.markdown(f"**Total P&L:** ${total_value - total_cost:,.2f}")

            # Save daily performance
            today = datetime.today().strftime("%Y-%m-%d")
            if not any(row["date"] == today for row in st.session_state.performance_history):
                st.session_state.performance_history.append({
                    "date": today,
                    "total_value": round(total_value, 2),
                    "total_cost": round(total_cost, 2),
                    "pnl": round(total_value - total_cost, 2)
                })
                with open("performance_history.json", "w") as f:
                    json.dump(st.session_state.performance_history, f, indent=2)

            # Asset Allocation
            st.subheader(t["allocation"])
            fig1, ax1 = plt.subplots()
            ax1.pie(df["Current Value"], labels=df["Ticker"], autopct="%1.1f%%")
            st.pyplot(fig1)

            # Portfolio vs. S&P
            st.subheader(t["compare"])
            start_date = "2023-01-01"
            prices = pd.DataFrame()
            for item in portfolio:
                hist = yf.download(item["ticker"], start=start_date)["Close"]
                prices[item["ticker"]] = hist * item["shares"]
            portfolio_value = prices.sum(axis=1)
            sp500 = yf.download("^GSPC", start=start_date)["Close"]
            sp500 = sp500 / sp500.iloc[0] * portfolio_value.iloc[0]
            fig2, ax2 = plt.subplots()
            portfolio_value.plot(ax=ax2, label="Portfolio")
            sp500.plot(ax=ax2, label="S&P 500", linestyle="--")
            ax2.legend()
            ax2.set_title("Portfolio vs. S&P 500")
            st.pyplot(fig2)

            # Performance Metrics
            st.subheader(t["metrics"])
            returns = portfolio_value.pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            drawdown = ((portfolio_value / portfolio_value.cummax()) - 1).min()
            days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
            cagr = (portfolio_value[-1] / portfolio_value[0]) ** (365.0 / days) - 1

            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col2.metric("Max Drawdown", f"{drawdown:.2%}")
            col3.metric("CAGR", f"{cagr:.2%}")

            # History
            st.subheader(t["history"])
            history_df = pd.DataFrame(st.session_state.performance_history)
            if not history_df.empty and "total_value" in history_df.columns and "pnl" in history_df.columns:
                history_df["date"] = pd.to_datetime(history_df["date"])
                history_df = history_df.set_index("date").sort_index()
                st.line_chart(history_df[["total_value", "pnl"]])
                st.dataframe(history_df)
            else:
                st.info("📭 No performance history available.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
