import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from fpdf import FPDF

# ----- CONFIG -----
st.set_page_config(page_title="Investment Dashboard", layout="wide")
history_file_path = "performance_history.json"

# ----- LANGUAGE SWITCH -----
lang = st.sidebar.selectbox("Language / Langue / 언어", ["English", "Français", "한국어"])
texts = {
    "English": {
        "login": "Login to Your Investment Dashboard",
        "email": "Email",
        "password": "Password",
        "warning": "Please enter your credentials to log in.",
        "success": "Logged in successfully!",
        "title": "Investment Portfolio Tracker",
        "tickers": "Enter tickers (comma separated)",
        "shares": "Enter number of shares (same order)",
        "buy_prices": "Enter buy prices (same order)",
        "track": "Track Portfolio",
        "summary": "Portfolio Summary",
        "allocation": "Asset Allocation",
        "compare": "Portfolio vs. S&P 500",
        "metrics": "Performance Metrics"
    },
    "Français": {
        "login": "Connexion à votre tableau de bord d'investissement",
        "email": "E-mail",
        "password": "Mot de passe",
        "warning": "Veuillez saisir vos identifiants pour vous connecter.",
        "success": "Connexion réussie !",
        "title": "Suivi de portefeuille d'investissement",
        "tickers": "Entrez les tickers (séparés par des virgules)",
        "shares": "Entrez le nombre d'actions (même ordre)",
        "buy_prices": "Entrez les prix d'achat (même ordre)",
        "track": "Suivre le portefeuille",
        "summary": "Résumé du portefeuille",
        "allocation": "Répartition des actifs",
        "compare": "Portefeuille vs. S&P 500",
        "metrics": "Indicateurs de performance"
    },
    "한국어": {
        "login": "투자 대시보드에 로그인",
        "email": "이메일",
        "password": "비밀번호",
        "warning": "로그인하려면 자격 증명을 입력하세요.",
        "success": "성공적으로 로그인되었습니다!",
        "title": "투자 포트폴리오 추적기",
        "tickers": "티커 입력 (쉼표로 구분)",
        "shares": "보유 주식 수 입력 (순서대로)",
        "buy_prices": "매수 가격 입력 (순서대로)",
        "track": "포트폴리오 추적",
        "summary": "포트폴리오 요약",
        "allocation": "자산 배분",
        "compare": "포트폴리오 vs. S&P 500",
        "metrics": "성과 지표"
    }
}
t = texts[lang]

# ----- LOGIN -----
EMAIL = st.secrets.get("EMAIL", "amahali.we@gmail.com")
PASSWORD = st.secrets.get("PASSWORD", "changeme")

st.title(f"🔐 {t['login']}")

email_input = st.text_input(t['email'])
password_input = st.text_input(t['password'], type="password")
login_button = st.button("Access Dashboard")

if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

if login_button:
    if email_input == EMAIL and password_input == PASSWORD:
        st.session_state.show_dashboard = True
        st.success(t['success'])
    else:
        st.error("❌ Invalid credentials. Please try again.")

# ----- DASHBOARD START -----
if st.session_state.show_dashboard:

    if "performance_history" not in st.session_state:
        if os.path.exists(history_file_path):
            with open(history_file_path, "r") as f:
                st.session_state.performance_history = json.load(f)
        else:
            st.session_state.performance_history = []

    st.title(f"📊 {t['title']}")
    tickers = st.text_input(t['tickers'], "AAPL, TSLA, VOO")
    shares = st.text_input(t['shares'], "10, 5, 7")
    buy_prices = st.text_input(t['buy_prices'], "145, 700, 380")

    if st.button(t['track']):
        tickers = [x.strip().upper() for x in tickers.split(",")]
        shares = [int(x.strip()) for x in shares.split(",")]
        buy_prices = [float(x.strip()) for x in buy_prices.split(",")]

        if not (len(tickers) == len(shares) == len(buy_prices)):
            st.error("⚠️ Ticker, share, and buy price counts must match.")
            st.stop()

        portfolio = [
            {"ticker": tickers[i], "shares": shares[i], "buy_price": buy_prices[i]}
            for i in range(len(tickers))
        ]

        price_data = yf.download(tickers, period="1d")
        if isinstance(price_data.columns, pd.MultiIndex):
            data = price_data['Adj Close'].dropna().iloc[-1]
        else:
            data = pd.Series({tickers[0]: price_data['Adj Close'].dropna().iloc[-1]})

        results = []
        total_value = 0
        total_cost = 0

        for item in portfolio:
            ticker = item['ticker']
            shares = item['shares']
            buy_price = item['buy_price']
            current_price = data[ticker]
            value = shares * current_price
            cost = shares * buy_price
            pnl = value - cost
            return_pct = (pnl / cost) * 100

            total_value += value
            total_cost += cost

            results.append({
                'Ticker': ticker,
                'Shares': shares,
                'Buy Price': buy_price,
                'Current Price': round(current_price, 2),
                'Current Value': round(value, 2),
                'P&L': round(pnl, 2),
                'Return %': round(return_pct, 2)
            })

        df = pd.DataFrame(results)
        st.subheader(t['summary'])
        st.dataframe(df)

        st.markdown(f"**Total Cost:** ${round(total_cost,2)}")
        st.markdown(f"**Total Value:** ${round(total_value,2)}")
        st.markdown(f"**Total P&L:** ${round(total_value - total_cost,2)}")

        # Save performance to history
        today_str = datetime.today().strftime("%Y-%m-%d")
        existing_dates = [entry["date"] for entry in st.session_state.performance_history]
        if today_str not in existing_dates:
            st.session_state.performance_history.append({
                "date": today_str,
                "total_value": round(total_value, 2),
                "total_cost": round(total_cost, 2),
                "pnl": round(total_value - total_cost, 2)
            })
            with open(history_file_path, "w") as f:
                json.dump(st.session_state.performance_history, f, indent=4)

        # Allocation pie chart
        st.subheader(t['allocation'])
        fig1, ax1 = plt.subplots()
        ax1.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%')
        st.pyplot(fig1)

        # Performance chart vs S&P 500
        st.subheader(t['compare'])
        start_date = "2023-01-01"
        prices = pd.DataFrame()
        for item in portfolio:
            hist = yf.download(item['ticker'], start=start_date)
            data_hist = hist['Adj Close'] if 'Adj Close' in hist else hist['Close']
            prices[item['ticker']] = data_hist * item['shares']

        portfolio_value = prices.sum(axis=1)
        sp500 = yf.download("^GSPC", start=start_date)["Adj Close"]
        sp500 = sp500 / sp500.iloc[0] * portfolio_value.iloc[0]

        fig2, ax2 = plt.subplots()
        portfolio_value.plot(ax=ax2, label="Your Portfolio")
        sp500.plot(ax=ax2, label="S&P 500", linestyle="--")
        ax2.set_title("Portfolio vs. S&P 500")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # Performance metrics
        st.subheader(t['metrics'])
        returns = portfolio_value.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        roll_max = portfolio_value.cummax()
        drawdown = (portfolio_value - roll_max) / roll_max
        max_drawdown = drawdown.min()
        days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
        cagr = (portfolio_value[-1] / portfolio_value[0])**(365.0/days) - 1

        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col2.metric("Max Drawdown", f"{max_drawdown:.2%}")
        col3.metric("CAGR", f"{cagr:.2%}")

        # Historical P&L chart
        if st.session_state.performance_history:
            history_df = pd.DataFrame(st.session_state.performance_history)
            history_df["date"] = pd.to_datetime(history_df["date"])
            history_df = history_df.sort_values("date")
            history_df.set_index("date", inplace=True)

            st.subheader("📊 Historical Portfolio Performance")
            if all(col in history_df.columns for col in ["total_value", "pnl"]):
                st.line_chart(history_df[["total_value", "pnl"]])
                st.dataframe(history_df)
                csv_export = history_df.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download History as CSV",
                    data=csv_export,
                    file_name="performance_history.csv",
                    mime="text/csv"
                )
            else:
                st.warning("🚫 Missing 'total_value' or 'pnl' in history. Please track a portfolio again.")
        else:
            st.info("📬 No historical performance yet. Track a portfolio to begin.")
else:
    st.warning(t['warning'])
