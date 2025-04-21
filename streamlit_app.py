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

# --------------------------------------------
# 🔧 PAGE SETTINGS
# --------------------------------------------
st.set_page_config(page_title="Your Investment Dashboard", layout="wide")
st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# --------------------------------------------
# 🌐 LANGUAGES
# --------------------------------------------
lang = st.sidebar.selectbox("Language / 언어 / Langue", ["English", "Français", "한국어"])
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
        "metrics": "Performance Metrics",
        "history": "📊 Historical Portfolio Performance"
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
        "metrics": "Indicateurs de performance",
        "history": "📊 Historique du portefeuille"
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
        "metrics": "성과 지표",
        "history": "📊 포트폴리오 히스토리"
    }
}
t = texts[lang]

# --------------------------------------------
# 🔐 LOGIN SYSTEM
# --------------------------------------------
EMAIL = st.secrets.get("EMAIL", "amahali.we@gmail.com")
PASSWORD = st.secrets.get("PASSWORD", "changeme")

st.title(f"🔒 {t['login']}")

if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

if 'performance_history' not in st.session_state:
    if os.path.exists("performance_history.json"):
        with open("performance_history.json", "r") as f:
            st.session_state.performance_history = json.load(f)
    else:
        st.session_state.performance_history = []

email_input = st.text_input(t['email'])
password_input = st.text_input(t['password'], type="password")
login_button = st.button("Access Dashboard")

if login_button:
    if email_input == EMAIL and password_input == PASSWORD:
        st.session_state.show_dashboard = True
    else:
        st.error("❌ Invalid credentials. Please try again.")

# --------------------------------------------
# 📊 DASHBOARD UI
# --------------------------------------------
if st.session_state.show_dashboard:
    st.title(f"📊 {t['title']}")

    uploaded_file = st.file_uploader("📂 Load portfolio (JSON or CSV)", type=["json", "csv"])
    tickers = st.text_input(t['tickers'], st.session_state.get("tickers", "AAPL, TSLA, VOO"))
    shares = st.text_input(t['shares'], st.session_state.get("shares", "10, 5, 7"))
    buy_prices = st.text_input(t['buy_prices'], st.session_state.get("buy_prices", "145, 700, 380"))

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".json"):
                data = json.load(uploaded_file)
            else:
                df_uploaded = pd.read_csv(uploaded_file)
                data = df_uploaded.to_dict(orient="records")
            tickers = ", ".join([x["ticker"] for x in data])
            shares = ", ".join([str(x["shares"]) for x in data])
            buy_prices = ", ".join([str(x["buy_price"]) for x in data])
            st.success("✅ Portfolio loaded and fields filled!")
        except Exception as e:
            st.error(f"❌ Failed to load portfolio: {e}")

    if st.button(t['track']):
        try:
            tickers = [x.strip().upper() for x in tickers.split(",")]
            shares = [int(x.strip()) for x in shares.split(",")]
            buy_prices = [float(x.strip()) for x in buy_prices.split(",")]

            if not (len(tickers) == len(shares) == len(buy_prices)):
                st.error("⚠️ The number of tickers, shares, and buy prices must match.")
                st.stop()

            portfolio = []
            for i in range(len(tickers)):
                portfolio.append({
                    'ticker': tickers[i],
                    'shares': shares[i],
                    'buy_price': buy_prices[i]
                })

            price_data = yf.download(tickers, period="5d", group_by='ticker')

            # Handle single ticker edge case
            if isinstance(price_data.columns, pd.MultiIndex):
                price_row = {ticker: price_data[ticker]["Adj Close"].dropna().iloc[-1] for ticker in tickers}
            else:
                price_row = {tickers[0]: price_data["Adj Close"].dropna().iloc[-1]}

            price_row = pd.Series(price_row)


            results = []
            total_value = 0
            total_cost = 0

            for item in portfolio:
                ticker = item["ticker"]
                shares = item["shares"]
                buy_price = item["buy_price"]
                current_price = price_row[ticker]
                value = shares * current_price
                cost = shares * buy_price
                pnl = value - cost
                return_pct = (pnl / cost) * 100

                results.append({
                    "Ticker": ticker,
                    "Shares": shares,
                    "Buy Price": buy_price,
                    "Current Price": round(current_price, 2),
                    "Current Value": round(value, 2),
                    "P&L": round(pnl, 2),
                    "Return %": round(return_pct, 2)
                })

                total_value += value
                total_cost += cost

            df = pd.DataFrame(results)
            st.subheader(t['summary'])
            st.dataframe(df)

            st.markdown(f"**Total Cost:** ${round(total_cost, 2)}")
            st.markdown(f"**Total Value:** ${round(total_value, 2)}")
            st.markdown(f"**Total P&L:** ${round(total_value - total_cost, 2)}")

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
                    json.dump(st.session_state.performance_history, f, indent=4)

            # 📈 Charts
            st.subheader(t['allocation'])
            fig1, ax1 = plt.subplots()
            ax1.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%')
            st.pyplot(fig1)

            st.subheader(t['compare'])
            start_date = "2023-01-01"
            prices = pd.DataFrame()
            for item in portfolio:
                hist = yf.download(item["ticker"], start=start_date)["Adj Close"]
                prices[item["ticker"]] = hist * item["shares"]

            portfolio_value = prices.sum(axis=1)
            sp500 = yf.download("^GSPC", start=start_date)["Adj Close"]
            sp500 = sp500 / sp500.iloc[0] * portfolio_value.iloc[0]

            fig2, ax2 = plt.subplots()
            portfolio_value.plot(ax=ax2, label="Your Portfolio")
            sp500.plot(ax=ax2, label="S&P 500", linestyle="--")
            ax2.legend()
            ax2.set_title("Portfolio vs. S&P 500")
            ax2.grid(True)
            st.pyplot(fig2)

            st.subheader(t['metrics'])
            returns = portfolio_value.pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            max_drawdown = ((portfolio_value / portfolio_value.cummax()) - 1).min()
            days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
            cagr = (portfolio_value[-1] / portfolio_value[0])**(365.0/days) - 1

            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            col2.metric("Max Drawdown", f"{max_drawdown:.2%}")
            col3.metric("CAGR", f"{cagr:.2%}")

            # 🔁 History Chart
            st.subheader(t["history"])
            history_df = pd.DataFrame(st.session_state.performance_history)
            if not history_df.empty:
                history_df["date"] = pd.to_datetime(history_df["date"])
                history_df = history_df.sort_values("date").set_index("date")
                st.line_chart(history_df[["total_value", "pnl"]])
                st.dataframe(history_df)
                csv_export = history_df.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download History CSV", data=csv_export, file_name="performance_history.csv")
            else:
                st.info("📭 No performance history yet.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
