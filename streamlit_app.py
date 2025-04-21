import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# 🔧 THIS LINE MUST COME RIGHT AFTER IMPORTS
st.set_page_config(page_title="Investment Dashboard", layout="wide")

# --- LANGUAGE SWITCH ---
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

# --- LOGIN SYSTEM ---
EMAIL = st.secrets.get("EMAIL", "amahali.we@gmail.com")
PASSWORD = st.secrets.get("PASSWORD", "changeme")

st.title(f"🔒 {t['login']}")

email_input = st.text_input(t['email'])
password_input = st.text_input(t['password'], type="password")

if email_input == EMAIL and password_input == PASSWORD:
    st.success(t['success'])

    st.title(f"📊 {t['title']}")
    st.subheader(t['tickers'])
    tickers = st.text_input("", "AAPL, TSLA, VOO")
    st.subheader(t['shares'])
    shares = st.text_input("", "10, 5, 7")
    st.subheader(t['buy_prices'])
    buy_prices = st.text_input("", "145, 700, 380")

    if st.button(t['track']):
        tickers = [x.strip().upper() for x in tickers.split(",")]
        shares = [int(x.strip()) for x in shares.split(",")]
        buy_prices = [float(x.strip()) for x in buy_prices.split(",")]

        portfolio = []
        for i in range(len(tickers)):
            portfolio.append({
                'ticker': tickers[i],
                'shares': shares[i],
                'buy_price': buy_prices[i]
            })

        price_data = yf.download(tickers, period="5d")

        # 🛡 Handle unexpected structure
        
        if isinstance(price_data.columns, pd.MultiIndex):
            try:
                if 'Adj Close' in price_data.columns:
                   data = price_data['Adj Close'].dropna().iloc[-1]
                elif 'Close' in price_data.columns:
                    data = price_data['Close'].dropna().iloc[-1]
                else:
                    st.error("❌ Neither 'Adj Close' nor 'Close' was found in the price data.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Could not fetch latest price. Details: {e}")
                st.stop()
            except KeyError as e:
                st.error(f"❌ 'Adj Close' not found. Problem likely with: {str(e)} — check ticker symbols or try again later.")
                st.stop()
            except IndexError:
                st.error("❌ No recent data available. Market might be closed or ticker might be invalid.")
                st.stop()
        elif 'Adj Close' in price_data.columns:
            # Single ticker
            try:
                data = pd.Series({tickers[0]: price_data['Adj Close'].iloc[-1]})
            except Exception as e:
                st.error(f"❌ Failed to extract price data: {e}")
                st.stop()
        else:
            st.error("❌ Price data format not recognized. Check your tickers.")
            st.stop()

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

        st.subheader(t['allocation'])
        fig1, ax1 = plt.subplots()
        ax1.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%')
        st.pyplot(fig1)

        st.subheader(t['compare'])
        start_date = "2023-01-01"
        prices = pd.DataFrame()
        for item in portfolio:
            ticker = item['ticker']
            shares = item['shares']
            
            hist = yf.download(ticker, start=start_date)
            
            if 'Adj Close' in hist.columns:
                data_hist = hist['Adj Close']
            elif 'Close' in hist.columns:
                data_hist = hist['Close']
            else:
                st.error(f"❌ No valid price data found for {ticker}. Please check the symbol.")
                st.stop()
            
            prices[ticker] = data_hist * shares

        portfolio_value = prices.sum(axis=1)
        sp500 = yf.download("^GSPC", start=start_date)['Adj Close']
        sp500 = sp500 / sp500.iloc[0] * portfolio_value.iloc[0]

        fig2, ax2 = plt.subplots()
        portfolio_value.plot(ax=ax2, label="Your Portfolio")
        sp500.plot(ax=ax2, label="S&P 500", linestyle="--")
        ax2.set_title("Portfolio vs. S&P 500")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

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

else:
    st.warning(t['warning'])
