import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import json
from fpdf import FPDF
from datetime import datetime

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
    }
}
t = texts[lang]

# --- LOGIN SYSTEM ---
EMAIL = st.secrets.get("EMAIL", "amahali.we@gmail.com")
PASSWORD = st.secrets.get("PASSWORD", "changeme")

st.title(f"🔒 {t['login']}")

email_input = st.text_input(t['email'])
password_input = st.text_input(t['password'], type="password")
login_button = st.button("Access Dashboard")

# Initialize login state
if 'show_dashboard' not in st.session_state:
    st.session_state['show_dashboard'] = False

# Handle login logic
if login_button:
    if email_input.strip().lower() == EMAIL.lower() and password_input.strip() == PASSWORD:
        st.success(t['success'])
        st.session_state['show_dashboard'] = True
    else:
        st.error("❌ Invalid credentials. Please try again.")
        st.session_state['show_dashboard'] = False

# Show dashboard if logged in
if st.session_state['show_dashboard']:
    st.title(f"📊 {t['title']}")

    portfolio = []

    # Upload a file to pre-fill input
    st.subheader("📥 Load a portfolio (to auto-fill)")
    uploaded_file = st.file_uploader("Upload your portfolio file (JSON or CSV)", type=["json", "csv"], key="upload_prefill")

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".json"):
                portfolio_loaded = json.load(uploaded_file)
            else:
                df_uploaded = pd.read_csv(uploaded_file)
                portfolio_loaded = df_uploaded.to_dict(orient="records")

            # Auto-fill values
            tickers_default = ', '.join([item['ticker'] for item in portfolio_loaded])
            shares_default = ', '.join([str(item['shares']) for item in portfolio_loaded])
            buy_prices_default = ', '.join([str(item['buy_price']) for item in portfolio_loaded])
            st.success("✅ Portfolio loaded and fields pre-filled!")
        except Exception as e:
            st.error("❌ Failed to load portfolio: {}".format(e))
            tickers_default = shares_default = buy_prices_default = ""
    else:
        tickers_default = "AAPL, TSLA, VOO"
        shares_default = "10, 5, 7"
        buy_prices_default = "145, 700, 380"

    # Manual fields (populated with above if available)
    st.subheader(t['tickers'])
    tickers = st.text_input("", tickers_default)
    st.subheader(t['shares'])
    shares = st.text_input("", shares_default)
    st.subheader(t['buy_prices'])
    buy_prices = st.text_input("", buy_prices_default)

    if st.button(t['track']):
        tickers = [x.strip().upper() for x in tickers.split(",")]
        shares = [int(x.strip()) for x in shares.split(",")]
        buy_prices = [float(x.strip()) for x in buy_prices.split(",")]

        if not (len(tickers) == len(shares) == len(buy_prices)):
            st.error("⚠️ The number of tickers, shares, and buy prices must match.")
            st.stop()

        for i in range(len(tickers)):
            portfolio.append({
                'ticker': tickers[i],
                'shares': shares[i],
                'buy_price': buy_prices[i]
            })

        # Add collapsible block to merge more portfolios
        with st.expander("📁 Add another portfolio (merge)", expanded=False):
            uploaded_file_post = st.file_uploader("📂 Upload to merge with current portfolio", type=["json", "csv"], key="upload_merge")
            if uploaded_file_post:
                try:
                    if uploaded_file_post.name.endswith(".json"):
                        additional_portfolio = json.load(uploaded_file_post)
                    else:
                        df_extra = pd.read_csv(uploaded_file_post)
                        additional_portfolio = df_extra.to_dict(orient="records")

                    if isinstance(additional_portfolio, list) and all('ticker' in x and 'shares' in x and 'buy_price' in x for x in additional_portfolio):
                        for item in additional_portfolio:
                            item['ticker'] = item['ticker'].strip().upper()
                        portfolio.extend(additional_portfolio)
                        st.success("✅ Portfolio successfully merged!")
                    else:
                        st.error("⚠️ Invalid format.")
                except Exception as e:
                    st.error("❌ Failed to merge portfolio: {}".format(e))

        # --- Price Fetching ---
        price_data = yf.download([item['ticker'] for item in portfolio], period="5d")

        # Handle single vs. multi-ticker
        if isinstance(price_data.columns, pd.MultiIndex):
            if 'Adj Close' in price_data.columns:
                data = price_data['Adj Close'].dropna().iloc[-1]
            elif 'Close' in price_data.columns:
                data = price_data['Close'].dropna().iloc[-1]
            else:
                st.error("❌ Neither 'Adj Close' nor 'Close' was found in the price data.")
                st.stop()
        else:
            # Single-ticker fallback
            if 'Adj Close' in price_data.columns:
                ticker = [item['ticker'] for item in portfolio][0]
                data = pd.Series({ticker: price_data['Adj Close'].dropna().iloc[-1]})
            elif 'Close' in price_data.columns:
                ticker = [item['ticker'] for item in portfolio][0]
                data = pd.Series({ticker: price_data['Close'].dropna().iloc[-1]})
            else:
                st.error("❌ Price column missing. Please check the ticker.")
                st.stop()

        results = []
        total_value = 0
        total_cost = 0

        for item in portfolio:
            ticker = item['ticker']
            shares = item['shares']
            buy_price = item['buy_price']
            try:
                current_price = data[ticker]
            except KeyError:
                st.error(f"❌ Price for {ticker} not found.")
                st.stop()

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

        st.markdown(f"**Total Cost:** ${round(total_cost, 2)}")
        st.markdown(f"**Total Value:** ${round(total_value, 2)}")
        st.markdown(f"**Total P&L:** ${round(total_value - total_cost, 2)}")

        # --- Pie Chart ---
        st.subheader(t['allocation'])
        fig1, ax1 = plt.subplots()
        ax1.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%')
        st.pyplot(fig1)

        # --- Comparison Chart ---
        st.subheader(t['compare'])
        start_date = "2023-01-01"
        prices = pd.DataFrame()
        for item in portfolio:
            ticker = item['ticker']
            shares = item['shares']
            hist = yf.download(ticker, start=start_date)
            if 'Adj Close' in hist:
                data_hist = hist['Adj Close']
            else:
                data_hist = hist['Close']
            prices[ticker] = data_hist * shares

        portfolio_value = prices.sum(axis=1)
        # --- Daily and Cumulative P&L ---
        daily_pnl = portfolio_value.diff().fillna(0)
        cumulative_pnl = portfolio_value - portfolio_value.iloc[0]
        portfolio_value = prices.sum(axis=1)
        cumulative_pnl = (portfolio_value - total_cost)

        st.subheader("📈 Daily P&L Over Time")
        st.line_chart(daily_pnl)

        st.subheader("📊 Cumulative P&L Over Time")
        st.line_chart(cumulative_pnl)
        
        returns = portfolio_value.pct_change().dropna()
        
        # --- Monthly Returns ---
        monthly_returns = portfolio_value.resample('M').ffill().pct_change().dropna()
        st.subheader("📆 Monthly Returns")
        st.bar_chart(monthly_returns)

        # --- Monthly Volatility (Optional) ---
        monthly_volatility = returns.resample('M').std()
        st.subheader("🌪️ Monthly Volatility")
        st.line_chart(monthly_volatility)


        # --- Compare to Benchmark (S&P 500) ---
        st.subheader(t['compare'])
        sp500_data = yf.download("^GSPC", start=start_date)
        if 'Adj Close' in sp500_data:
            sp500 = sp500_data['Adj Close']
        else:
            sp500 = sp500_data['Close']
        sp500 = sp500 / sp500.iloc[0] * portfolio_value.iloc[0]

        fig2, ax2 = plt.subplots()
        portfolio_value.plot(ax=ax2, label="Your Portfolio")
        sp500.plot(ax=ax2, label="S&P 500", linestyle="--")
        ax2.set_title("Portfolio vs. S&P 500")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # --- Performance Metrics ---
        st.subheader(t['metrics'])
        returns = portfolio_value.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        roll_max = portfolio_value.cummax()
        drawdown = (portfolio_value - roll_max) / roll_max
        max_drawdown = drawdown.min()
        days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
        cagr = (portfolio_value[-1] / portfolio_value[0]) ** (365.0 / days) - 1

        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col2.metric("Max Drawdown", f"{max_drawdown:.2%}")
        col3.metric("CAGR", f"{cagr:.2%}")

        # Save all charts to PNG files
        chart_paths = {}
        
        # Pie chart
        fig1, ax1 = plt.subplots()
        ax1.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%')
        ax1.set_title("Asset Allocation", fontsize=10)
        chart_paths['asset_allocation'] = "/tmp/asset_allocation.png"
        fig1.savefig(chart_paths['asset_allocation'])

        # Daily P&L
        fig2, ax2 = plt.subplots()
        daily_pnl.plot(ax=ax2)
        ax2.set_title("Daily P&L Over Time", fontsize=10)
        chart_paths['daily_pnl'] = "/tmp/daily_pnl.png"
        fig2.savefig(chart_paths['daily_pnl'])

        # Cumulative P&L
        fig3, ax3 = plt.subplots()
        cumulative_pnl.plot(ax=ax3)
        ax3.set_title("Cumulative P&L Over Time", fontsize=10)
        chart_paths['cumulative_pnl'] = "/tmp/cumulative_pnl.png"
        fig3.savefig(chart_paths['cumulative_pnl'])

        # Monthly Returns
        fig4, ax4 = plt.subplots()
        monthly_returns.plot(kind='bar', ax=ax4)
        ax4.set_title("Monthly Returns", fontsize=10)
        chart_paths['monthly_returns'] = "/tmp/monthly_returns.png"
        fig4.savefig(chart_paths['monthly_returns'])

        # Monthly Volatility
        fig5, ax5 = plt.subplots()
        monthly_volatility.plot(kind='bar', ax=ax5, color='orange')
        ax5.set_title("Monthly Volatility", fontsize=10)
        chart_paths['monthly_volatility'] = "/tmp/monthly_volatility.png"
        fig5.savefig(chart_paths['monthly_volatility'])

        # Portfolio vs S&P
        fig6, ax6 = plt.subplots()
        portfolio_value.plot(ax=ax6, label="Your Portfolio")
        sp500.plot(ax=ax6, label="S&P 500", linestyle="--")
        ax6.set_title("Portfolio vs. S&P 500", fontsize=10)
        ax6.legend()
        chart_paths['portfolio_vs_sp500'] = "/tmp/portfolio_vs_sp500.png"
        fig6.savefig(chart_paths['portfolio_vs_sp500'])



        # 📄 Full Styled PDF Export with Charts
        pdf_data = generate_full_pdf(
            df=df,
            total_cost=total_cost,
            total_value=total_value,
            pnl=total_value - total_cost,
            sharpe=sharpe_ratio,
            drawdown=max_drawdown,
            cagr=cagr,
            chart_paths=chart_paths
        )

            
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Portfolio CSV",
            data=csv_data,
            file_name="portfolio_summary.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="📑 Download Portfolio PDF",
            data=pdf_data,
            file_name="portfolio_summary.pdf",
            mime="application/pdf"
        )


