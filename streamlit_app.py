import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import json
from fpdf import FPDF
from datetime import datetime
import os

history_file_path = "performance_history.json"

# Load performance history from file if exists
if "performance_history" not in st.session_state:
    if os.path.exists("performance_history.json"):
        with open("performance_history.json", "r") as f:
            st.session_state.performance_history = json.load(f)
    else:
        st.session_state.performance_history = []

# 🔧 THIS LINE MUST COME RIGHT AFTER IMPORTS
st.set_page_config(page_title="Investment Dashboard", layout="wide")

# --- Initialize session keys ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

# --- Session storage for daily performance ---
if "performance_history" not in st.session_state:
    st.session_state.performance_history = []

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

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title(f"🔒 {t['login']}")

    with st.form("login_form", clear_on_submit=False):
        email_input = st.text_input(t['email'])
        password_input = st.text_input(t['password'], type="password")
        submit_login = st.form_submit_button("Access Dashboard")

    if submit_login:
        if email_input.strip().lower() == EMAIL.lower() and password_input.strip() == PASSWORD:
            st.session_state.logged_in = True
            st.session_state.show_dashboard = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please try again.")
    st.stop()
# LOGGED IN: Add Logout in Sidebar
with st.sidebar:
    st.markdown("### 👤 Account")
    if st.button("🚪 Log out"):
        st.session_state.logged_in = False
        st.session_state.show_dashboard = False
        st.rerun()

if "history" not in st.session_state:
    if os.path.exists(history_file_path):
        with open(history_file_path, "r") as f:
            st.session_state.history = json.load(f)
            history_df = pd.DataFrame(st.session_state.history)
    else:
        st.session_state.history = []

import os
if os.path.exists(history_file_path):
    st.markdown("### 📄 Raw Performance History JSON")
    with open(history_file_path, "r") as f:
        st.code(f.read(), language="json")

# Show dashboard if logged in
if st.session_state.show_dashboard:
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
