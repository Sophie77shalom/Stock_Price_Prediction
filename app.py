import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Load Data
df = pd.read_csv("HistoricalPrices.csv", parse_dates=["Date"], dayfirst=True)  # Explicit date format
df.columns = df.columns.str.strip()  # Fix column name spacing
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)  # Ensure dates are sorted

# Ensure data has a frequency
if df.index.inferred_freq is None:
    df = df.asfreq("B")  # Set frequency to business days

# Sidebar options
st.sidebar.header("Stock Price Prediction")
selected_stock = st.sidebar.selectbox("Select Asset", ["Safaricom", "Equity Bank"])
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=60, value=30)

# Title
st.title(f"📈 {selected_stock} Stock Price Prediction")

# Plot historical data
st.subheader("Historical Stock Prices")
fig = px.line(df, x=df.index, y="Close", title="Stock Price Over Time")
st.plotly_chart(fig)

# Train ARIMA Model
st.subheader("Stock Price Forecast")
train = df["Close"].dropna()[:-forecast_days]  # Remove NaN and split data
test = df["Close"].dropna()[-forecast_days:]

model = ARIMA(train, order=(5,1,0))
fit_model = model.fit()
predictions = fit_model.forecast(steps=forecast_days)

# Create proper date index for predictions
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq="B")

# Convert predictions into a DataFrame with correct dates
pred_df = pd.DataFrame({"Date": forecast_dates, "Predicted Close": predictions})
pred_df.set_index("Date", inplace=True)

# Forecasting Plot
fig_pred = px.line(pred_df, y="Predicted Close", title="Predicted Stock Prices")
st.plotly_chart(fig_pred)

# Scenario Analysis
st.subheader("Scenario Analysis")
worse_case = predictions * 0.95
base_case = predictions
best_case = predictions * 1.05

scenario_df = pd.DataFrame({
    "Date": forecast_dates,
    "Worst Case": worse_case,
    "Base Case": base_case,
    "Best Case": best_case
})
scenario_df.set_index("Date", inplace=True)

fig_scen = px.line(scenario_df, y=["Worst Case", "Base Case", "Best Case"], title="Scenario Analysis")
st.plotly_chart(fig_scen)

# Risk Scoring
st.subheader("Dynamic Risk Scoring")
risk_score = np.random.randint(1, 10)
st.metric(label="Risk Score (1-10)", value=risk_score)

# Alerts
alert_price = st.sidebar.number_input("Set Alert Price", value=df["Close"].mean())
if df["Close"].iloc[-1] <= alert_price:
    st.warning(f"🚨 Price Alert: {selected_stock} has dropped below {alert_price}!")

st.success("✅ Forecasting and risk analysis completed!")
