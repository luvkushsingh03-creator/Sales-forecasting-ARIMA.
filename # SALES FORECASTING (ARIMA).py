# SALES FORECASTING (ARIMA)
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------
# Step 1: Load dataset
# -------------------------
csv_path = r"E:/DS/mock_kaggle.csv"  # <-- Change to your CSV path
data = pd.read_csv(csv_path)

print("Columns:", data.columns)

# Map your CSV columns
date_col = 'data'    # Date column
sales_col = 'venda'  # Sales column

# Parse dates and set index
data[date_col] = pd.to_datetime(data[date_col])
data.set_index(date_col, inplace=True)
data[sales_col] = data[sales_col].ffill()  # Forward fill missing sales

# Detect frequency
if len(data) >= 24:
    freq = 'M'  # Monthly data with enough points for SARIMA
else:
    freq = 'M'  # Keep monthly, but will use non-seasonal ARIMA
data = data.asfreq(freq)

# -------------------------
# Step 2: Visualize Historical Sales
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(data[sales_col], label='Sales')
plt.title('Historical Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# -------------------------
# Step 3: Train/Test Split
# -------------------------
if len(data) > 12:
    train = data.iloc[:-12]
    test = data.iloc[-12:]
else:
    train = data.iloc[:-6]
    test = data.iloc[-6:]

# -------------------------
# Step 4: Determine ARIMA type
# -------------------------
if len(train) >= 24:
    # Enough data for seasonal model
    seasonal = True
    m = 12  # Monthly seasonality
else:
    # Small dataset → non-seasonal
    seasonal = False
    m = 1

# -------------------------
# Step 5: Fit Auto ARIMA
# -------------------------
auto_model = pm.auto_arima(train[sales_col],
                           seasonal=seasonal,
                           m=m,
                           stepwise=True,
                           suppress_warnings=True,
                           trace=True)

print(auto_model.summary())

# -------------------------
# Step 6: Forecast Test Period
# -------------------------
forecast = auto_model.predict(n_periods=len(test))
rmse = np.sqrt(mean_squared_error(test[sales_col], forecast))
print(f"Test RMSE: {rmse}")

# -------------------------
# Step 7: Plot Actual vs Forecast
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(train.index, train[sales_col], label='Train')
plt.plot(test.index, test[sales_col], label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('Forecast vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# -------------------------
# Step 8: Forecast Future Sales (Next 12 Months)
# -------------------------
future_periods = 12
future_forecast = auto_model.predict(n_periods=future_periods)
future_index = pd.date_range(data.index[-1]+pd.offsets.MonthBegin(), periods=future_periods, freq='M')

plt.figure(figsize=(10,5))
plt.plot(data.index, data[sales_col], label='Historical Sales')
plt.plot(future_index, future_forecast, label='Future Forecast', color='green')
plt.title('Next 12-Month Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()