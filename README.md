# Sales Forecasting Using ARIMA

# Project Overview
This project demonstrates sales forecasting using **ARIMA / SARIMA** models in Python.  
It predicts monthly sales based on historical data and evaluates the model’s performance.  

- Dataset: 19 months of sales data (`mock_kaggle.csv`)  
- Columns: `data` (date), `venda` (sales), `estoque`, `preco`  
- Goal: Forecast future sales and visualize predictions  

# Project Structure
sales-forecasting-arima
├── data/
│ └── mock_kaggle.csv
├── plots/
│ ├── historical_sales.png
│ ├── forecast_vs_actual.png
│ └── future_forecast.png
├── output/
│ └── model_summary
├── sale forecasting(ARIMA).py
├── requirements


# How to Run
1. Download or clone the repository.  
2. Install required Python packages:

python forecast.py
pip install -r requirements
Requirements
Python 3.x
Packages:
 pandas
 matplotlib
 pmdarima
 scikit-learn
 numpy

#OUTPUT
================================ SARIMAX Results =================================
Dep. Variable:                      y   No. Observations:                   19
Model:             SARIMAX(0, 0, 0)x(0, 0, 0, 0)   Log Likelihood                -105.545
Date:                Mon, 23 Feb 2026   AIC                            215.090
Time:                        19:27:26   BIC                            216.979
Sample:                    01-31-2014   HQIC                           215.410
                         - 07-31-2015                                         
Covariance Type:                  opg                                         

==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept     76.2632     16.340      4.667      0.000      44.238     108.288
sigma2      3913.2465   1521.351      2.572      0.010     931.453    6895.040
================================================================================
Ljung-Box (L1) (Q):                   0.27   Jarque-Bera (JB):                 1.34
Prob(Q):                              0.60   Prob(JB):                         0.51
Heteroskedasticity (H):               0.81   Skew:                             0.64
Prob(H) (two-sided):                  0.81   Kurtosis:                         2.81
================================================================================

