#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[57]:


# Data Import
# PS: all the imported data already rearrange by frequency
import pandas as pd
GDP = pd.read_csv("https://fred.stlouisfed.org/series/GDP/downloaddata/GDP.csv") #quartely #billions of dollar
consumption = pd.read_csv("https://raw.githubusercontent.com/jialis3-2133498/Econ481/main/consumer.csv") #quarterly #billions
investment = pd.read_csv("https://fred.stlouisfed.org/series/GPDIC1/downloaddata/GPDIC1.csv") #quarterly #billions
gov_expenditures = pd.read_csv("https://fred.stlouisfed.org/series/FGEXPND/downloaddata/FGEXPND.csv") #quarterly #billions
net_export = pd.read_csv("https://fred.stlouisfed.org/series/NETEXP/downloaddata/NETEXP.csv") #quarterly #billions
M2 = pd.read_csv("https://raw.githubusercontent.com/jialis3-2133498/Econ481/main/WM2NS.csv") #quarterly #billions
M2V = pd.read_csv("https://fred.stlouisfed.org/series/M2V/downloaddata/M2V.csv") #ratio
gov_securities = "https://raw.githubusercontent.com/jialis3-2133498/Econ481/main/TREAST.csv" # quarterly #millions
federal_funds_rate = "https://raw.githubusercontent.com/jialis3-2133498/Econ481/main/DFF.csv" #daily
ffr = pd.read_csv(federal_funds_rate)
govern_spending = pd.read_csv("https://fred.stlouisfed.org/series/FGEXPND/downloaddata/FGEXPND.csv") # quarterly # billions
income = pd.read_csv('https://fred.stlouisfed.org/series/TNWBSHNO/downloaddata/TNWBSHNO.csv') # quarterly # billions 

# change the units
new_gov_securities = pd.read_csv(gov_securities)
new_gov_securities['TREAST'] = pd.to_numeric(new_gov_securities['TREAST'], errors='coerce')
new_gov_securities['TREAST'] = new_gov_securities['TREAST'] * 0.001
#print(new_gov_securities)
#print(income)

# Research Question 1 -- Function
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset

warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')

def plot_close(df: pd.DataFrame, start: str = '1981-01-01', end: str = '2024-01-01') -> None:
    if 'DATE' not in df.columns:
        print("Error: 'DATE' column not found in DataFrame.")
        return

    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df = df.loc[start:end]
    
    if df.empty:
        print("DataFrame is empty for the specified date range.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['predict'], color='blue', label='Predicted', linestyle='--', alpha=0.5)
    plt.plot(df.index, df['VALUE_y'], color='red', label='True', linestyle='-', alpha=0.5)
    plt.title(f'Predicted GDP VS True GDP from {start} to {end}')
    plt.xlabel('Date')
    plt.ylabel('Billions of Dollars')
    plt.legend()
    plt.grid(True)
    plt.show()


# Multiple Linear Regression -- Function
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def perform_ols_regression(dependent_var, dependent_var_col, independent_vars):
    """
    Perform OLS regression analysis.
    """
    dependent_var['DATE'] = pd.to_datetime(dependent_var['DATE'])
    for var in independent_vars:
        var['DATE'] = pd.to_datetime(var['DATE'])
    
    # Rename the dependent variable column to 'dependent'
    merged_data = dependent_var.rename(columns={dependent_var_col: 'dependent'})
    
    # Merge all independent variables with the dependent variable 
    for i, var in enumerate(independent_vars):
        var = var.rename(columns={var.columns[1]: f'independent_{i}'})
        merged_data = pd.merge(merged_data, var, on='DATE', how='inner')
    
    Y = merged_data['dependent']
    X = merged_data.drop(columns=['DATE', 'dependent'])
    X = sm.add_constant(X)
    model = sm.OLS(Y.astype(float), X.astype(float)).fit()
    
    return model.summary(), X, merged_data


# In[60]:


# Multiple Linear Regression -- Function
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Model 2
def multiple_same_freq(dependent_var, predictors, predictor_names):
    """
    Multiple Linear Regression
    """
    df_dep = dependent_var
    df_dep['DATE'] = pd.to_datetime(df_dep['DATE'])    
    dep_var_name = 'VALUE'
    
    merged_data = df_dep[['DATE', dep_var_name]]

    for predictor, predictor_name in zip(predictors, predictor_names):
        df_pred = predictor
        df_pred['DATE'] = pd.to_datetime(df_pred['DATE'])
        
        if 'VALUE' in df_pred.columns:
            df_pred.columns = ['DATE', predictor_name]
        
        merged_data = pd.merge(merged_data, df_pred, on='DATE', how='inner')

    Y = merged_data[dep_var_name]
    X = merged_data.drop(columns=['DATE', dep_var_name])
    X = sm.add_constant(X)
    model = sm.OLS(Y.astype(float), X.astype(float)).fit()

    return model.summary(), merged_data

# Research Question 1
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
# M2 and GDP
# Merge and calculate the predicted GDP

M2V.rename(columns={'VALUE': 'M2V'}, inplace=True)

predict_GDP = pd.merge(M2, M2V, on='DATE', how='inner')
predict_GDP['WM2NS'] = pd.to_numeric(predict_GDP['WM2NS'], errors='coerce')
predict_GDP['M2V'] = pd.to_numeric(predict_GDP['M2V'], errors='coerce')
predict_GDP['predict'] = predict_GDP['WM2NS']* predict_GDP['M2V']
predict_GDP = pd.merge(predict_GDP, GDP, on='DATE', how='inner')
predict_GDP = pd.merge(predict_GDP, GDP, on='DATE', how='inner')
print(predict_GDP)

plot_close(predict_GDP)


# Research Question 2 -- Model 1
dependent_var = consumption[['DATE', 'PCE']]
dependent_var_col = 'PCE'
independent_vars = [
    ffr[['DATE', 'DFF']],
    income[['DATE', 'VALUE']]
]
model = perform_ols_regression(dependent_var, dependent_var_col, independent_vars)
print(model)


# In[64]:


# Research Question 2 -- Model 2
# Effect of Monetary and Fiscal Policies on the GDP
predictors = [ffr, income, investment, govern_spending]
# We choose the ffr and government_spending to represent the monetary and fiscal policies. 

predictor_names = ['ffr','income','investment', 'govern_spending'] # exclude the net export due to multicolinerity(High FFR will lead to the dollar appreciation and less export)
# frr and income represent the consumer consumption

print(multiple_same_freq(GDP, predictors, predictor_names))



# In[ ]:



# Research Question 3 -- Forcasting

# Select the GDP column (assuming it is named 'GDP' or similar)
gdp = predict_GDP['VALUE_y'] 

# Fit ARIMA model
model = ARIMA(gdp, order=(0, 2, 2))
model_fit = model.fit()

# Forecast
forecast_steps = 50
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=gdp.index[-1], periods=forecast_steps + 1, freq='Q')[1:]

# Extract forecast mean and confidence intervals
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(gdp, label='Observed')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='r')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('GDP (Billions of Dollars)')
plt.title('GDP Forecast')
plt.legend()
plt.show()





