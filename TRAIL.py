# https://github.com/ugursavci/Customer_Lifetime_Value_Prediction/blob/main/Customer_Lifetime_Value.ipynb

# Import required libraries
import lifetimes
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.preprocessing import MinMaxScaler
from lifetimes.plotting import plot_frequency_recency_matrix

# Display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load dataset
df = pd.read_csv('Online_Retail.csv')

# Convert InvoiceDate to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Convert CustomerID to string (avoid missing values)
df.dropna(subset=['CustomerID'], inplace=True)
df['CustomerID'] = df['CustomerID'].astype(str)

# Remove negative values
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Remove returned items (invoices containing 'C')
df = df[~df['InvoiceNo'].astype(str).str.contains("C", na=False)]

# Define function for outlier capping
def find_boundaries(df, variable, q1=0.05, q2=0.95):
    lower_boundary = df[variable].quantile(q1)
    upper_boundary = df[variable].quantile(q2)
    return upper_boundary, lower_boundary

def capping_outliers(df, variable):
    upper_boundary, lower_boundary = find_boundaries(df, variable)
    df[variable] = np.where(df[variable] > upper_boundary, upper_boundary,
                            np.where(df[variable] < lower_boundary, lower_boundary, df[variable]))

# Apply outlier capping
capping_outliers(df, 'UnitPrice')
capping_outliers(df, 'Quantity')

# Filter data for UK only
df = df[df['Country'] == 'United Kingdom']

# Create Total Price column
df['TotalPrice'] = df['UnitPrice'] * df['Quantity']

# Compute summary table for CLTV analysis
clv = summary_data_from_transaction_data(df, 'CustomerID', 'InvoiceDate', 'TotalPrice',
                                         observation_period_end=pd.to_datetime('2011-12-09'))

# Filter customers who purchased more than once
clv = clv[clv['frequency'] > 1]

# Fit BetaGeoFitter model
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(clv['frequency'], clv['recency'], clv['T'])

# Plot Frequency-Recency Matrix
plt.figure(figsize=(12, 9))
plot_frequency_recency_matrix(bgf)
plt.show()

# Predict purchases for next 180 days
t = 180
clv['expected_purc_6_months'] = bgf.conditional_expected_number_of_purchases_up_to_time(t,
                                                                                        clv['frequency'],
                                                                                        clv['recency'],
                                                                                        clv['T'])

# Check correlation between frequency and monetary value
print(clv[['frequency', 'monetary_value']].corr())

# Fit Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(clv["frequency"], clv["monetary_value"])

# Compute CLTV for 6 months
clv['6_Months_CLV'] = ggf.customer_lifetime_value(bgf,
                                                  clv["frequency"],
                                                  clv["recency"],
                                                  clv["T"],
                                                  clv["monetary_value"],
                                                  time=6,
                                                  freq='D',
                                                  discount_rate=0.01)

# Display top customers by CLTV
print(clv.sort_values(by='6_Months_CLV', ascending=False).head())

# Segment customers into 4 groups based on CLTV
clv['Segment'] = pd.qcut(clv['6_Months_CLV'], 4, labels=['Hibernating', 'Need Attention', 'Loyal Customers', 'Champions'])

# Display segment-wise CLTV averages
print(clv.groupby('Segment').mean())
