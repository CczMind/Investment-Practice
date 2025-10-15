# # Lab 2: Basic Pandas for Financial Data Analysis

# ## Learning Objectives
# By the end of this lab, you will be able to:
# - Load and manipulate real financial time series data using pandas
# - Work with proper datetime indexing for OHLCV (Open, High, Low, Close, Volume) data
# - Perform basic data exploration using .info(), .describe(), .head(), and .tail()
# - Apply time-based indexing and slicing with DatetimeIndex
# - Calculate simple financial metrics: returns, moving averages, and basic volatility
# - Handle missing data using forward fill and interpolation methods
# - Use basic time series operations: resampling, rolling windows, and shifting
# - Perform simple correlation analysis between stocks
# - Export results to CSV and Excel formats
# 
# ## Introduction
# This lab introduces you to essential pandas skills using real financial data. You'll learn fundamental data manipulation techniques that form the foundation for financial analysis, using actual stock market data from major companies like Apple, Microsoft, Google, and Tesla.
# 
# ## Basic Concepts
# 
# Before diving into code, let's understand the key concepts we'll work with:
# 
# ### 1. OHLCV Data Structure
# 
# Financial time series data typically includes:
# - **Open**: Opening price for the trading period
# - **High**: Highest price during the trading period
# - **Low**: Lowest price during the trading period
# - **Close**: Closing price for the trading period
# - **Volume**: Number of shares traded
# 
# ### 2. Simple Returns
# 
# **Percentage Change Formula:**
# ```
# Return = (Current Price - Previous Price) / Previous Price × 100
# ```
# 
# In pandas: `df['Close'].pct_change() * 100`
# 
# ### 3. Moving Averages
# 
# **Simple Moving Average:**
# ```
# SMA = Average of last N periods
# ```
# 
# In pandas: `df['Close'].rolling(window=N).mean()`
# 
# ### 4. Basic Volatility
# 
# **Standard Deviation of Returns:**
# ```
# Volatility = Standard deviation of returns over a period
# ```
# 
# In pandas: `df['Close'].pct_change().rolling(window=N).std()`

# ## Setup and Imports
# 
# ### Essential Libraries for Financial Data Analysis
# 
# We'll use the standard data science libraries for basic pandas operations:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output formatting
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 6)
pd.set_option('display.float_format', '{:.4f}'.format)

# Set style for better plots
plt.style.use('default')

print("📊 Basic Pandas Environment Ready")
print(f"📈 Pandas version: {pd.__version__}")
print(f"🔢 NumPy version: {np.__version__}")
print("=" * 50)


# ## Exercise 1: Loading Real Financial Data
# Load real stock market data from CSV files and understand proper datetime indexing.

# 1.1 Load real stock data with proper datetime indexing
# Using the recommended method: parse_dates and index_col together

def load_stock_data(symbol):
    """
    Load stock data with proper datetime indexing.

    Parameters:
    symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')

    Returns:
    pd.DataFrame: OHLCV data with DatetimeIndex
    """
    file_path = f'./{symbol}_stock_data.csv'

    # Load with proper datetime parsing and indexing
    #CODE HERE - Use pd.read_csv with index_col='Date' and parse_dates=True
    df = pd.read_csv(f"{file_path}", index_col = "Date", parse_dates = True)

    # Keep only the essential OHLCV columns
    #CODE HERE - Select only 'Open', 'High', 'Low', 'Close', 'Volume' columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    return df

# Load data for major tech stocks
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
stock_data = {}

print("📊 Loading Real Stock Market Data...")
print("=" * 40)

for symbol in symbols:
    try:
        #CODE HERE - Use the load_stock_data function to load each stock
        stock_data[symbol] = load_stock_data(symbol)
        print(f"✅ {symbol}: {len(stock_data[symbol])} trading days loaded")
        print(f"   📅 Period: {stock_data[symbol].index.min().strftime('%Y-%m-%d')} to {stock_data[symbol].index.max().strftime('%Y-%m-%d')}")
    except FileNotFoundError:
        print(f"❌ {symbol}: Data file not found")
        continue

print(f"\n🎯 Successfully loaded {len(stock_data)} stocks")

# Display sample data structure for AAPL
print(f"\n📈 Sample AAPL Data (First 5 rows):")
#CODE HERE - Print the first 5 rows of AAPL data
print(stock_data["AAPL"].head(5))

print(f"\n📊 AAPL Data Info:")
#CODE HERE - Use .info() method on AAPL data
stock_data["AAPL"].info()

print(f"\n📏 AAPL Data Shape: {stock_data['AAPL'].shape}")
print(f"📅 Index Type: {type(stock_data['AAPL'].index)}")

# 1.2 Create a combined close prices DataFrame for easy comparison
#CODE HERE - Create an empty DataFrame called close_prices
close_prices = pd.DataFrame()
for symbol in symbols:
    if symbol in stock_data:
        #CODE HERE - Add the 'Close' column from each stock to close_prices DataFrame
        close_prices[symbol] = stock_data[symbol]["Close"]

print(f"\n💰 Combined Close Prices (Last 5 days):")
#CODE HERE - Print the last 5 rows of close_prices
print(close_prices.tail(5))
print(f"\nShape: {close_prices.shape}")
print(f"Column names: {list(close_prices.columns)}")


# ## Exercise 2: Basic Data Exploration
# Explore the structure and characteristics of our financial datasets using fundamental pandas methods.

# 2.1 Basic DataFrame exploration methods
print("🔍 Basic Data Exploration")
print("=" * 40)

# Explore AAPL stock data using fundamental pandas methods
#CODE HERE - Assign AAPL data from stock_data dictionary to aapl_data variable
aapl_data = stock_data["AAPL"]

print("📊 AAPL Data Overview:")
print(f"Dataset shape: {aapl_data.shape}")
print(f"Number of rows: {len(aapl_data)}")
print(f"Number of columns: {len(aapl_data.columns)}")
print(f"Column names: {list(aapl_data.columns)}")

print(f"\n📈 First 5 rows (.head()):")
#CODE HERE - Use .head() method to show first 5 rows
print(aapl_data.head(5))

print(f"\n📉 Last 5 rows (.tail()):")
#CODE HERE - Use .tail() method to show last 5 rows
print(aapl_data.tail(5))

print(f"\n📊 Basic statistics (.describe()):")
#CODE HERE - Use .describe() method to show statistical summary
print(aapl_data.describe())

print(f"\n🔍 Data types and info (.info()):")
#CODE HERE - Use .info() method
aapl_data.info()

# 2.2 Check for missing values
print(f"\n🔍 Missing Values Analysis:")
print("Missing values per column:")
#CODE HERE - Use .isnull().sum() to count missing values per column
missing_values = aapl_data.isnull().sum()
print(missing_values)

print(f"\nTotal missing values: {aapl_data.isnull().sum().sum()}")
print(f"Percentage of missing data: {(aapl_data.isnull().sum().sum() / (len(aapl_data) * len(aapl_data.columns))) * 100:.2f}%")

# 2.3 Basic data quality checks
print(f"\n✅ Data Quality Checks:")

# Check if High >= Low (should always be true)
#CODE HERE - Create boolean check that High >= Low for all rows using .all()
high_low_check = aapl_data["High"].all() >= aapl_data["Low"].all()
print(f"High >= Low check: {'✅ PASS' if high_low_check else '❌ FAIL'}")

# Check if Open and Close are between High and Low
#CODE HERE - Check if Open is between Low and High using boolean conditions
open_range_check = ((aapl_data['Open'] >= aapl_data['Low']) & (aapl_data['Open'] <= aapl_data['High'])).all()
#CODE HERE - Check if Close is between Low and High using boolean conditions
close_range_check = ((aapl_data['Close'] >= aapl_data['Low']) & (aapl_data['Close'] <= aapl_data['High'])).all()

print(f"Open within High-Low range: {'✅ PASS' if open_range_check else '❌ FAIL'}")
print(f"Close within High-Low range: {'✅ PASS' if close_range_check else '❌ FAIL'}")

# Check for zero or negative values
#CODE HERE - Check that all OHLC prices are positive using > 0
zero_negative_check = (aapl_data[['Open', 'High', 'Low', 'Close']] > 0).all().all()
print(f"No zero/negative prices: {'✅ PASS' if zero_negative_check else '❌ FAIL'}")

# 2.4 Date index exploration
print(f"\n📅 Date Index Analysis:")
print(f"Start date: {aapl_data.index.min()}")
print(f"End date: {aapl_data.index.max()}")
print(f"Date range: {(aapl_data.index.max() - aapl_data.index.min()).days} days")
print(f"Index frequency: {aapl_data.index.freq}")  # Will be None for irregular data
print(f"Unique dates: {len(aapl_data.index.unique())}")
print(f"Duplicate dates: {aapl_data.index.duplicated().sum()}")

# 2.5 Compare all stocks basic statistics
print(f"\n📊 All Stocks Price Summary (Close Prices):")
#CODE HERE - Use .describe() on close_prices DataFrame and round to 2 decimals
print(close_prices.describe().round(2))


# ## Exercise 3: Time-Based Indexing and Slicing
# Learn to filter and slice financial data using datetime indexing - a crucial skill for financial analysis.

# 3.1 Time-based filtering using boolean indexing
print("📅 Time-Based Indexing and Slicing")
print("=" * 40)

# Using AAPL data for demonstration
aapl = stock_data['AAPL']

# Method 1: Boolean indexing with datetime components
print("📊 Method 1: Boolean indexing with datetime components")

# Filter data for specific year
#CODE HERE - Filter data where index.year equals 2024
data_2024 = aapl[aapl.index.year == 2024]
print(f"Data for 2024: {len(data_2024)} trading days")
print(data_2024.head())

# Filter data for specific month and year
#CODE HERE - Filter data where index.year equals 2024 AND index.month equals 1
data_jan_2024 = aapl[(aapl.index.year == 2024) & (aapl.index.month == 1)]
print(f"\nData for January 2024: {len(data_jan_2024)} trading days")
print(data_jan_2024.head())

# Filter data for multiple years
#CODE HERE - Filter data where index.year is greater than or equal to 2023
data_recent = aapl[aapl.index.year >= 2023]
print(f"\nData from 2023 onwards: {len(data_recent)} trading days")

# Method 2: Date range slicing using .loc
print(f"\n📊 Method 2: Date range slicing with .loc")

# Slice using complete date strings
#CODE HERE - Use .loc to slice from '2024-01-01' to '2024-01-31'
jan_2024_slice = aapl.loc["2024-01-01":"2024-01-31"]
print(f"January 2024 using .loc slice: {len(jan_2024_slice)} days")
print(jan_2024_slice.head())

# Slice for a specific quarter
#CODE HERE - Use .loc to slice Q1 2024 ('2024-01-01' to '2024-03-31')
q1_2024 = aapl.loc["2024-01-01":"2024-03-31"]
print(f"\nQ1 2024: {len(q1_2024)} trading days")

# Slice for last 30 trading days
#CODE HERE - Use .tail(30) to get last 30 trading days
last_30_days = aapl.tail(30)
print(f"\nLast 30 trading days:")
print(last_30_days[['Close']].head())

# 3.2 Advanced time-based operations
print(f"\n📈 Advanced Time-Based Operations:")

# Group by year and calculate basic statistics
#CODE HERE - Group by aapl.index.year and aggregate Close prices with ['first', 'last', 'min', 'max', 'mean']
yearly_stats = aapl.groupby(aapl.index.year)['Close'].agg(['first', 'last', 'min', 'max', 'mean'])

yearly_stats.columns = ['First_Price', 'Last_Price', 'Min_Price', 'Max_Price', 'Avg_Price']
#CODE HERE - Calculate year return percentage: ((Last_Price - First_Price) / First_Price * 100)
yearly_stats['Year_Return_%'] = ((yearly_stats['Last_Price'] - yearly_stats['First_Price']) / yearly_stats['First_Price'] * 100).round(2)

print("Yearly Statistics:")
print(yearly_stats)

# Group by month in 2024
#CODE HERE - Group data_2024 by month and aggregate Close prices with ['mean', 'min', 'max']
monthly_2024 = data_2024.groupby(data_2024.index.month)['Close'].agg(['mean', 'min', 'max'])
monthly_2024.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_2024)]
print(f"\nMonthly Statistics for 2024:")
print(monthly_2024.round(2))

# 3.3 Weekday analysis
print(f"\n📊 Weekday Analysis:")

# Add weekday information
aapl_with_weekday = aapl.copy()
#CODE HERE - Add 'Weekday' column using index.day_name()
aapl_with_weekday['Weekday'] = aapl_with_weekday.index.day_name()
#CODE HERE - Add 'Returns' column using Close.pct_change() * 100
aapl_with_weekday['Returns'] = aapl_with_weekday["Close"].pct_change() * 100

# Calculate average returns by weekday
#CODE HERE - Group by 'Weekday' and aggregate Close and Returns columns
weekday_analysis = aapl_with_weekday.groupby("Weekday").agg({
    "Close": "mean",
    "Returns": "mean"
}).round(2)
print("Average performance by weekday:")
print(weekday_analysis)

# Show trading day frequency (should be Mon-Fri for stock data)
#CODE HERE - Use .value_counts() on Weekday column
weekday_counts = aapl_with_weekday["Weekday"].value_counts()
print(f"\nTrading day frequency:")
print(weekday_counts)


# ## Exercise 4: Basic Financial Calculations
# Calculate fundamental financial metrics: returns, moving averages, and simple volatility measures.

# 4.1 Calculate simple daily returns
print("💹 Basic Financial Calculations")
print("=" * 40)

# Using AAPL data for demonstration
aapl = stock_data['AAPL']

# Calculate daily percentage returns using .pct_change()
#CODE HERE - Calculate daily returns using .pct_change() on Close prices and multiply by 100
aapl_returns = aapl["Close"].pct_change() * 100
print("📊 Daily Returns Calculation:")
print("First 10 daily returns (%):")
print(aapl_returns.head(10).round(4))

print(f"\nReturns statistics:")
print(aapl_returns.describe().round(4))

# Calculate returns for all stocks
print(f"\n📈 Returns for All Stocks:")
#CODE HERE - Calculate daily returns for close_prices using .pct_change() * 100
daily_returns = close_prices.pct_change() * 100
print("Average daily returns (%):")
#CODE HERE - Calculate mean of daily_returns and round to 4 decimals
print(daily_returns.mean().round(4))

print(f"\nDaily volatility (standard deviation of returns):")
#CODE HERE - Calculate standard deviation of daily_returns and round to 4 decimals
print(daily_returns.std().round(4))

# 4.2 Simple Moving Averages
print(f"\n📊 Simple Moving Averages:")

# Calculate different moving averages for AAPL
#CODE HERE - Calculate 5-day moving average using .rolling(window=5).mean()
aapl_sma_5 = aapl["Close"].rolling(window = 5).mean()
#CODE HERE - Calculate 20-day moving average using .rolling(window=20).mean()
aapl_sma_20 = aapl["Close"].rolling(window = 20).mean()
#CODE HERE - Calculate 50-day moving average using .rolling(window=50).mean()
aapl_sma_50 = aapl["Close"].rolling(window = 50).mean()

# Create a summary DataFrame
#CODE HERE - Create DataFrame with Close, SMA_5, SMA_20, SMA_50 columns
ma_summary = pd.DataFrame({
    'Close': aapl["Close"],
    'SMA_5': aapl_sma_5,
    'SMA_20': aapl_sma_20,
    'SMA_50': aapl_sma_50
})

print("AAPL Moving Averages (Last 10 days):")
print(ma_summary.tail(10).round(2))

# 4.3 Basic volatility calculations
print(f"\n📊 Basic Volatility Measures:")

# Rolling volatility (20-day window)
#CODE HERE - Calculate 20-day rolling standard deviation of aapl_returns
aapl_vol_20 = aapl_returns.rolling(window = 20).std()
print("20-day rolling volatility (Last 10 values):")
print(aapl_vol_20.tail(10).round(4))

# Calculate volatility for all stocks
#CODE HERE - Calculate 20-day rolling standard deviation for daily_returns
all_volatility = daily_returns.rolling(window = 20).std()
print(f"\nCurrent 20-day volatility for all stocks:")
print(all_volatility.tail(1).round(4))

# 4.4 Simple correlation analysis
print(f"\n🔗 Basic Correlation Analysis:")

# Calculate correlation matrix for close prices
#CODE HERE - Calculate correlation matrix using .corr() method
price_correlation = close_prices.corr()
print("Correlation matrix (Close prices):")
print(price_correlation.round(4))

# Calculate correlation matrix for returns
#CODE HERE - Calculate correlation matrix for daily_returns
return_correlation = daily_returns.corr()
print(f"\nCorrelation matrix (Daily returns):")
print(return_correlation.round(4))

# 4.5 Calculate cumulative returns
print(f"\n📈 Cumulative Returns:")

# Calculate cumulative returns (compound growth)
#CODE HERE - Calculate cumulative returns using (1 + daily_returns/100).cumprod() - 1
cumulative_returns = (1 + daily_returns / 100).cumprod() - 1
print("Total cumulative returns (%):")
print((cumulative_returns.tail(1) * 100).round(2))

# Show investment growth of $10,000
initial_investment = 10000
#CODE HERE - Calculate final values: initial_investment * (1 + cumulative_returns.tail(1))
final_values = initial_investment * (1 + cumulative_returns.tail(1))
print(f"\nGrowth of $10,000 investment:")
for symbol in final_values.columns:
    print(f"{symbol}: ${final_values[symbol].iloc[0]:,.2f}")

# 4.6 Price ranges and performance metrics
print(f"\n📏 Price Range Analysis:")

# Calculate 52-week highs and lows (approximate using available data)
price_ranges = pd.DataFrame()
for symbol in close_prices.columns:
    #CODE HERE - Get last 252 trading days (approximately 1 year) using .tail(252)
    recent_data = close_prices[symbol].tail(252)
    price_ranges[symbol] = [
        recent_data.min(),
        recent_data.max(),
        recent_data.iloc[-1],  # Current price
        ((recent_data.iloc[-1] - recent_data.min()) / recent_data.min() * 100),  # % from low
        ((recent_data.max() - recent_data.iloc[-1]) / recent_data.max() * 100)   # % from high
    ]

price_ranges.index = ['52W_Low', '52W_High', 'Current', '%_From_Low', '%_From_High']
print("52-Week Performance Summary:")
print(price_ranges.round(2))


# ## Exercise 5: Time Series Operations
# Learn basic time series operations: resampling, rolling windows, and shifting.

# 5.1 Basic resampling operations
print("⏰ Time Series Operations")
print("=" * 40)

# Using AAPL data for demonstration
aapl = stock_data['AAPL']

# Resample to weekly data (using 'W' instead of 'W-FRI')
print("📊 Resampling to Different Frequencies:")

# Weekly data - take the last price of each week
#CODE HERE - Resample Close prices to weekly frequency using .resample('W').last()
weekly_prices = aapl['Close'].resample('W').last()
print(f"Weekly prices (last 10 weeks):")
print(weekly_prices.tail(10).round(2))

# Monthly data - take the last price of each month
#CODE HERE - Resample Close prices to monthly frequency using .resample('M').last()
monthly_prices = aapl.resample("M").last()
print(f"\nMonthly prices (last 6 months):")
print(monthly_prices.tail(6).round(2))

# Weekly summary statistics
#CODE HERE - Resample AAPL data to weekly and aggregate with specified functions
weekly_summary = aapl.resample('W').agg({
    'Open': 'first',    # First price of the week
    'High': 'max',      # Highest price of the week
    'Low': 'min',       # Lowest price of the week
    'Close': 'last',    # Last price of the week
    'Volume': 'sum'     # Total volume for the week
})

print(f"\nWeekly OHLCV summary (last 5 weeks):")
print(weekly_summary.tail(5).round(2))

# 5.2 Rolling window operations
print(f"\n📊 Rolling Window Operations:")

# Basic rolling statistics
#CODE HERE - Create DataFrame with rolling statistics for AAPL Close prices
rolling_stats = pd.DataFrame({
    'Price': aapl['Close'],
    'SMA_10': aapl["Close"].rolling(window = 10).mean(),      # 10-day simple moving average
    'SMA_30': aapl["Close"].rolling(window = 30).mean(),      # 30-day simple moving average
    'Rolling_Min_20': aapl["Close"].rolling(window = 20).min(),   # 20-day rolling minimum
    'Rolling_Max_20': aapl["Close"].rolling(window = 20).max(),   # 20-day rolling maximum
    'Rolling_Std_20': aapl["Close"].rolling(window = 20).std()    # 20-day rolling standard deviation
})

print("Rolling statistics (last 10 days):")
print(rolling_stats.tail(10).round(2))

# Rolling volume analysis
#CODE HERE - Create DataFrame with volume analysis
volume_stats = pd.DataFrame({
    'Volume': aapl['Volume'],
    'Avg_Volume_20': aapl["Volume"].rolling(window = 20).mean(),     # 20-day average volume
    'Volume_Ratio': aapl["Volume"] / aapl["Volume"].rolling(window = 20).mean()       # Current volume / 20-day average volume
})

print(f"\nVolume analysis (last 10 days):")
print(volume_stats.tail(10).round(2))

# 5.3 Expanding window operations
print(f"\n📈 Expanding Window Operations:")

# Expanding statistics (cumulative from start)
#CODE HERE - Create DataFrame with expanding statistics for AAPL Close prices
expanding_stats = pd.DataFrame({
    'Price': aapl['Close'],
    'Expanding_Mean': aapl['Close'].expanding().mean(),    # Expanding mean
    'Expanding_Min': aapl['Close'].expanding().min(),     # Expanding minimum
    'Expanding_Max': aapl['Close'].expanding().max(),     # Expanding maximum
    'Expanding_Std': aapl['Close'].expanding().std()      # Expanding standard deviation
})

print("Expanding statistics (last 10 days):")
print(expanding_stats.tail(10).round(2))

# 5.4 Shifting operations for lagged analysis
print(f"\n🔄 Shifting Operations:")

# Create lagged variables
#CODE HERE - Create DataFrame with shifted Close prices
lagged_data = pd.DataFrame({
    'Close': aapl['Close'],
    'Close_Lag1': aapl['Close'].shift(1),        # Previous day using .shift(1)
    'Close_Lag5': aapl['Close'].shift(5),        # 5 days ago using .shift(5)
    'Close_Lead1': aapl['Close'].shift(-1)        # Next day using .shift(-1)
})

# Calculate price changes
#CODE HERE - Calculate daily change: Close - Close_Lag1
lagged_data['Daily_Change'] = lagged_data["Close"] - lagged_data["Close_Lag1"]
#CODE HERE - Calculate 5-day change: Close - Close_Lag5
lagged_data['5Day_Change'] = lagged_data["Close"] - lagged_data["Close_Lag5"]

print("Lagged price analysis (showing days 10-15):")
print(lagged_data.iloc[10:15].round(2))

# 5.5 Multi-stock resampling
print(f"\n📊 Multi-Stock Resampling:")

# Resample all stocks to monthly frequency
#CODE HERE - Resample close_prices to monthly using .resample('M').last()
monthly_closes = close_prices.resample("M").last()
print("Monthly closing prices (last 6 months):")
print(monthly_closes.tail(6).round(2))

# Calculate monthly returns
#CODE HERE - Calculate monthly returns using .pct_change() * 100
monthly_returns = monthly_closes.pct_change() * 100
print(f"\nMonthly returns (%):")
print(monthly_returns.tail(6).round(2))

# 5.6 Business day operations
print(f"\n📅 Business Day Analysis:")

# Count trading days per month in 2024
if len(aapl[aapl.index.year == 2024]) > 0:
    #CODE HERE - Group 2024 data by month and count trading days using .size()
    trading_days_2024 = aapl[aapl.index.year == 2024].groupby(aapl[aapl.index.year == 2024].index.month).size()
    trading_days_2024.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(trading_days_2024)]
    print("Trading days per month in 2024:")
    print(trading_days_2024)

# Calculate average daily volume by month
#CODE HERE - Group AAPL data by month and calculate mean volume
avg_volume_monthly = aapl.groupby(aapl.index.month)["Volume"].mean()
avg_volume_monthly.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print(f"\nAverage daily volume by month:")
print((avg_volume_monthly / 1000000).round(2))  # Convert to millions


# ## Exercise 6: Missing Data Handling
# Learn to identify and handle missing values in financial time series data.

# 6.1 Identify missing data
print("🔍 Missing Data Handling")
print("=" * 40)

# Check for missing values in our datasets
print("📊 Missing Data Assessment:")

# Check each stock individually
for symbol in symbols:
    if symbol in stock_data:
        #CODE HERE - Count total missing values using .isnull().sum().sum()
        missing_count = stock_data[symbol].isnull().sum().sum()
        total_cells = len(stock_data[symbol]) * len(stock_data[symbol].columns)
        missing_pct = (missing_count / total_cells) * 100
        print(f"{symbol}: {missing_count} missing values ({missing_pct:.2f}%)")

# Check combined close prices
print(f"\nCombined close prices missing data:")
#CODE HERE - Count missing values per column using .isnull().sum()
print(close_prices.isnull().sum())

# 6.2 Create sample data with missing values for demonstration
print(f"\n🛠️ Missing Data Handling Techniques:")

# Create a copy of AAPL data and introduce some missing values for demonstration
#CODE HERE - Create a copy of AAPL Close prices
demo_data = stock_data['AAPL']['Close'].copy()

# Introduce missing values at specific positions
demo_data.iloc[100:103] = np.nan  # Remove 3 consecutive days
demo_data.iloc[200] = np.nan      # Remove single day
demo_data.iloc[250:252] = np.nan  # Remove 2 consecutive days

print(f"Created demo data with missing values:")
print(f"Original data points: {len(stock_data['AAPL'])}")
print(f"Missing values introduced: {demo_data.isnull().sum()}")

# Show the missing data areas
print(f"\nMissing data around position 100:")
print(demo_data.iloc[98:105])

# 6.3 Forward Fill (ffill) method
print(f"\n📈 Forward Fill Method:")
#CODE HERE - Use .fillna(method='ffill') to forward fill missing values
demo_ffill = demo_data.fillna(method = "ffill")
print("Forward fill - carries last known value forward:")
print(demo_ffill.iloc[98:105].round(2))

# 6.4 Backward Fill (bfill) method
print(f"\n📉 Backward Fill Method:")
#CODE HERE - Use .fillna(method='bfill') to backward fill missing values
demo_bfill = demo_data.fillna(method = "bfill")
print("Backward fill - uses next known value:")
print(demo_bfill.iloc[98:105].round(2))

# 6.5 Linear Interpolation
print(f"\n📊 Linear Interpolation Method:")
#CODE HERE - Use .interpolate(method='linear') to interpolate missing values
demo_interpolate = demo_data.interpolate(method = "linear")
print("Linear interpolation - estimates values between known points:")
print(demo_interpolate.iloc[98:105].round(2))

# 6.6 Compare different methods
print(f"\n🔍 Comparison of Methods:")
#CODE HERE - Create DataFrame comparing all methods
comparison_df = pd.DataFrame({
    'Original': demo_data.iloc[98:105],
    'Forward_Fill': demo_ffill.iloc[98:105],
    'Backward_Fill': demo_bfill.iloc[98:105],
    'Interpolation': demo_interpolate.iloc[98:105]
})

print(comparison_df.round(2))

# 6.7 Handle missing data in returns calculation
print(f"\n💹 Missing Data in Returns Calculation:")

# Calculate returns with missing data
#CODE HERE - Calculate returns using .pct_change() * 100 on demo_data
demo_returns = demo_data.pct_change() * 100
print(f"Returns with missing data (around position 100):")
print(demo_returns.iloc[98:105].round(4))

# Calculate returns after forward fill
#CODE HERE - Calculate returns using .pct_change() * 100 on demo_ffill
demo_returns_filled = demo_ffill.pct_change() * 100
print(f"\nReturns after forward fill:")
print(demo_returns_filled.iloc[98:105].round(4))

# 6.8 Best practices for financial data
print(f"\n✅ Best Practices for Financial Missing Data:")

print("1. Forward Fill (ffill): Most common for prices")
print("   - Assumes last known price continues until new information")
print("   - Conservative approach for price data")

print(f"\n2. Interpolation: Good for smooth data")
print("   - Estimates missing values based on trend")
print("   - Use with caution in volatile markets")

print(f"\n3. Dropping missing values:")
# Show effect of dropping missing values
#CODE HERE - Drop missing values using .dropna()
demo_dropped = demo_data.dropna()
print(f"   - Original length: {len(demo_data)}")
print(f"   - After dropping NaN: {len(demo_dropped)}")
print(f"   - Data points lost: {len(demo_data) - len(demo_dropped)}")

# 6.9 Handling missing data in multi-stock analysis
print(f"\n📊 Multi-Stock Missing Data:")

# Check alignment of dates across stocks
#CODE HERE - Get dates with all stocks present using .dropna().index
common_dates = close_prices.dropna().index
print(f"Dates with all stocks present: {len(common_dates)}")
print(f"Total trading days: {len(close_prices)}")
print(f"Days with any missing data: {len(close_prices) - len(common_dates)}")

# Forward fill all stocks
#CODE HERE - Forward fill close_prices using .fillna(method='ffill')
close_prices_filled = close_prices.fillna(method = "ffill")
print(f"\nMissing values after forward fill:")
print(close_prices_filled.isnull().sum())

# Calculate correlation with and without missing data handling
print(f"\n🔗 Correlation Impact of Missing Data:")
#CODE HERE - Calculate correlation matrices
corr_with_missing = close_prices.corr()
corr_filled = close_prices_filled.corr()

print("Correlation difference (AAPL vs MSFT):")
print(f"With missing data: {corr_with_missing.loc['AAPL', 'MSFT']:.4f}")
print(f"After forward fill: {corr_filled.loc['AAPL', 'MSFT']:.4f}")
print(f"Difference: {abs(corr_with_missing.loc['AAPL', 'MSFT'] - corr_filled.loc['AAPL', 'MSFT']):.4f}")


# ## Exercise 7: Basic Grouping and Aggregation

# 7.1 Group by time periods
print("📊 Basic Grouping and Aggregation")
print("=" * 40)

# Using AAPL data for demonstration
aapl = stock_data['AAPL']

# Group by year and calculate basic statistics
print("📅 Yearly Analysis:")
#CODE HERE - Group by year and aggregate OHLCV data with specified functions
yearly_stats = aapl.groupby(aapl.index.year).agg({
    'Open': 'first',      # First trading day price
    'Close': 'last',      # Last trading day price
    'High': 'max',        # Highest price of the year
    'Low': 'min',         # Lowest price of the year
    'Volume': 'mean'      # Average daily volume
}).round(2)

# Add yearly return calculation
#CODE HERE - Calculate yearly return percentage
yearly_stats['Return_%'] = ((yearly_stats["Close"] - yearly_stats["Open"]) / yearly_stats["Open"] * 100).round(2)

print("AAPL Yearly Statistics:")
print(yearly_stats)

# Group by month (for recent data)
print(f"\n📅 Monthly Analysis (2024):")
if len(aapl[aapl.index.year == 2024]) > 0:
    #CODE HERE - Group 2024 data by month and aggregate Close and Volume
    monthly_2024 = aapl[aapl.index.year == 2024].groupby(aapl[aapl.index.year == 2024].index.month).agg({
        'Close': ['first', 'last', 'min', 'max', 'mean'],
        'Volume': 'mean'
    }).round(2)

    # Flatten column names
    monthly_2024.columns = ['_'.join(col).strip() for col in monthly_2024.columns]
    monthly_2024.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_2024)]

    print(monthly_2024)

# 7.2 Group by weekday
print(f"\n📅 Weekday Analysis:")
aapl_copy = aapl.copy()
#CODE HERE - Add Weekday column using .day_name()
aapl_copy['Weekday'] = aapl_copy.index.day_name()
#CODE HERE - Add Returns column using .pct_change() * 100
aapl_copy['Returns'] = aapl_copy["Close"].pct_change() * 100

#CODE HERE - Group by weekday and aggregate Close, Returns, and Volume
weekday_analysis = aapl_copy.groupby('Weekday').agg({
    'Close': 'mean',
    'Returns': ['mean', 'std'],
    'Volume': 'mean'
}).round(4)

print("Average performance by weekday:")
print(weekday_analysis)

# 7.3 Multi-stock comparison
print(f"\n📊 Multi-Stock Comparison:")

# Calculate basic metrics for all stocks
stock_comparison = pd.DataFrame()
for symbol in symbols:
    if symbol in stock_data:
        stock = stock_data[symbol]
        #CODE HERE - Calculate returns using .pct_change() * 100
        returns = stock["Close"].pct_change() * 100

        stock_comparison[symbol] = [
            stock['Close'].iloc[-1],                    # Current price
            stock['Close'].mean(),                      # Average price
            returns.mean(),                             # Average return
            returns.std(),                              # Volatility
            stock['Volume'].mean() / 1000000,           # Average volume (millions)
            (stock['Close'].iloc[-1] / stock['Close'].iloc[0] - 1) * 100  # Total return
        ]

stock_comparison.index = ['Current_Price', 'Avg_Price', 'Avg_Return_%', 'Volatility_%', 'Avg_Volume_M', 'Total_Return_%']
print("Stock comparison summary:")
print(stock_comparison.round(2))

# 7.4 Performance ranking
print(f"\n🏆 Performance Rankings:")

# Rank stocks by different metrics
#CODE HERE - Create rankings DataFrame with rank() method for different metrics
rankings = pd.DataFrame({
    'Total_Return_%': stock_comparison.loc["Total_Return_%"].rank(ascending = False),        # Rank by total return (descending)
    'Avg_Return_%': stock_comparison.loc["Avg_Return_%"].rank(ascending = False),          # Rank by average return (descending)
    'Volatility_%': stock_comparison.loc["Volatility_%"].rank(ascending = True),          # Rank by volatility (ascending - lower is better)
    'Volume_Rank': stock_comparison.loc["Avg_Volume_M"].rank(ascending = False)            # Rank by volume (descending)
})

print("Stock rankings (1 = best):")
print(rankings.astype(int))

# 7.5 Calculate portfolio statistics
print(f"\n💼 Simple Portfolio Analysis:")

# Equal-weighted portfolio
equal_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights for 4 stocks
#CODE HERE - Calculate portfolio returns using equal weights
portfolio_returns = (close_prices.pct_change() * equal_weights).sum(axis = 1) * 100

portfolio_stats = {
    'Portfolio_Avg_Return_%': portfolio_returns.mean(),
    'Portfolio_Volatility_%': portfolio_returns.std(),
    'Portfolio_Total_Return_%': ((close_prices.iloc[-1] * equal_weights).sum() / (close_prices.iloc[0] * equal_weights).sum() - 1) * 100
}

print("Equal-weighted portfolio performance:")
for key, value in portfolio_stats.items():
    print(f"{key}: {value:.4f}")

# Compare portfolio vs individual stocks
print(f"\n📊 Portfolio vs Individual Stocks:")
#CODE HERE - Create comparison table with portfolio and individual stock metrics
comparison_table = pd.DataFrame({
    'Avg_Return_%': [portfolio_stats['Portfolio_Avg_Return_%']] + [stock_comparison.loc['Avg_Return_%', symbol] for symbol in symbols],
    'Volatility_%': [portfolio_stats['Portfolio_Volatility_%']] + [stock_comparison.loc['Volatility_%', symbol] for symbol in symbols],
    'Total_Return_%': [portfolio_stats['Portfolio_Total_Return_%']] + [stock_comparison.loc['Total_Return_%', symbol] for symbol in symbols]
}, index=['Portfolio'] + symbols)

print(comparison_table.round(2))


# ## Exercise 8: Data Export and File Operations
# Learn to save your analysis results in various formats for further use.

# 8.1 Export data to CSV files
print("💾 Data Export and File Operations")
print("=" * 40)

# Export individual stock data
print("📁 Exporting individual datasets:")
for symbol in symbols:
    if symbol in stock_data:
        filename = f'./Lab2_{symbol}_analysis.csv'

        # Create analysis summary for each stock
        stock = stock_data[symbol]
        #CODE HERE - Calculate returns using .pct_change() * 100
        returns = stock["Close"].pct_change() * 100

        #CODE HERE - Create DataFrame with analysis columns
        analysis_df = pd.DataFrame({
            'Date': stock.index,
            'Close': stock['Close'],
            'SMA_20': stock["Close"].rolling(window = 20).mean(),         # 20-day moving average
            'SMA_50': stock["Close"].rolling(window = 50).mean(),         # 50-day moving average
            'Daily_Return_%': returns,
            'Rolling_Vol_20': returns.rolling(window = 20).std(), # 20-day rolling volatility
            'Volume_M': stock['Volume'] / 1000000
        })

        #CODE HERE - Export to CSV using .to_csv() with index=False
        analysis_df.to_csv(filename, index=False)
        print(f"✅ {symbol}: {filename} ({len(analysis_df)} rows)")

# Export combined close prices
#CODE HERE - Export close_prices to CSV
close_prices.to_csv('Lab2_close_prices.csv')
print(f"✅ Combined close prices: Lab2_close_prices.csv")

# Export daily returns
#CODE HERE - Calculate and export daily returns
daily_returns = close_prices.pct_change() * 100
daily_returns.to_csv('Lab2_daily_returns.csv')
print(f"✅ Daily returns: Lab2_daily_returns.csv")

# 8.2 Export to Excel with multiple sheets
print(f"\n📊 Creating comprehensive Excel workbook:")

#CODE HERE - Create Excel writer using pd.ExcelWriter with openpyxl engine
with pd.ExcelWriter('Lab2_Stock_Analysis.xlsx', engine='openpyxl') as writer:
    # Sheet 1: Close prices
    #CODE HERE - Export close_prices to 'Close_Prices' sheet
    close_prices.to_excel(writer, sheet_name='Close_Prices')

    # Sheet 2: Daily returns
    #CODE HERE - Export daily_returns to 'Daily_Returns' sheet
    daily_returns.to_excel(writer, sheet_name='Daily_Returns')

    # Sheet 3: Correlation matrix
    #CODE HERE - Export correlation matrix to 'Correlations' sheet
    close_prices.corr().to_excel(writer, sheet_name='Correlations')

    # Sheet 4: Summary statistics
    #CODE HERE - Create and export summary statistics DataFrame
    summary_stats = pd.DataFrame({
        'Avg_Price': close_prices.mean(),
        'Std_Price': close_prices.std(),
        'Min_Price': close_prices.min(),
        'Max_Price': close_prices.max(),
        'Avg_Return_%': daily_returns.mean(),
        'Volatility_%': daily_returns.std(),
        'Total_Return_%': (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
    })
    summary_stats.to_excel(writer, sheet_name='Summary_Stats')

print(f"✅ Excel workbook: Lab2_Stock_Analysis.xlsx (4 sheets)")

# 8.3 Create a simple summary report
print(f"\n📝 Creating summary report:")

# Calculate key metrics
#CODE HERE - Find best and worst performers using .idxmax() and .idxmin()
best_performer = (close_prices.iloc[-1] / close_prices.iloc[0] - 1).idxmax()
worst_performer = (close_prices.iloc[-1] / close_prices.iloc[0] - 1).idxmin()
#CODE HERE - Find lowest and highest volatility stocks
lowest_volatility = daily_returns.std().idxmin()
highest_volatility = daily_returns.std().idxmax()
#CODE HERE - Calculate total returns
total_returns = (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100

#CODE HERE - Create summary report string with key findings
summary_report = f"""
BASIC PANDAS FINANCIAL ANALYSIS REPORT
=====================================

Analysis Period: {close_prices.index.min().strftime('%Y-%m-%d')} to {close_prices.index.max().strftime('%Y-%m-%d')}
Number of Trading Days: {len(close_prices)}
Stocks Analyzed: {', '.join(close_prices.columns)}

PERFORMANCE SUMMARY:
-------------------
Best Performer: {best_performer} ({total_returns[best_performer]:.2f}%)
Worst Performer: {worst_performer} ({total_returns[worst_performer]:.2f}%)
Lowest Volatility: {lowest_volatility} ({daily_returns.std()[lowest_volatility]:.2f}%)
Highest Volatility: {highest_volatility} ({daily_returns.std()[highest_volatility]:.2f}%)

INDIVIDUAL STOCK PERFORMANCE:
-----------------------------
"""

for symbol in close_prices.columns:
    avg_return = daily_returns[symbol].mean()
    volatility = daily_returns[symbol].std()
    total_return = total_returns[symbol]

    summary_report += f"""
{symbol}:
  - Total Return: {total_return:.2f}%
  - Average Daily Return: {avg_return:.4f}%
  - Volatility (Daily): {volatility:.4f}%
  - Current Price: ${close_prices[symbol].iloc[-1]:.2f}
"""

summary_report += f"""

CORRELATION ANALYSIS:
--------------------
Highest Correlation: {close_prices.corr().unstack().drop_duplicates().sort_values(ascending=False).iloc[1]:.4f}
Average Correlation: {close_prices.corr().unstack().drop_duplicates().mean():.4f}

DATA QUALITY:
------------
Missing Values: {close_prices.isnull().sum().sum()}
Data Completeness: {((len(close_prices) * len(close_prices.columns) - close_prices.isnull().sum().sum()) / (len(close_prices) * len(close_prices.columns))) * 100:.1f}%

FILES CREATED:
--------------
- Lab2_close_prices.csv
- Lab2_daily_returns.csv
- Lab2_Stock_Analysis.xlsx
- Individual stock analysis files for each symbol

Generated using Basic Pandas for Financial Data Analysis
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save the report
#CODE HERE - Save summary_report to text file
with open('Lab2_Analysis_Report.txt', 'w') as f:
    f.write(summary_report)

print(f"✅ Summary report: Lab2_Analysis_Report.txt")

# 8.4 Display file summary
print(f"\n📋 Files Created Summary:")
print("=" * 30)
print("1. Individual stock CSV files")
print("2. Lab2_close_prices.csv - Combined closing prices")
print("3. Lab2_daily_returns.csv - Daily return percentages") 
print("4. Lab2_Stock_Analysis.xlsx - Comprehensive Excel workbook")
print("5. Lab2_Analysis_Report.txt - Summary analysis report")

print(f"\n✅ All data export operations completed successfully!")
print(f"🎯 You now have comprehensive datasets ready for further analysis or presentation.")

print(summary_report)
