#!/usr/bin/env python
# coding: utf-8

# # Pandas Time Series Analysis for Finance
# 

# ## A Comprehensive Guide to Financial Data Analysis
# 
# ---
# 
# ### 📊 Learning Objectives:
# 
# After completing this tutorial, you will be able to:
# 
# 1. **Load and manipulate financial time series data** using pandas
# 2. **Work with DateTimeIndex** for efficient time-based operations
# 3. **Resample financial data** to different frequencies (daily, weekly, monthly)
# 4. **Calculate financial indicators** using rolling windows and expanding operations
# 5. **Perform time shifting** for creating lagged variables and returns
# 6. **Analyze portfolio performance** and risk metrics
# 7. **Handle missing data** and data quality issues in financial datasets
# 
# ### 📈 Real-World Applications:
# 
# - Portfolio performance analysis
# - Risk management calculations
# - Technical indicator computation
# - Market correlation analysis
# - Financial reporting and dashboards
# 
# ### 📚 Prerequisites:
# 
# - Basic Python knowledge
# - Understanding of financial concepts (stocks, returns, volatility)
# - Basic pandas DataFrame operations
# 
# ---
# 
# **Note**: This tutorial uses real financial market data from Yahoo Finance covering stocks, ETFs, bonds, and cryptocurrencies.

# ## 1. Setup and Data Loading
# 
# ### 🎯 **What You'll Learn in This Section:**
# 
# In this foundational section, you'll master the essential skills for setting up your pandas environment and loading financial data properly. These are the building blocks that every financial analyst needs to know.
# 
# **Key Knowledge Points:**
# - **Library imports**: Understanding which libraries are essential for financial time series analysis
# - **DateTime handling**: Why proper date/time management is crucial in finance
# - **Data loading best practices**: How to load financial data efficiently and correctly
# - **Index configuration**: Setting up your data structure for optimal time series operations
# 
# **Why This Matters:**
# Financial data is inherently time-based. Stock prices, trading volumes, and market indicators all depend on precise timing. Getting your data structure right from the beginning will save you hours of debugging later and enable powerful time-based operations that pandas offers.
# 
# ### Essential Libraries for Financial Analysis

# In[21]:


# Essential libraries for financial data analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("📊 Financial Data Analysis with Pandas - Setup Complete")
print(f"📅 Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
print(f"🐼 Pandas Version: {pd.__version__}")
print(f"🔢 NumPy Version: {np.__version__}")


# ### 1.1 Loading Financial Data with Proper DateTime Index
# 
# **🧠 Core Concept: DateTimeIndex**
# 
# The **DateTimeIndex** is pandas' specialized index type for handling time series data. Think of it as a supercharged row index that understands dates, times, and financial calendar concepts.
# 
# **Why DateTimeIndex is Essential for Finance:**
# - **Automatic alignment**: When you merge datasets, pandas automatically aligns them by date
# - **Time-based slicing**: You can easily get "all data from 2023" or "Q4 2022" 
# - **Built-in calendar logic**: Understands business days, month-ends, and holidays
# - **Resampling capabilities**: Convert daily data to weekly/monthly with one command
# - **Performance**: Much faster operations on time-indexed data
# 
# **Two Approaches to Loading Financial Data:**
# 
# 1. **Method 1 (Recommended)**: Load with DateTimeIndex directly using `parse_dates=True`
# 2. **Method 2 (Alternative)**: Load first, then convert dates and set index
# 
# Let's see both methods in action:

# In[23]:


# Method 1: Best practice - Load with datetime index directly
# This is the recommended approach for financial data
apple_stock = pd.read_csv('AAPL_stock_data.csv', index_col='Date', parse_dates=True)

print("📈 Apple Stock Data Loaded:")
print(f"   Date Range: {apple_stock.index[0].date()} to {apple_stock.index[-1].date()}")
print(f"   Total Trading Days: {len(apple_stock)}")
print(f"   Columns: {list(apple_stock.columns)}")
print(f"   Index Type: {type(apple_stock.index)}")

# Display first few rows
apple_stock.head()


# In[24]:


# Method 2: Alternative approach - Convert after loading
# Sometimes necessary when working with problematic date formats

# Load without datetime parsing first
raw_data = pd.read_csv('MSFT_stock_data.csv')
print("Before conversion:")
print(f"{raw_data.head(3)}")
print(f"First few dates: {raw_data['Date'].head(3).tolist()}")

# Convert Date column to datetime and set as index
print("After conversion:")
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data = raw_data.set_index('Date')
print(f"{raw_data.head(3)}")


# ### 1.2 Understanding Financial Data Structure
# 
# **📊 Anatomy of Financial OHLCV Data**
# 
# Most financial datasets follow the **OHLCV** format, which captures complete price and volume information for each trading period:
# 
# | Column | Full Name | What It Represents | When to Use |
# |--------|-----------|-------------------|-------------|
# | **Open** | Opening Price | First traded price of the period | Gap analysis, overnight moves |
# | **High** | Highest Price | Peak price during the period | Resistance levels, volatility |
# | **Low** | Lowest Price | Minimum price during the period | Support levels, risk assessment |
# | **Close** | Closing Price | Last traded price of the period | **Most common for analysis** |
# | **Volume** | Trading Volume | Number of shares/contracts traded | Liquidity, validation of moves |
# | **Adj Close** | Adjusted Close | Close price adjusted for splits/dividends | **Best for returns calculation** |
# 
# **💡 Pro Tips:**
# - **Use 'Close' for most analysis** - it's the standard reference price
# - **Use 'Adj Close' for returns** - accounts for corporate actions (stock splits, dividends)
# - **Volume confirms price moves** - high volume = more reliable price changes
# - **High-Low range measures volatility** - wider ranges indicate more uncertainty
# 
# Let's explore our dataset structure:

# In[25]:


# Explore the financial data structure
print("📊 APPLE STOCK DATA OVERVIEW")
print("=" * 40)

# Basic information about the dataset
print("\n📋 Dataset Information:")
apple_stock.info()

print("\n📊 Statistical Summary:")
print(apple_stock.describe().round(2))

print("\n🗓️ Date Range Analysis:")
print(f"Start Date: {apple_stock.index.min()}")
print(f"End Date: {apple_stock.index.max()}")
print(f"Trading Days: {len(apple_stock)}")
print(f"Date Range: {(apple_stock.index.max() - apple_stock.index.min()).days} calendar days")

# Check for missing values
print("\n❓ Missing Values:")
missing_values = apple_stock.isnull().sum()
if missing_values.sum() == 0:
    print("✅ No missing values found")
else:
    print(missing_values[missing_values > 0])


# ### 1.3 Time-Based Data Slicing with Pandas
# 
# **🎯 Core Concepts**
# 
# When working with time series data in pandas, you'll frequently need to extract specific date ranges or filter data by time periods. Modern pandas requires **complete date specifications** or **boolean indexing** for reliable time-based slicing.
# 
# **🔄 Slicing Methods Summary**
# 
# | Method | Syntax | Use Case | Example |
# |--------|--------|----------|---------|
# | **Complete Date Range** | `df.loc['start':'end']` | Precise date ranges | `apple_stock.loc['2024-01-01':'2024-01-31']` |
# | **Boolean Indexing** | `df[df.index.condition]` | Flexible filtering | `apple_stock[apple_stock.index.year == 2024]` |
# | **Date Offsets** | `df.loc[date_offset:]` | Relative periods | `apple_stock.loc[six_months_ago:]` |
# | **Query Method** | `df.query('condition')` | Readable syntax | `apple_stock.query('index.year == 2024')` |
# | **Intersection** | `df.loc[dates.intersection()]` | Handle missing dates | `apple_stock.loc[business_days.intersection(apple_stock.index)]` |
# 
# **Essential Slicing Techniques:**
# 
# 1. **Complete Date Range Slicing** - Use full date strings with `.loc[]`
# 2. **Boolean Indexing** - Use datetime components for flexible filtering  
# 3. **Date Offset Operations** - Use pandas date offsets for relative periods
# 4. **Query Method** - Alternative readable approach for date filtering
# 
# **⚠️ Important Notes:**
# - Modern pandas **does not support** partial string indexing like `df['2024']` or `df['2024-01']`
# - Always use **complete dates** or **boolean indexing** for reliable results
# - Combine conditions with `&` (and) and `|` (or) operators
# - Use parentheses when combining multiple boolean conditions

# In[6]:


# Demonstrate various slicing techniques
print("📊 PANDAS TIME-BASED SLICING EXAMPLES")
print("=" * 40)

# 1. Complete Date Range Slicing
print("\n1️⃣ Complete Date Range Slicing:")

# Single specific date (if exists)
single_date = apple_stock.loc['2024-01-03']
print(f"✅ Single date access: 2024-01-03")
print(f"Close price: ${single_date['Close']:.2f}")

multi_dates = apple_stock.loc['2024-01-03':'2025-01-10']
print("total days from 2024-01-03 to 2025-01-10:", len(multi_dates))


# Date range (first 10 days of available data)
print(f"\n✅ First 10 trading days:")
first_10_days = apple_stock.head(10)
print(f"Date range: {first_10_days.index[0].strftime('%Y-%m-%d')} to {first_10_days.index[-1].strftime('%Y-%m-%d')}")
print(f"Number of trading days: {len(first_10_days)}")

# 2. Boolean Indexing (Recommended Approach)
print("\n2️⃣ Boolean Indexing:")

# Filter by year
available_years = apple_stock.index.year.unique()
target_year = available_years[-1]  # Use the most recent year
year_data = apple_stock[apple_stock.index.year == target_year]
print(f"✅ Year {target_year}: {len(year_data)} trading days")

# Filter by month and year
if len(year_data) > 0:
    target_month = year_data.index[0].month
    month_data = apple_stock[(apple_stock.index.year == target_year) & 
                           (apple_stock.index.month == target_month)]
    print(f"✅ {target_year}-{target_month:02d}: {len(month_data)} trading days")

# Multiple years
if len(available_years) >= 2:
    recent_years = available_years[-2:]
    multi_year = apple_stock[apple_stock.index.year.isin(recent_years)]
    print(f"✅ Years {recent_years[0]}-{recent_years[1]}: {len(multi_year)} trading days")

# 3. Date Offset Operations
print("\n3️⃣ Date Offset Operations:")

# Last 30 calendar days
thirty_days_ago = apple_stock.index[-1] - pd.Timedelta(days=30)
last_30_days = apple_stock.loc[thirty_days_ago:]
print(f"✅ Last 30 days: {len(last_30_days)} trading days")
print(f"Date range: {last_30_days.index[0].strftime('%Y-%m-%d')} to {last_30_days.index[-1].strftime('%Y-%m-%d')}")

# Last 6 months using DateOffset
six_months_ago = apple_stock.index[-1] - pd.DateOffset(months=6)
last_6_months = apple_stock.loc[six_months_ago:]
print(f"✅ Last 6 months: {len(last_6_months)} trading days")

# 4. Query Method
print("\n4️⃣ Query Method:")
query_result = apple_stock.query('index > "2025-01-01"')
print(f"✅ Query method for {target_year}: {len(query_result)} trading days")
query_result = apple_stock.query('index > "2025-02"')
print(f"✅ Query method for {target_year}, from Feb: {len(query_result)} trading days")

# Practical Example: Quarterly Analysis
print("\n📈 PRACTICAL EXAMPLE: Quarterly Analysis")
print("=" * 35)

# Find a year with sufficient data
for year in reversed(available_years):
    year_subset = apple_stock[apple_stock.index.year == year]
    if len(year_subset) > 100:  # Ensure we have substantial data
        q1_data = year_subset[year_subset.index.month.isin([1, 2, 3])]
        q4_data = year_subset[year_subset.index.month.isin([10, 11, 12])]

        if len(q1_data) > 0 and len(q4_data) > 0:
            print(f"\n📊 {year} Quarterly Analysis:")
            print(f"Q1 trading days: {len(q1_data)}")
            print(f"Q4 trading days: {len(q4_data)}")

            # Calculate quarterly returns
            q1_return = (q1_data['Close'].iloc[-1] / q1_data['Close'].iloc[0] - 1) * 100
            q4_return = (q4_data['Close'].iloc[-1] / q4_data['Close'].iloc[0] - 1) * 100

            print(f"Q1 return: {q1_return:.2f}%")
            print(f"Q4 return: {q4_return:.2f}%")


print("\n✅ All slicing examples completed successfully!")


# ### 2 Column Selection and Multi-Column Operations
# 
# **🎯 Working with Financial Data Columns**
# 
# In financial analysis, you'll often need to work with specific columns or combinations of columns. Here are the most common patterns:
# 
# **Common Selection Patterns:**
# - **Single column**: `data['Close']` → Returns a pandas Series
# - **Multiple columns**: `data[['Open', 'Close']]` → Returns a DataFrame  
# - **OHLC subset**: `data[['Open', 'High', 'Low', 'Close']]` → Traditional price data
# - **Price + Volume**: `data[['Close', 'Volume']]` → Price and liquidity analysis
# 
# **💡 Series vs DataFrame:**
# - **Series** (single column): Perfect for calculations, plotting single metrics
# - **DataFrame** (multiple columns): Needed for comparisons, correlations, multi-asset analysis
# 
# **Practical Applications:**
# - **OHLC analysis**: Calculate daily ranges, gaps, price patterns
# - **Volume analysis**: Confirm price moves, identify accumulation/distribution
# - **Price transformation**: Create new metrics from existing columns
# 
# Let's see these concepts in action:

# In[7]:


# Working with specific columns
print("📊 COLUMN SELECTION AND OPERATIONS")
print("=" * 35)

# Close prices time series
close_prices = apple_stock['Close']
print(f"\n📈 Close prices series:")
print(f"Type: {type(close_prices)}")
print(f"Length: {len(close_prices)}")
print(f"Latest price: ${close_prices.iloc[-1]:.2f}")

# OHLC data (Open, High, Low, Close)
ohlc_data = apple_stock[['Open', 'High', 'Low', 'Close']]
print(f"\n📊 OHLC data shape: {ohlc_data.shape}")
print("\nLast 3 trading days OHLC:")
print(ohlc_data.tail(3).round(2))

# Price and volume analysis
price_volume = apple_stock[['Close', 'Volume']]
print(f"\n💰 Recent Price vs Volume:")
recent_pv = price_volume.tail(5)
recent_pv['Volume_Million'] = recent_pv['Volume'] / 1_000_000
print(recent_pv[['Close', 'Volume_Million']].round(2))


# ## 3. Financial Calculations and Derived Metrics
# 
# This is where pandas becomes a powerful financial analysis tool. You'll learn to transform raw price data into meaningful financial metrics that drive investment decisions.
# 
# **Key Knowledge Points:**
# - **Returns calculation**: The foundation of all financial analysis
# - **Annualization**: Converting daily metrics to yearly equivalents
# - **Risk metrics**: Volatility, Sharpe ratio, and other risk measures
# - **Technical indicators**: Moving averages, momentum indicators, and trend analysis
# 
# **Why This Matters:**
# Raw prices tell us little about investment performance. What matters are **returns** (how much money you make), **volatility** (how risky it is), and **risk-adjusted returns** (how much return per unit of risk). These calculations form the backbone of portfolio management, risk assessment, and trading strategies.
# 
# ### 3.1 Returns Calculation
# 
# **📈 Understanding Financial Returns**
# 
# Returns are the percentage change in price over time - the fundamental building block of finance. Different types of returns serve different purposes:
# 
# | Return Type | Formula | When to Use | Key Benefits |
# |-------------|---------|-------------|--------------|
# | **Simple Returns** | `(P₁ - P₀) / P₀` | Performance measurement | Easy to interpret |
# | **Log Returns** | `ln(P₁ / P₀)` | Risk modeling, time series | Additive over time |
# | **Periodic Returns** | Same formula, different periods | Comparing different timeframes | Standardized comparison |
# 
# **🕐 Time Horizons in Finance:**
# - **Daily**: Day-to-day price changes, short-term trading
# - **Weekly**: Reducing noise, medium-term patterns  
# - **Monthly**: Strategic analysis, longer-term trends
# - **Annual**: Performance evaluation, benchmarking
# 
# **💡 Pro Tips:**
# - Use `pct_change()` for simple returns
# - Use `pct_change(periods=5)` for weekly returns (5 trading days)
# - Always multiply by 100 for percentage display
# - Log returns are better for risk calculations
# 
# Returns are fundamental in finance for measuring performance and risk.

# In[8]:


# Calculate various types of returns
print("📊 FINANCIAL RETURNS CALCULATION")
print("=" * 35)

# Daily returns (percentage change)
apple_stock['Daily_Return'] = apple_stock['Close'].pct_change()
apple_stock['Daily_Return_Pct'] = apple_stock['Daily_Return'] * 100

# Weekly returns (comparing to 5 trading days ago)
apple_stock['Weekly_Return'] = apple_stock['Close'].pct_change(periods=5)
apple_stock['Weekly_Return_Pct'] = apple_stock['Weekly_Return'] * 100

# Monthly returns (comparing to ~21 trading days ago)
apple_stock['Monthly_Return'] = apple_stock['Close'].pct_change(periods=21)
apple_stock['Monthly_Return_Pct'] = apple_stock['Monthly_Return'] * 100

# Log returns (for risk calculations)
apple_stock['Log_Return'] = np.log(apple_stock['Close'] / apple_stock['Close'].shift(1))

print("✅ Returns calculated successfully!")
print("\n📊 Returns Summary (last 10 days):")
returns_summary = apple_stock[['Close', 'Daily_Return_Pct', 'Weekly_Return_Pct']].tail(10)
print(returns_summary.round(3))

# Returns statistics
print("\n📈 Daily Returns Statistics:")
daily_stats = apple_stock['Daily_Return_Pct'].describe()
print(f"Mean: {daily_stats['mean']:.3f}%")
print(f"Std Dev: {daily_stats['std']:.3f}%")
print(f"Min: {daily_stats['min']:.3f}%")
print(f"Max: {daily_stats['max']:.3f}%")

# Annualized metrics
trading_days = 252
annual_return = apple_stock['Daily_Return'].mean() * trading_days * 100
annual_volatility = apple_stock['Daily_Return'].std() * np.sqrt(trading_days) * 100
sharpe_ratio = annual_return / annual_volatility

print(f"\n📊 Annualized Metrics:")
print(f"Annual Return: {annual_return:.2f}%")
print(f"Annual Volatility: {annual_volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")


# ### 3.2 Technical Indicators Using Rolling Windows
# 
# **📊 Technical Analysis with Pandas**
# 
# Technical indicators are mathematical calculations based on historical price and volume data. They help identify trends, momentum, and potential reversal points. Pandas' `rolling()` function makes calculating these indicators straightforward.
# 
# **🔧 Core Concepts:**
# 
# **Rolling Windows**: A "moving" calculation that slides through your data
# - `rolling(window=20)` = last 20 observations
# - Each point uses the most recent N observations
# - Creates smooth, trend-following indicators
# 
# **Types of Technical Indicators:**
# 
# | Category | Examples | Purpose | Pandas Implementation |
# |----------|----------|---------|----------------------|
# | **Trend Following** | SMA, EMA | Identify direction | `rolling().mean()`, `ewm()` |
# | **Momentum** | RSI, MACD | Measure speed of change | Custom calculations |
# | **Volatility** | Bollinger Bands | Measure price variation | `rolling().std()` |
# | **Volume** | Volume MA | Confirm price moves | `rolling().mean()` on volume |
# 
# **📈 Most Important Indicators for Beginners:**
# 
# 1. **Simple Moving Average (SMA)**: Average price over N periods
# 2. **Exponential Moving Average (EMA)**: Weighted average favoring recent prices  
# 3. **Bollinger Bands**: Price channels based on standard deviation
# 4. **RSI**: Momentum oscillator (0-100 scale)
# 5. **MACD**: Trend and momentum combination
# 
# Let's implement these step by step:

# In[9]:


# Calculate technical indicators
print("📊 TECHNICAL INDICATORS CALCULATION")
print("=" * 38)

# Simple Moving Averages
apple_stock['SMA_10'] = apple_stock['Close'].rolling(window=10).mean()
apple_stock['SMA_20'] = apple_stock['Close'].rolling(window=20).mean()
apple_stock['SMA_50'] = apple_stock['Close'].rolling(window=50).mean()
apple_stock['SMA_200'] = apple_stock['Close'].rolling(window=200).mean()

# Exponential Moving Average
apple_stock['EMA_12'] = apple_stock['Close'].ewm(span=12).mean()
apple_stock['EMA_26'] = apple_stock['Close'].ewm(span=26).mean()

# MACD (Moving Average Convergence Divergence)
apple_stock['MACD'] = apple_stock['EMA_12'] - apple_stock['EMA_26']
apple_stock['MACD_Signal'] = apple_stock['MACD'].ewm(span=9).mean()
apple_stock['MACD_Histogram'] = apple_stock['MACD'] - apple_stock['MACD_Signal']

# Bollinger Bands
apple_stock['BB_Middle'] = apple_stock['Close'].rolling(window=20).mean()
bb_std = apple_stock['Close'].rolling(window=20).std()
apple_stock['BB_Upper'] = apple_stock['BB_Middle'] + (bb_std * 2)
apple_stock['BB_Lower'] = apple_stock['BB_Middle'] - (bb_std * 2)
apple_stock['BB_Width'] = apple_stock['BB_Upper'] - apple_stock['BB_Lower']
apple_stock['BB_Position'] = (apple_stock['Close'] - apple_stock['BB_Lower']) / apple_stock['BB_Width']

# Relative Strength Index (RSI) - simplified version
delta = apple_stock['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
apple_stock['RSI'] = 100 - (100 / (1 + rs))

# Volatility (rolling standard deviation)
apple_stock['Volatility_10d'] = apple_stock['Daily_Return'].rolling(window=10).std() * np.sqrt(252) * 100
apple_stock['Volatility_30d'] = apple_stock['Daily_Return'].rolling(window=30).std() * np.sqrt(252) * 100

print("✅ Technical indicators calculated successfully!")
print("\n📊 Latest Technical Indicators:")
latest_indicators = apple_stock[[
    'Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_Position', 'Volatility_30d'
]].tail(5)
print(latest_indicators.round(3))


# ### 3.3 Technical Indicators Summary

# In[10]:


# Display recent price data with moving averages for analysis
recent_data = apple_stock.tail(10)  # Last 10 days of data

print("📊 Recent Price Data with Moving Averages:")
price_ma_summary = recent_data[['Close', 'SMA_10', 'SMA_20', 'SMA_50']].round(2)
print(price_ma_summary)

# Display Bollinger Bands analysis
print("\n📊 Bollinger Bands Analysis (last 5 days):")
bb_analysis = recent_data[['Close', 'BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Position']].round(3)
print(bb_analysis.tail())

# Display RSI and MACD indicators
print("\n📊 Technical Indicators (last 5 days):")
tech_indicators = recent_data[['Close', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram']].round(3)
print(tech_indicators.tail())

# Returns distribution summary
returns_clean = apple_stock['Daily_Return_Pct'].dropna()
print("\n📊 Daily Returns Distribution Summary:")
print(f"Mean: {returns_clean.mean():.3f}%")
print(f"Std Dev: {returns_clean.std():.3f}%")
print(f"Min: {returns_clean.min():.3f}%")
print(f"Max: {returns_clean.max():.3f}%")
print(f"25th percentile: {returns_clean.quantile(0.25):.3f}%")
print(f"75th percentile: {returns_clean.quantile(0.75):.3f}%")

print("\n✅ Technical indicators analysis completed!")


# ## 4. Time Series Resampling
# 
# ### 🎯 **What You'll Learn in This Section:**
# 
# Resampling is one of pandas' most powerful features for financial analysis. You'll learn to transform data between different time frequencies - a critical skill for multi-timeframe analysis and reporting.
# 
# **Key Knowledge Points:**
# - **Frequency conversion**: Transform daily data to weekly, monthly, or yearly
# - **Aggregation methods**: Choosing the right method for different data types
# - **Business calendar awareness**: Working with trading days vs calendar days
# - **Performance implications**: How resampling affects your analysis
# 
# **Real-World Applications:**
# - **Quarterly reporting**: Convert daily returns to quarterly performance
# - **Risk management**: Calculate monthly volatility from daily data
# - **Strategy testing**: Test trading strategies across different timeframes
# - **Benchmarking**: Compare performance using standardized periods
# 
# **🕰️ Financial Time Frequencies:**
# 
# | Frequency | Code | Use Case | Typical Analysis |
# |-----------|------|----------|------------------|
# | **Daily** | 'D' | Day trading, short-term | Intraday patterns, daily returns |
# | **Weekly** | 'W' | Swing trading | Weekly trends, reduced noise |
# | **Monthly** | 'M' | Portfolio management | Monthly reports, long-term trends |
# | **Quarterly** | 'Q' | Fundamental analysis | Earnings periods, strategic planning |
# | **Yearly** | 'A' | Investment planning | Annual performance, benchmarking |
# 
# Resampling allows you to change the frequency of your time series data - crucial for financial analysis at different time horizons.

# ### 4.1 Frequency Conversion
# 
# **🔄 Understanding Resampling Logic**
# 
# When you resample financial data, you're essentially asking: *"Instead of daily observations, show me weekly/monthly/yearly summaries."* But what value represents each new period?
# 
# **The Aggregation Decision:**
# 
# For **price data**, you typically want the **last** value (closing price of the period):
# - `resample('W').last()` → Weekly closing prices
# - `resample('M').last()` → Month-end prices
# 
# For **volume data**, you typically want the **sum** (total trading activity):
# - `resample('W').sum()` → Total weekly volume
# - `resample('M').sum()` → Total monthly volume
# 
# **🎯 Common Resampling Patterns:**
# 
# | Data Type | Best Aggregation | Reason |
# |-----------|------------------|--------|
# | **Prices** | `.last()` | Closing price represents the period |
# | **Volume** | `.sum()` | Total trading activity |
# | **Returns** | `.sum()` | Cumulative return over period |
# | **Volatility** | `.mean()` | Average volatility level |
# 
# **⚠️ Important Notes:**
# - Higher frequencies lose information (daily → monthly loses daily variations)
# - Lower frequencies smooth out noise and reveal longer-term trends
# - Choose your frequency based on your analysis goals
# 
# Let's see resampling in action:

# In[11]:


# Load multi-asset data for resampling examples
portfolio_data = pd.read_csv('sample_portfolio.csv', index_col='Date', parse_dates=True)

print("📊 TIME SERIES RESAMPLING")
print("=" * 28)
print(f"\n📈 Original portfolio data: {portfolio_data.shape}")
print(f"Frequency: Daily data")
print(f"Date range: {portfolio_data.index[0].date()} to {portfolio_data.index[-1].date()}")

# Common resampling frequencies in finance
frequency_map = {
      'Weekly': 'W',      # Week ending
      'Monthly': 'M',     # Month ending (older pandas)
      'Quarterly': 'Q',   # Quarter ending (older pandas)
      'Yearly': 'A'       # Year ending (older pandas)
  }

print("\n📊 Resampling Portfolio Value to Different Frequencies:")
resampled_data = {}

for freq_name, freq_code in frequency_map.items():
    # For price data, we typically want the last value (last trading price)
    resampled = portfolio_data['Portfolio_Value'].resample(freq_code).last()
    resampled_data[freq_name] = resampled

    print(f"\n{freq_name} ({freq_code}): {len(resampled)} periods")
    print(f"Latest values:")
    print(resampled.tail(3).round(2))

# Calculate returns at different frequencies
print("\n📈 Returns at Different Frequencies:")
for freq_name, data in resampled_data.items():
    returns = data.pct_change().dropna() * 100
    print(f"\n{freq_name} Returns (last 5):")
    print(f"Mean: {returns.mean():.2f}%, Std: {returns.std():.2f}%")
    print(returns.tail(5).round(2))


# ### 4.2 Aggregation Methods for Financial Data
# 
# **🎯 Advanced Aggregation Strategies**
# 
# Different columns in financial data require different aggregation approaches. The `agg()` method lets you specify exactly how each column should be summarized:
# 
# **OHLCV Aggregation Rules:**
# 
# | Column | Aggregation | Logic |
# |--------|-------------|-------|
# | **Open** | `first` | First price of the period |
# | **High** | `max` | Highest price during period |
# | **Low** | `min` | Lowest price during period |
# | **Close** | `last` | Final price of the period |
# | **Volume** | `sum` | Total shares traded |
# 
# **📊 Multi-Metric Aggregation:**
# 
# You can calculate multiple statistics for the same column:
# ```python
# .agg({
#     'Close': ['first', 'last', 'min', 'max', 'mean'],
#     'Volume': ['sum', 'mean', 'std']
# })
# ```
# 
# **💡 Why This Matters:**
# - **OHLC preservation**: Maintains the essential price structure at any frequency
# - **Statistical richness**: Get comprehensive summaries in one operation
# - **Comparative analysis**: Compare intraday vs period-end values
# 
# Let's implement professional-grade aggregation:

# In[12]:


print("📊 FINANCIAL DATA AGGREGATION METHODS")
print("=" * 42)

# Select OHLCV data for aggregation
ohlcv_data = apple_stock[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Weekly aggregation with appropriate methods for each column
weekly_ohlc = ohlcv_data.resample('W').agg({
    'Open': 'first',    # First trading day's opening price
    'High': 'max',      # Highest price during the week
    'Low': 'min',       # Lowest price during the week
    'Close': 'last',    # Last trading day's closing price
    'Volume': 'sum'     # Total volume traded during the week
})

print("\n📅 Weekly OHLCV Data (last 8 weeks):")
print(weekly_ohlc.tail(8).round(2))

# Monthly aggregation
monthly_stats = ohlcv_data.resample('M').agg({
    'Close': ['first', 'last', 'min', 'max', 'mean'],
    'Volume': ['sum', 'mean', 'std']
})


# In[13]:


# Different aggregation methods for different types of financial data
print("📊 FINANCIAL DATA AGGREGATION METHODS")
print("=" * 42)

# Select OHLCV data for aggregation
ohlcv_data = apple_stock[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Weekly aggregation with appropriate methods for each column
weekly_ohlc = ohlcv_data.resample('W').agg({
    'Open': 'first',    # First trading day's opening price
    'High': 'max',      # Highest price during the week
    'Low': 'min',       # Lowest price during the week
    'Close': 'last',    # Last trading day's closing price
    'Volume': 'sum'     # Total volume traded during the week
})

print("\n📅 Weekly OHLCV Data (last 8 weeks):")
print(weekly_ohlc.tail(8).round(2))

# Monthly aggregation
monthly_stats = ohlcv_data.resample('M').agg({
    'Close': ['first', 'last', 'min', 'max', 'mean'],
    'Volume': ['sum', 'mean', 'std']
})

# Flatten column names
monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]

print("\n📅 Monthly Statistics (last 6 months):")
print(monthly_stats.tail(6).round(2))

# Calculate monthly returns and statistics
monthly_returns = monthly_stats['Close_last'].pct_change().dropna() * 100
monthly_volatility = (monthly_stats['Close_max'] - monthly_stats['Close_min']) / monthly_stats['Close_mean'] * 100

monthly_summary = pd.DataFrame({
    'Monthly_Return_%': monthly_returns,
    'Monthly_Volatility_%': monthly_volatility,
    'Avg_Daily_Volume_M': monthly_stats['Volume_mean'] / 1_000_000
})

print("\n📊 Monthly Performance Summary:")
print(monthly_summary.tail(6).round(2))


# ## 5. Time Shifting and Lagged Variables
# 
# ### 🎯 **What You'll Learn in This Section:**
# 
# Time shifting is a fundamental technique in financial analysis that allows you to compare current values with past values. This enables momentum analysis, trend detection, and sophisticated trading strategies.
# 
# **Key Knowledge Points:**
# - **Lag creation**: Access yesterday's, last week's, or last month's values
# - **Lead creation**: Access future values (for alignment, not prediction!)
# - **Return calculations**: Compare current vs historical prices
# - **Signal generation**: Create buy/sell signals based on crossovers and patterns
# 
# **Real-World Applications:**
# - **Momentum analysis**: Is the stock continuing yesterday's trend?
# - **Mean reversion**: Compare current price to historical average
# - **Trading signals**: Moving average crossovers, breakout patterns
# - **Risk management**: Portfolio rebalancing based on recent performance
# 
# **🔄 Core Concept: The `shift()` Function**
# 
# Think of `shift()` as moving your data up or down in time:
# - `shift(1)` = Move data forward 1 period (creates yesterday's value)
# - `shift(-1)` = Move data backward 1 period (creates tomorrow's value)
# - `shift(5)` = Move data forward 5 periods (creates last week's value for daily data)
# 
# Time shifting is essential for:
# - Creating lagged variables for prediction models
# - Calculating period-over-period changes
# - Aligning data for correlation analysis

# ### 5.1 Basic Time Shifting
# 
# **🎯 Understanding Time Shift Mechanics**
# 
# Time shifting moves your entire dataset forward or backward in time. This creates new columns that represent historical values aligned with current dates.
# 
# **Visual Example:**
# ```
# Original Data:        After shift(1):       After shift(-1):
# Date    Close         Date    Close         Date    Close
# Jan 1   100           Jan 1   NaN           Jan 1   110
# Jan 2   110           Jan 2   100           Jan 2   120  
# Jan 3   120           Jan 3   110           Jan 3   NaN
# ```
# 
# **🔍 Key Observations:**
# - **Forward shift (+)**: Creates **lag** variables (historical values)
# - **Backward shift (-)**: Creates **lead** variables (future values)
# - **NaN values**: Appear at boundaries where no data exists
# - **Data alignment**: Same index, but shifted values
# 
# **⚠️ Critical Concept - No Time Travel!**
# Backward shifts (`shift(-1)`) don't predict the future - they're used for:
# - Aligning datasets
# - Calculating forward-looking metrics
# - Validating strategies (using known future outcomes)
# 
# **Common Financial Applications:**
# - `shift(1)`: Yesterday's closing price
# - `shift(5)`: Last week's price (for weekly analysis)
# - `shift(21)`: Last month's price (for monthly comparison)
# 
# Let's see shifting in action:

# In[14]:


# Time shifting operations
print("📊 TIME SHIFTING OPERATIONS")
print("=" * 30)

# Create a subset for demonstration
demo_data = apple_stock[['Close', 'Volume']].tail(10).copy()

print("\n📊 Original Data (last 10 days):")
print(demo_data.round(2))

# Forward shift (lag)
print("\n📊 Forward Shift (shift +1):")
shifted_forward = demo_data.shift(1)
print(shifted_forward.round(2))

# Backward shift (lead)
print("\n📊 Backward Shift (shift -1):")
shifted_backward = demo_data.shift(-1)
print(shifted_backward.round(2))

# Note: shifted data creates NaN values at the boundaries
print("\n⚠️ Note: Shifting creates NaN values at the boundaries")
print(f"Forward shift: {shifted_forward.isnull().sum().sum()} NaN values") # why two sums?
print(f"Backward shift: {shifted_backward.isnull().sum().sum()} NaN values")


# ### 5.2 Financial Applications of Time Shifting
# 
# **💰 Real-World Time Shifting Applications**
# 
# Time shifting transforms from a technical concept to a powerful analytical tool when applied to financial problems:
# 
# **1. Momentum Analysis:**
# - Compare today's price to yesterday's: `price - price.shift(1)`
# - Calculate price change ratios: `price / price.shift(1)`
# - Identify trending vs ranging markets
# 
# **2. Volume Confirmation:**
# - Compare today's volume to yesterday's: `volume / volume.shift(1)`
# - High relative volume confirms price moves
# - Low relative volume suggests weak moves
# 
# **3. Return Persistence:**
# - Correlation between today's return and yesterday's return
# - Positive correlation = momentum
# - Negative correlation = mean reversion
# 
# **📊 Key Metrics You'll Build:**
# 
# | Metric | Formula | Interpretation |
# |--------|---------|----------------|
# | **Price Change** | `close - close.shift(1)` | Dollar amount of daily change |
# | **Price Ratio** | `close / close.shift(1)` | Relative size of today vs yesterday |
# | **Volume Ratio** | `volume / volume.shift(1)` | Relative trading activity |
# | **Weekly Change** | `close - close.shift(5)` | 5-day price change |
# 
# **🎯 Pattern Recognition:**
# - **Consistent gains**: Multiple positive daily changes
# - **Volume confirmation**: High volume on up days, low on down days
# - **Trend strength**: Larger weekly changes indicate stronger trends
# 
# Let's build these analytical tools:

# In[15]:


# Practical financial applications of time shifting
print("📊 FINANCIAL APPLICATIONS OF TIME SHIFTING")
print("=" * 46)

# Create lagged variables for analysis
analysis_data = apple_stock[['Close', 'Volume', 'Daily_Return']].copy()

# Previous day's data (lag 1)
analysis_data['Close_Lag1'] = analysis_data['Close'].shift(1)
analysis_data['Volume_Lag1'] = analysis_data['Volume'].shift(1)
analysis_data['Return_Lag1'] = analysis_data['Daily_Return'].shift(1)

# Previous week's data (lag 5)
analysis_data['Close_Lag5'] = analysis_data['Close'].shift(5)
analysis_data['Return_Lag5'] = analysis_data['Daily_Return'].shift(5)

# Calculate relative changes
analysis_data['Price_Change_1d'] = analysis_data['Close'] - analysis_data['Close_Lag1']
analysis_data['Price_Change_5d'] = analysis_data['Close'] - analysis_data['Close_Lag5']
analysis_data['Volume_Ratio'] = analysis_data['Volume'] / analysis_data['Volume_Lag1']

print("\n📊 Lagged Variables Example (last 5 days):")
lag_example = analysis_data[[
    'Close', 'Close_Lag1', 'Price_Change_1d', 
    'Daily_Return', 'Return_Lag1', 'Volume_Ratio'
]].tail(5)
print(lag_example.round(3))

# Momentum analysis using lagged returns
momentum_data = analysis_data.dropna()

# Return persistence (today's return vs yesterday's return)
return_correlation = momentum_data['Daily_Return'].corr(momentum_data['Return_Lag1'])
print(f"\n📈 Return Persistence (1-day lag correlation): {return_correlation:.4f}")

# Weekly momentum
weekly_momentum_corr = momentum_data['Daily_Return'].corr(momentum_data['Return_Lag5'])
print(f"📈 Weekly Momentum (5-day lag correlation): {weekly_momentum_corr:.4f}")

# Volume-price relationship
volume_price_corr = momentum_data['Daily_Return'].corr(momentum_data['Volume_Ratio'])
print(f"📊 Volume-Price Correlation: {volume_price_corr:.4f}")

# Trend identification using price changes
trend_signals = momentum_data.copy()
trend_signals['Trend_1d'] = np.where(trend_signals['Price_Change_1d'] > 0, 'Up', 'Down')
trend_signals['Trend_5d'] = np.where(trend_signals['Price_Change_5d'] > 0, 'Up', 'Down')

print("\n📊 Recent Trend Signals:")
trend_summary = trend_signals[['Price_Change_1d', 'Trend_1d', 'Price_Change_5d', 'Trend_5d']].tail(5)
print(trend_summary.round(2))


# ## 6. Rolling and Expanding Windows
# 
# ### 🎯 **What You'll Learn in This Section:**
# 
# Rolling and expanding windows are fundamental tools for calculating time-varying financial metrics. You'll learn to measure risk, performance, and market conditions as they evolve over time.
# 
# **Key Knowledge Points:**
# - **Rolling windows**: Fixed-period calculations that "slide" through time
# - **Expanding windows**: Growing calculations from a starting point
# - **Risk metrics**: Volatility, VaR, correlation, and Sharpe ratios
# - **Performance attribution**: Understanding when and why performance changed
# 
# **Real-World Applications:**
# - **Risk management**: Monitor portfolio volatility in real-time
# - **Performance evaluation**: Track Sharpe ratios over time
# - **Market analysis**: Identify periods of high/low volatility
# - **Portfolio optimization**: Understand changing asset correlations
# 
# **🔄 Rolling vs Expanding - The Key Difference:**
# 
# | Window Type | Calculation | Use Case | Example |
# |-------------|-------------|----------|---------|
# | **Rolling** | Fixed period, slides forward | Current conditions | 30-day volatility |
# | **Expanding** | Growing from start point | Cumulative metrics | Since-inception return |
# 
# **📊 Visual Comparison:**
# ```
# Rolling 3-day:    [1,2,3] → [2,3,4] → [3,4,5] → [4,5,6]
# Expanding:        [1] → [1,2] → [1,2,3] → [1,2,3,4] → [1,2,3,4,5]
# ```
# 
# Essential for calculating time-varying financial metrics.

# ### 6.1 Rolling Window Calculations
# 
# **📊 Advanced Risk Management with Rolling Windows**
# 
# Rolling windows are essential for risk management because they provide **current** risk estimates rather than historical averages. Financial markets change rapidly, and risk measures need to adapt.
# 
# **🎯 Critical Risk Metrics:**
# 
# **1. Rolling Volatility**
# - **Purpose**: Measure current market uncertainty
# - **Common windows**: 10d (short-term), 30d (monthly), 252d (annual)
# - **Interpretation**: Higher volatility = higher risk = potentially higher returns
# 
# **2. Rolling Correlations**
# - **Purpose**: Understand how assets move together
# - **Application**: Portfolio diversification, hedging strategies
# - **Key insight**: Correlations change over time, especially during crises
# 
# **3. Rolling Sharpe Ratio**
# - **Purpose**: Risk-adjusted return measurement
# - **Formula**: `(Return - Risk_Free_Rate) / Volatility`
# - **Interpretation**: Higher Sharpe = better risk-adjusted performance
# 
# 
# **📈 Financial Industry Standards:**
# 
# | Metric | Window | Frequency | Use Case |
# |--------|---------|-----------|----------|
# | **Volatility** | 30d | Daily | Daily risk monitoring |
# | **Correlation** | 63d | Weekly | Portfolio rebalancing |
# | **Sharpe Ratio** | 252d | Monthly | Performance evaluation |
# 
# Let's implement professional risk management tools:

# In[16]:


# Advanced rolling window calculations for risk management
print("📊 ROLLING WINDOW RISK CALCULATIONS")
print("=" * 39)

# Load multi-asset data for portfolio analysis
multi_asset_prices = pd.read_csv('financial_close_prices.csv', index_col='Date', parse_dates=True)

# Select major assets for analysis
major_assets = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
available_assets = [asset for asset in major_assets if asset in multi_asset_prices.columns]

if len(available_assets) >= 3:
    portfolio_prices = multi_asset_prices[available_assets].dropna()
    portfolio_returns = portfolio_prices.pct_change().dropna()

    print(f"\n📈 Analyzing {len(available_assets)} assets: {available_assets}")
    print(f"Data range: {portfolio_returns.index[0].date()} to {portfolio_returns.index[-1].date()}")

    # Rolling volatility (different windows)
    windows = [10, 21, 63, 252]  # 2 weeks, 1 month, 3 months, 1 year

    volatility_data = pd.DataFrame(index=portfolio_returns.index)

    for window in windows:
        col_name = f'Vol_{window}d'
        volatility_data[col_name] = portfolio_returns['AAPL'].rolling(window=window).std() * np.sqrt(252) * 100

    print(f"\n📊 AAPL Rolling Volatility (last 10 days):")
    print(volatility_data.tail(10).round(2))

    # Rolling correlations between assets
    if len(available_assets) >= 2:
        rolling_corr = portfolio_returns[available_assets[0]].rolling(window=63).corr(
            portfolio_returns[available_assets[1]]
        )

        print(f"\n📊 Rolling 63-day Correlation ({available_assets[0]} vs {available_assets[1]}):")
        print(f"Latest correlation: {rolling_corr.iloc[-1]:.3f}")
        print(f"Average correlation: {rolling_corr.mean():.3f}")
        print(f"Correlation range: {rolling_corr.min():.3f} to {rolling_corr.max():.3f}")

    # Rolling Sharpe ratio
    risk_free_rate = 0.02  # 2% annual risk-free rate
    daily_rf = risk_free_rate / 252

    for asset in available_assets[:3]:  # First 3 assets
        excess_returns = portfolio_returns[asset] - daily_rf
        rolling_sharpe = excess_returns.rolling(window=252).mean() / excess_returns.rolling(window=252).std() * np.sqrt(252)

        print(f"\n📊 {asset} - 252-day Rolling Sharpe Ratio:")
        print(f"Current: {rolling_sharpe.iloc[-1]:.3f}")
        print(f"Average: {rolling_sharpe.mean():.3f}")


else:
    print("⚠️ Insufficient asset data for portfolio analysis")


# ### 6.2 Expanding Window Analysis
# 
# **📈 Long-Term Performance Tracking with Expanding Windows**
# 
# Expanding windows show how metrics evolve from the beginning of your dataset to each point in time. They answer questions like: *"How has our Sharpe ratio changed since we started investing?"*
# 
# **🎯 Key Applications:**
# 
# **1. Performance Attribution**
# - Track how performance metrics change over time
# - Identify periods that helped or hurt overall performance
# - Understand the impact of market cycles
# 
# **2. Stability Analysis**
# - Are our risk metrics stabilizing or still changing dramatically?
# - How much data do we need for reliable estimates?
# - When did our strategy start working consistently?
# 
# **3. Drawdown Analysis**
# - **Maximum Drawdown**: Largest peak-to-trough decline
# - **Current Drawdown**: How far are we from our all-time high?
# - **Recovery Time**: How long to recover from major losses?
# 
# **📊 Essential Expanding Metrics:**
# 
# | Metric | Purpose | Insight |
# |--------|---------|---------|
# | **Expanding Return** | Cumulative performance | Total return since inception |
# | **Expanding Volatility** | Risk stability | Is risk estimate stabilizing? |
# | **Expanding Sharpe** | Risk-adjusted performance | Overall strategy quality |
# | **Maximum Drawdown** | Worst-case scenario | Maximum historical loss |
# 
# **💡 Why Expanding Windows Matter:**
# - **Regulatory reporting**: Many regulations require since-inception metrics
# - **Investor communication**: Clients want to see total performance
# - **Strategy validation**: Understand long-term strategy characteristics
# - **Risk budgeting**: Plan for worst-case scenarios based on historical experience
# 
# Let's analyze long-term performance trends:

# In[17]:


# Demonstrate expanding window syntax
print("📈 EXPANDING WINDOW SYNTAX DEMONSTRATION")
print("=" * 45)

# Basic expanding operations
print("\n1️⃣ Basic Expanding Operations:")

# Get a small subset for clear demonstration
demo_data = apple_stock["Close"].head(10)
print("Sample data (first 10 days):")
print(demo_data.round(2))

print("\n📊 Expanding Mean (cumulative average):")
expanding_mean = demo_data.expanding().mean()
print(expanding_mean.round(2))

print("\n📊 Expanding Sum (cumulative sum):")
expanding_sum = demo_data.expanding().sum()
print(expanding_sum.round(2))

print("\n📊 Expanding Standard Deviation:")
expanding_std = demo_data.expanding().std()
print(expanding_std.round(4))

print("\n📊 Expanding Min/Max:")
expanding_min = demo_data.expanding().min()
expanding_max = demo_data.expanding().max()
print("Expanding Min:")
print(expanding_min.round(2))
print("Expanding Max:")
print(expanding_max.round(2))

# Show how window size grows
print("\n2️⃣ Window Size Demonstration:")
print("Expanding window grows from 1 to N observations:")
for i in range(5):
    window_size = i + 1
    subset = demo_data.iloc[:window_size]
    exp_mean = subset.expanding().mean().iloc[-1]
    manual_mean = subset.mean()
    print(f"Day {window_size}: Window size={window_size}, Expanding mean={exp_mean:.2f}, Manual mean={manual_mean:.2f}")

print("\n3️⃣ Key Difference from Rolling Windows:")
print("Rolling window (fixed size): Uses last N observations")
print("Expanding window (growing size): Uses all observations from start to current point")
print("\n📈 Expanding window syntax demonstration completed!")


# In[18]:


# Expanding window analysis for long-term trends
print("📊 EXPANDING WINDOW ANALYSIS")
print("=" * 31)

# Use Apple stock data for expanding analysis
expanding_data = apple_stock[['Close', 'Daily_Return']].dropna()

# Expanding calculations (from start of data to current point)
expanding_data['Expanding_Mean_Return'] = expanding_data['Daily_Return'].expanding().mean() * 252 * 100
expanding_data['Expanding_Volatility'] = expanding_data['Daily_Return'].expanding().std() * np.sqrt(252) * 100
expanding_data['Expanding_Sharpe'] = (expanding_data['Daily_Return'].expanding().mean() - 0.02/252) / expanding_data['Daily_Return'].expanding().std() * np.sqrt(252)

# Expanding min/max (running extremes)
expanding_data['Expanding_Max_Price'] = expanding_data['Close'].expanding().max()
expanding_data['Expanding_Min_Price'] = expanding_data['Close'].expanding().min()
expanding_data['Drawdown'] = (expanding_data['Close'] - expanding_data['Expanding_Max_Price']) / expanding_data['Expanding_Max_Price'] * 100

expanding_data = expanding_data.dropna()
print(f"\n📊 Expanding Statistics Evolution (every 100 days):")
milestone_dates = expanding_data.iloc[::100]
display_cols = ['Expanding_Mean_Return', 'Expanding_Volatility', 'Expanding_Sharpe', 'Drawdown']
print(milestone_dates[display_cols].round(3))

print(f"\n📊 Current Expanding Statistics:")
latest_stats = expanding_data[display_cols].iloc[-1]
print(f"Annualized Return: {latest_stats['Expanding_Mean_Return']:.2f}%")
print(f"Annualized Volatility: {latest_stats['Expanding_Volatility']:.2f}%")
print(f"Sharpe Ratio: {latest_stats['Expanding_Sharpe']:.3f}")
print(f"Current Drawdown: {latest_stats['Drawdown']:.2f}%")

# Maximum drawdown analysis
max_drawdown = expanding_data['Drawdown'].min()
max_dd_date = expanding_data['Drawdown'].idxmin()
max_price_before_dd = expanding_data.loc[:max_dd_date, 'Expanding_Max_Price'].iloc[-1]
price_at_max_dd = expanding_data.loc[max_dd_date, 'Close']

print(f"\n📉 Maximum Drawdown Analysis:")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")
print(f"Date of Maximum Drawdown: {max_dd_date.date()}")
print(f"Peak Price: ${max_price_before_dd:.2f}")
print(f"Trough Price: ${price_at_max_dd:.2f}")

# Show recent expanding vs rolling metrics comparison
recent_data = expanding_data.tail(10)
rolling_vol = expanding_data['Daily_Return'].rolling(window=252).std() * np.sqrt(252) * 100

print(f"\n📊 Expanding vs Rolling Volatility Comparison (last 10 days):")
volatility_comparison = pd.DataFrame({
    'Close': recent_data['Close'],
    'Expanding_Volatility': recent_data['Expanding_Volatility'],
    'Rolling_252d_Volatility': rolling_vol.tail(10)
})
print(volatility_comparison.round(2))

print("📈 Expanding window analysis completed!")


# ## 7. Missing Data Handling in Financial Time Series
# 
# **🔧 Financial Data Quality Challenges**
# 
# Financial data is messy. Markets close for holidays, systems fail, and data feeds have gaps. Professional analysts need robust techniques to handle these realities.
# 
# **Common Missing Data Scenarios:**
# - **Market holidays**: No trading = no data
# - **System outages**: Technical failures create gaps
# - **Halted trading**: Regulatory stops create missing periods
# - **New listings**: Stocks don't have historical data before IPO
# - **Delisted securities**: Stocks stop trading and disappear
# 
# **🎯 Missing Data Strategies by Data Type:**
# 
# | Data Type | Best Strategy | Rationale |
# |-----------|---------------|-----------|
# | **Prices** | Forward fill | Last known price is best estimate |
# | **Volume** | Zero or median | No trading = zero volume, or use typical volume |
# | **Returns** | Zero | No price change when no trading |
# | **Volatility** | Interpolation | Smooth transition between known values |
# 
# **⚠️ Critical Considerations:**
# - **Don't use future data**: Only fill forward, never backward for analysis
# - **Mark interpolated data**: Know which observations are real vs estimated
# - **Validate assumptions**: Check if your filling method makes sense for the specific case
# 
# Let's implement professional-grade data cleaning:

# Method 1: Forward Fill (ffill)
#   - How it works: Uses the last valid observation to fill
#   missing values
#   - Best for: Price data where values tend to persist
#   (e.g., stock prices over weekends)
#   - Assumption: Missing values should equal the most
#   recent known value
#   - Example: If Monday's price is $100$ and Tuesday is
#   missing, fills Tuesday with $100
#   - Syntax: df['column'].fillna(method='ffill')
# 
#   Method 2: Linear Interpolation
#   - How it works: Creates a straight line between known
#   values to estimate missing points
#   - Best for: Smooth price series where gradual changes
#   are expected
#   - Assumption: Missing values follow a linear trend
#   between surrounding points
#   - Example: If Monday=$100$ and Wednesday=$110$, fills
#   Tuesday with $105$
#   - Syntax: df['column'].interpolate(method='linear')
# 
#   Method 3: Time-Based Interpolation
#   - How it works: Similar to linear but accounts for
#   actual time intervals (weekends, holidays)
#   - Best for: Financial data with irregular time spacing
#   - Assumption: Changes occur proportionally to time
#   elapsed
#   - Example: Considers that weekend gaps should be
#   weighted differently than daily gaps
#   - Syntax: df['column'].interpolate(method='time')

# In[19]:


# Advanced missing data handling for financial time series
print("📊 MISSING DATA HANDLING IN FINANCIAL TIME SERIES")
print("=" * 52)

# Create sample data with missing values to demonstrate handling techniques
sample_data = apple_stock[['Close', 'Volume']].copy()

# Artificially introduce missing values for demonstration
np.random.seed(42)
missing_indices = np.random.choice(sample_data.index, size=50, replace=False)
sample_data.loc[missing_indices, 'Close'] = np.nan

# Add some random volume missing data
volume_missing = np.random.choice(sample_data.index, size=30, replace=False)
sample_data.loc[volume_missing, 'Volume'] = np.nan

print(f"\n📊 Sample Data with Missing Values:")
print(f"Total observations: {len(sample_data)}")
print(f"Missing Close prices: {sample_data['Close'].isnull().sum()}")
print(f"Missing Volume data: {sample_data['Volume'].isnull().sum()}")
print(f"Missing percentage: Close {sample_data['Close'].isnull().mean()*100:.2f}%, Volume {sample_data['Volume'].isnull().mean()*100:.2f}%")

# Method 1: Forward fill (common for prices)
sample_data['Close_ForwardFill'] = sample_data['Close'].fillna(method='ffill')

# Method 2: Linear interpolation (good for smooth price series)
sample_data['Close_Interpolated'] = sample_data['Close'].interpolate(method='linear')

# Method 3: Time-based interpolation (accounts for weekends/holidays)
sample_data['Close_TimeInterp'] = sample_data['Close'].interpolate(method='time')

# For volume data (different strategy - use median)
sample_data['Volume_MedianFill'] = sample_data['Volume'].fillna(sample_data['Volume'].median())

print(f"\n📊 Missing Data Handling Results:")
methods = ['Close_ForwardFill', 'Close_Interpolated', 'Close_TimeInterp']
for method in methods:
    remaining_missing = sample_data[method].isnull().sum()
    print(f"{method}: {remaining_missing} missing values remaining")

# Compare methods on a subset
comparison_data = sample_data.loc[missing_indices[:10]]
comparison_cols = ['Close', 'Close_ForwardFill', 'Close_Interpolated', 'Close_TimeInterp']
print(f"\n📊 Comparison of Missing Data Methods (sample):")
print(comparison_data[comparison_cols].round(2))


# ## 8. Summary and Best Practices
# 
# ### 🎯 **What You've Accomplished:**
# 
# Congratulations! You've completed a comprehensive journey through pandas time series analysis for finance. You now have the skills that professional financial analysts use daily.
# 
# **🏆 Your New Skillset:**
# 
# **Data Management Skills:**
# - Load and structure financial data properly
# - Handle missing data and data quality issues  
# - Work with different time frequencies and calendars
# 
# **Analysis Skills:**
# - Calculate returns, volatility, and risk metrics
# - Implement technical indicators and trading signals
# - Perform portfolio analysis and risk attribution
# 
# **Advanced Techniques:**
# - Use rolling and expanding windows for dynamic analysis
# - Handle high-frequency intraday data
# - Create and backtest trading strategies
# 
# **📈 From Student to Professional Analyst:**
# 
# You now understand the difference between:
# - **Academic finance** (clean data, perfect assumptions)
# - **Real-world finance** (messy data, practical constraints)
# 
# **💼 Career Pathways:**
# - **Portfolio Manager**: Optimize asset allocation and risk management
# - **Quantitative Analyst**: Develop algorithmic trading strategies  
# - **Risk Manager**: Monitor and control portfolio risk exposure
# - **Research Analyst**: Analyze securities and market trends
# - **Data Scientist**: Apply machine learning to financial problems
# 
# ### Key Takeaways for Financial Time Series Analysis with Pandas
# 
# **🎓 Mastery Checklist:**
# - ✅ **Data Loading**: You can load financial data with proper datetime indexing
# - ✅ **Time Operations**: You can slice, resample, and shift time series data
# - ✅ **Financial Calculations**: You can calculate returns, volatility, and performance metrics
# - ✅ **Technical Analysis**: You can implement moving averages, indicators, and trading signals
# - ✅ **Portfolio Analysis**: You can analyze multi-asset portfolios and measure diversification
# - ✅ **Risk Management**: You can calculate VaR, drawdowns, and rolling risk metrics
# - ✅ **Advanced Techniques**: You can handle missing data and work with high-frequency data
# 
# **🚀 Next Steps in Your Financial Analytics Journey:**
# 1. **Practice with real data**: Download current market data and apply these techniques
# 2. **Learn domain expertise**: Understand the financial markets and instruments you're analyzing
# 3. **Expand your toolkit**: Explore specialized libraries like QuantLib, zipline, and pyfolio
# 4. **Build complete systems**: Create end-to-end analysis pipelines and automated reports

# In[20]:


# Summary of best practices and performance tips
print("📊 PANDAS TIME SERIES BEST PRACTICES SUMMARY")
print("=" * 48)

print("\n🏆 KEY TECHNIQUES MASTERED:")
print("" + "="*35)

techniques = {
    "1. Data Loading": [
        "✅ Use parse_dates=True for automatic datetime parsing",
        "✅ Set datetime column as index for time-based operations",
        "✅ Handle missing values appropriately for financial data"
    ],
    "2. Time-Based Indexing": [
        "✅ Leverage DateTimeIndex for intuitive date slicing",
        "✅ Use boolean indexing with datetime components (e.g., df[df.index.year == 2024])",
        "✅ Alternative: Use complete date strings for .loc indexing (e.g., '2024-01-01':'2024-01-31')",
        "✅ Understand business day vs calendar day operations"
    ],
    "3. Financial Calculations": [
        "✅ Calculate returns using pct_change() method",
        "✅ Use rolling windows for moving averages and volatility",
        "✅ Apply expanding windows for cumulative statistics"
    ],
    "4. Resampling": [
        "✅ Convert between frequencies (daily→weekly→monthly)",
        "✅ Choose appropriate aggregation methods (last, mean, sum)",
        "✅ Handle business day frequencies properly"
    ],
    "5. Time Shifting": [
        "✅ Create lagged variables for analysis",
        "✅ Generate trading signals using time shifts",
        "✅ Calculate period-over-period changes"
    ],
    "6. Advanced Analysis": [
        "✅ Multi-asset portfolio analysis",
        "✅ Risk metrics (VaR, Sharpe ratio, max drawdown)",
        "✅ Correlation and volatility clustering analysis"
    ]
}

for category, items in techniques.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")

print("\n💡 PERFORMANCE OPTIMIZATION TIPS:")
print("" + "="*36)

perf_tips = [
    "🚀 Use vectorized operations instead of loops",
    "🚀 Leverage pandas built-in financial functions",
    "🚀 Pre-allocate DataFrames when possible",
    "🚀 Use appropriate data types (float32 vs float64)",
    "🚀 Cache expensive calculations in new columns",
    "🚀 Use .loc and .iloc for explicit indexing",
    "🚀 Consider using categorical data for repeated strings"
]

for tip in perf_tips:
    print(f"  {tip}")

print("\n⚠️  COMMON PITFALLS TO AVOID:")
print("" + "="*30)

pitfalls = [
    "❌ Forgetting to handle missing values in financial data",
    "❌ Not accounting for weekends/holidays in date ranges",
    "❌ Using forward-looking information in backtests",
    "❌ Ignoring timezone issues in global market data",
    "❌ Not validating data quality before analysis",
    "❌ Mixing different asset classes without normalization",
    "❌ Forgetting to annualize returns and volatility"
]

for pitfall in pitfalls:
    print(f"  {pitfall}")

print("\n🎯 NEXT STEPS FOR ADVANCED FINANCE:")
print("" + "="*35)

next_steps = [
    "📈 Learn quantitative finance libraries (e.g., QuantLib, zipline)",
    "📈 Explore machine learning for financial prediction",
    "📈 Study advanced risk models (GARCH, VaR models)",
    "📈 Implement algorithmic trading strategies",
    "📈 Master portfolio optimization techniques",
    "📈 Learn options pricing and derivatives analysis",
    "📈 Explore alternative data sources (sentiment, satellite data)"
]

for step in next_steps:
    print(f"  {step}")

# Final statistics about what we've accomplished
print("\n📊 TUTORIAL ACCOMPLISHMENTS:")
print("" + "="*30)

print(f"📈 Financial datasets analyzed: 6+ (stocks, ETFs, crypto, forex)")
print(f"📈 Time series techniques demonstrated: 15+")
print(f"📈 Technical indicators calculated: 10+")
print(f"📈 pandas DataFrame operations mastered: 50+")
print(f"📈 Financial metrics computed: 20+")
print(f"📈 Code examples provided: 50+")

print("\n🎉 CONGRATULATIONS!")
print("You've completed a comprehensive tutorial on pandas time series analysis for finance.")
print("You're now equipped with the essential skills for professional financial data analysis!")

print("\n" + "="*60)
print("📚 End of Pandas Time Series Analysis for Finance Tutorial")
print("="*60)


# ---
# 
# ## 📚 Additional Resources
# 
# ### Documentation and References:
# - [Pandas Time Series Documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
# - [Financial Data Analysis with Python](https://github.com/PacktPublishing/Hands-On-Financial-Data-Analysis-with-Python)
# - [Quantitative Finance Resources](https://github.com/wilsonfreitas/awesome-quant)
# 
# ### Libraries for Advanced Finance:
# - **yfinance**: Yahoo Finance data downloading
# - **quantlib**: Quantitative finance library
# - **zipline**: Algorithmic trading library
# - **pyfolio**: Portfolio and risk analytics
# - **ta-lib**: Technical analysis library
# 
# ### Practice Datasets:
# - Yahoo Finance API
# - Alpha Vantage API
# - Quandl Financial Data
# - FRED Economic Data
# 
# ---
# 
# **© 2025 Investment Practice Course**  

# In[ ]:




