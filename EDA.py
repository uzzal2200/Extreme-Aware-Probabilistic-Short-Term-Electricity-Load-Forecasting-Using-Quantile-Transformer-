"""
Exploratory Data Analysis (EDA) for ERCOT Load Forecasting
Extreme-Aware Probabilistic Short-Term Electricity Load Forecasting
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Reproducibility
np.random.seed(42)

# Configuration
DATA_PATH = 'Data/Final_dataset_ERCOT_v2.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - ERCOT LOAD FORECASTING")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[STEP 1] Loading and Preparing Data...")

# Load data
df = pd.read_csv(DATA_PATH)

# Basic validation
required_cols = ['timestamp', 'tmpc', 'relh', 'sped', 'feel', 'p01m', 'ERCOT']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f'Missing columns: {missing_cols}')

# Parse datetime and sort
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Remove duplicate timestamps
dup_count = df['timestamp'].duplicated().sum()
if dup_count > 0:
    df = df.drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
    print(f'Removed duplicates: {dup_count}')

# Handle missing values: time-based interpolation for numeric columns
df = df.set_index('timestamp')
numeric_cols = ['tmpc', 'relh', 'sped', 'feel', 'p01m', 'ERCOT']
df[numeric_cols] = df[numeric_cols].interpolate(method='time')
df = df.reset_index()

# Drop any remaining NaNs
na_before = df.isnull().sum().sum()
df = df.dropna()
na_after = df.isnull().sum().sum()

# Rename target column
df = df.rename(columns={'ERCOT': 'Load'})
target_col = 'Load'

print(f'Data shape: {df.shape}')
print(f'Remaining NaNs: {na_after} (before drop: {na_before})')
print("\nFirst few rows:")
print(df.head())

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 2] Feature Engineering...")

# Time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['weekend_flag'] = (df['day_of_week'] >= 5).astype(int)

# Lag features (target lags)
df['lag_1'] = df['Load'].shift(1)
df['lag_24'] = df['Load'].shift(24)
df['lag_168'] = df['Load'].shift(168)

# Rolling features
df['rolling_mean_24'] = df['Load'].rolling(window=24).mean()
df['rolling_std_24'] = df['Load'].rolling(window=24).std()

# Extreme module
temp_90th = df['tmpc'].quantile(0.90)
df['extreme_temperature_flag'] = (df['tmpc'] > temp_90th).astype(int)

# Drop NaNs introduced by lag/rolling
before_drop = len(df)
df = df.dropna().reset_index(drop=True)
after_drop = len(df)

print(f'Dropped rows after feature engineering: {before_drop - after_drop}')
print("\nFeature engineered data:")
print(df[['timestamp', 'Load', 'tmpc', 'lag_1', 'lag_24', 'lag_168']].head())

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS - VISUALIZATIONS
# ============================================================================
print("\n[STEP 3] Creating EDA Visualizations...")

# ============================================================================
# [1] LOAD TIME SERIES
# ============================================================================
print("\n[1] Load Time Series Plot")
fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=380)
axes.plot(df['timestamp'], df['Load'], linewidth=0.5)
axes.set_title('Load Time Series')
axes.set_ylabel('Load (MWh)')
axes.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/EDA_01_load_time_series.png', dpi=380, bbox_inches='tight')
print("[SUCCESS] Load time series saved")
plt.close()

# ============================================================================
# [2] LOAD VS TEMPERATURE
# ============================================================================
print("\n[2] Load vs Temperature Scatter Plot")
fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=380)
axes.scatter(df['tmpc'], df['Load'], s=5, alpha=0.3)
axes.set_title('Load vs Temperature')
axes.set_xlabel('Temperature (C)')
axes.set_ylabel('Load (MWh)')
axes.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/EDA_02_load_vs_temperature.png', dpi=380, bbox_inches='tight')
print("[SUCCESS] Load vs temperature saved")
plt.close()

# ============================================================================
# [3] CORRELATION HEATMAP
# ============================================================================
print("\n[3] Correlation Heatmap")
fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=380)
numeric_cols = ['Load', 'tmpc', 'relh', 'sped', 'feel', 'p01m', 'hour']
corr_data = df[numeric_cols].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', ax=axes)
axes.set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('results/EDA_03_correlation_heatmap.png', dpi=380, bbox_inches='tight')
print("[SUCCESS] Correlation heatmap saved")
plt.close()

# ============================================================================
# [4] EXTREME EVENT DISTRIBUTION
# ============================================================================
print("\n[4] Extreme Event Distribution")
fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=380)
axes.hist(df[df['extreme_temperature_flag'] == 0]['Load'], bins=50, alpha=0.6, label='Normal')
axes.hist(df[df['extreme_temperature_flag'] == 1]['Load'], bins=50, alpha=0.6, label='Extreme')
axes.set_title('Load Distribution: Normal vs Extreme Temperature')
axes.set_xlabel('Load (MWh)')
axes.set_ylabel('Frequency')
axes.legend()
axes.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/EDA_04_extreme_event_distribution.png', dpi=380, bbox_inches='tight')
print("[SUCCESS] Extreme event distribution saved")
plt.close()

# ============================================================================
# [5] MONTHLY PATTERN
# ============================================================================
print("\n[5] Monthly Load Pattern")
fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=380)
monthly_load = df.groupby('month')['Load'].mean()
axes.bar(monthly_load.index, monthly_load.values, color='steelblue')
axes.set_title('Average Monthly Load')
axes.set_xlabel('Month')
axes.set_ylabel('Average Load (MWh)')
axes.set_xticks(range(1, 13))
axes.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/EDA_05_monthly_load_pattern.png', dpi=380, bbox_inches='tight')
print("[SUCCESS] Monthly load pattern saved")
plt.close()

# ============================================================================
# [6] SEASONAL DECOMPOSITION
# ============================================================================
print("\n[6] Seasonal Decomposition Analysis (Period=168 = Weekly Pattern)")

decomposition = seasonal_decompose(df[target_col], model='additive', period=168)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), dpi=100)

axes[0].plot(decomposition.observed, color='blue', linewidth=1)
axes[0].set_ylabel('Observed', fontweight='bold')
axes[0].set_title('Time Series Decomposition (Additive Model)', fontweight='bold', fontsize=12)
axes[0].grid(alpha=0.3)

axes[1].plot(decomposition.trend, color='red', linewidth=2)
axes[1].set_ylabel('Trend', fontweight='bold')
axes[1].grid(alpha=0.3)

axes[2].plot(decomposition.seasonal, color='green', linewidth=1)
axes[2].set_ylabel('Seasonal', fontweight='bold')
axes[2].grid(alpha=0.3)

axes[3].plot(decomposition.resid, color='orange', linewidth=1)
axes[3].set_ylabel('Residual', fontweight='bold')
axes[3].set_xlabel('Time Index')
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/EDA_06_seasonal_decomposition.png', dpi=380, bbox_inches='tight')
print("[SUCCESS] Seasonal decomposition saved")
plt.close()

print(f"Seasonal range: {decomposition.seasonal.max() - decomposition.seasonal.min():.2f} MWh")
print(f"Trend direction: {decomposition.trend.dropna().iloc[-1] - decomposition.trend.dropna().iloc[0]:.2f} MWh")

# ============================================================================
# [7] LAG ANALYSIS - TEMPORAL DEPENDENCIES
# ============================================================================
print("\n[7] Lag Analysis - Temporal Dependencies")

load_array = df[target_col].values
max_lag = 168

# Calculate correlations at key lags
lag_1_corr, lag_1_p = pearsonr(load_array[:-1], load_array[1:])
lag_24_corr, lag_24_p = pearsonr(load_array[:-24], load_array[24:])
lag_168_corr, lag_168_p = pearsonr(load_array[:-168], load_array[168:])

print(f"\nLag Correlations with Load:")
print(f"  Lag-1 (30 min):     r = {lag_1_corr:.4f} (p = {lag_1_p:.2e})")
print(f"  Lag-24 (12 hours):  r = {lag_24_corr:.4f} (p = {lag_24_p:.2e})")
print(f"  Lag-168 (1 week):   r = {lag_168_corr:.4f} (p = {lag_168_p:.2e})")

# Create lag correlation visualization
fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=100)

# Scatter plot: Lag-1
axes[0, 0].scatter(load_array[:-1], load_array[1:], alpha=0.5, s=10, color='blue')
axes[0, 0].set_xlabel('Load(t)', fontweight='bold')
axes[0, 0].set_ylabel('Load(t+1)', fontweight='bold')
axes[0, 0].set_title(f'Lag-1 Correlation: r={lag_1_corr:.4f}', fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Scatter plot: Lag-24
axes[0, 1].scatter(load_array[:-24], load_array[24:], alpha=0.5, s=10, color='green')
axes[0, 1].set_xlabel('Load(t)', fontweight='bold')
axes[0, 1].set_ylabel('Load(t+24)', fontweight='bold')
axes[0, 1].set_title(f'Lag-24 Correlation: r={lag_24_corr:.4f}', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Scatter plot: Lag-168
axes[1, 0].scatter(load_array[:-168], load_array[168:], alpha=0.5, s=10, color='red')
axes[1, 0].set_xlabel('Load(t)', fontweight='bold')
axes[1, 0].set_ylabel('Load(t+168)', fontweight='bold')
axes[1, 0].set_title(f'Lag-168 Correlation: r={lag_168_corr:.4f}', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Autocorrelation function curve
lags_range = list(range(1, max_lag+1))
acf_values = [pearsonr(load_array[:-lag], load_array[lag:])[0] for lag in lags_range]

axes[1, 1].plot(lags_range, acf_values, color='purple', linewidth=2, marker='o', markersize=3)
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.7, label='95% CI')
axes[1, 1].axhline(y=-0.05, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1, 1].axvline(x=24, color='green', linestyle='--', alpha=0.5, label='Daily cycle (24h)')
axes[1, 1].axvline(x=168, color='blue', linestyle='--', alpha=0.5, label='Weekly cycle (168h)')
axes[1, 1].set_xlabel('Lag (hours)', fontweight='bold')
axes[1, 1].set_ylabel('Correlation', fontweight='bold')
axes[1, 1].set_title('Autocorrelation vs Lag', fontweight='bold')
axes[1, 1].legend(loc='upper right', fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/EDA_07_lag_analysis.png', dpi=100, bbox_inches='tight')
print("[SUCCESS] Lag analysis saved")
plt.close()

# ============================================================================
# [8] LOAD DISTRIBUTION ANALYSIS - BOXPLOTS
# ============================================================================
print("\n[8] Load Distribution Analysis - Boxplots by Hour/Day/Month")

# Create timestamp features for boxplots
df['datetime'] = pd.to_datetime(df['timestamp'])
df['day_name'] = df['datetime'].dt.day_name()
df['month_name'] = df['datetime'].dt.month_name()

# Create 3-panel boxplot figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=380)

# Hourly boxplot
df.boxplot(column=target_col, by='hour', ax=axes[0])
axes[0].set_title('Load Distribution by Hour of Day', fontweight='bold', fontsize=11)
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Load (MWh)')
axes[0].grid(alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0)

# Day of week boxplot
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_data = [df[df['day_name'] == day][target_col].values for day in day_order]
bp1 = axes[1].boxplot(day_data, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
axes[1].set_title('Load Distribution by Day of Week', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Load (MWh)')
axes[1].grid(alpha=0.3)

# Monthly boxplot
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month_data = [df[df['month_name'] == month][target_col].values for month in month_order]
bp2 = axes[2].boxplot(month_data, labels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightcoral')
axes[2].set_title('Load Distribution by Month', fontweight='bold', fontsize=11)
axes[2].set_ylabel('Load (MWh)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/EDA_08_load_distribution_boxplot.png', dpi=100, bbox_inches='tight')
print("[SUCCESS] Load distribution boxplots saved")
plt.close()

print(f"\nHourly Stats:")
hourly_stats = df.groupby('hour')[target_col].agg(['mean', 'std', 'min', 'max'])
print(f"  Peak hour: {hourly_stats['mean'].idxmax()}:00 ({hourly_stats['mean'].max():.0f} MWh)")
print(f"  Min hour: {hourly_stats['mean'].idxmin()}:00 ({hourly_stats['mean'].min():.0f} MWh)")

print(f"\nWeekly Stats:")
weekly_stats = df.groupby('day_name')[target_col].agg(['mean', 'std'])
print(f"  Peak day: {weekly_stats['mean'].idxmax()} ({weekly_stats['mean'].max():.0f} MWh)")

print(f"\nMonthly Stats:")
monthly_stats = df.groupby('month_name')[target_col].agg(['mean', 'std'])
print(f"  Peak month: {monthly_stats['mean'].idxmax()} ({monthly_stats['mean'].max():.0f} MWh)")

# ============================================================================
# [9] TEMPERATURE & WEATHER RELATIONSHIP ANALYSIS
# ============================================================================
print("\n[9] Temperature & Weather Relationship Analysis")

# Calculate correlations
temp_cols = ['tmpc', 'relh', 'sped', 'feel']
correlations = {}
for col in temp_cols:
    if col in df.columns:
        corr = df[[col, target_col]].corr().iloc[0, 1]
        correlations[col] = corr
        print(f"\n{col.upper()} correlation with Load: {corr:.4f}")

# Create comprehensive weather analysis plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)

# Temperature vs Load scatter
if 'tmpc' in df.columns:
    axes[0, 0].scatter(df['tmpc'], df[target_col], alpha=0.3, s=10, color='red')
    z = np.polyfit(df['tmpc'].dropna(), df[target_col][df['tmpc'].notna()], 2)
    p = np.poly1d(z)
    x_line = np.linspace(df['tmpc'].min(), df['tmpc'].max(), 100)
    axes[0, 0].plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend')
    corr_tmpc = correlations.get('tmpc', 0)
    axes[0, 0].set_xlabel('Temperature (C)', fontweight='bold')
    axes[0, 0].set_ylabel('Load (MWh)', fontweight='bold')
    axes[0, 0].set_title(f'Temperature vs Load (r={correlations.get("tmpc", 0):.3f})', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

# Relative Humidity vs Load scatter
if 'relh' in df.columns:
    axes[0, 1].scatter(df['relh'], df[target_col], alpha=0.3, s=10, color='blue')
    corr_relh = correlations.get('relh', 0)
    axes[0, 1].set_xlabel('Relative Humidity (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Load (MWh)', fontweight='bold')
    axes[0, 1].set_title(f'Humidity vs Load (r={corr_relh:.3f})', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

# Feels Like Temperature vs Load
if 'feel' in df.columns:
    axes[1, 0].scatter(df['feel'], df[target_col], alpha=0.3, s=10, color='orange')
    corr_feel = correlations.get('feel', 0)
    axes[1, 0].set_xlabel('Feels Like Temp (C)', fontweight='bold')
    axes[1, 0].set_ylabel('Load (MWh)', fontweight='bold')
    axes[1, 0].set_title(f'Feels Like vs Load (r={corr_feel:.3f})', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

# Wind Speed vs Load
if 'sped' in df.columns:
    axes[1, 1].scatter(df['sped'], df[target_col], alpha=0.3, s=10, color='green')
    corr_sped = correlations.get('sped', 0)
    axes[1, 1].set_xlabel('Wind Speed (m/s)', fontweight='bold')
    axes[1, 1].set_ylabel('Load (MWh)', fontweight='bold')
    axes[1, 1].set_title(f'Wind Speed vs Load (r={corr_sped:.3f})', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/EDA_09_temperature_weather_analysis.png', dpi=100, bbox_inches='tight')
print("[SUCCESS] Temperature & weather analysis saved")
plt.close()

# Seasonal temperature profile
print("\n\nSeasonal Temperature Profile:")
seasonal_temp = df.groupby('month_name')[['tmpc', target_col]].mean()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
for month in month_order:
    if month in seasonal_temp.index:
        temp = seasonal_temp.loc[month, 'tmpc']
        load = seasonal_temp.loc[month, target_col]
        print(f"  {month}: Avg Temp {temp:.1f}C, Avg Load {load:.0f} MWh")

# ============================================================================
# [10] AUTOCORRELATION PLOT
# ============================================================================
print("\n[10] Autocorrelation Plot")
fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=380)
plot_acf(df['Load'].values, lags=48, ax=axes)
axes.set_title('Autocorrelation (Load)')
axes.set_xlabel('Lag (hours)')
axes.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/EDA_10_autocorrelation.png', dpi=380, bbox_inches='tight')
print("[SUCCESS] Autocorrelation plot saved")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nBasic Statistics:")
print(df[['Load', 'tmpc', 'relh', 'sped', 'feel', 'p01m']].describe())

print("\n\nExtreme Temperature Events:")
normal_count = (df['extreme_temperature_flag'] == 0).sum()
extreme_count = (df['extreme_temperature_flag'] == 1).sum()
print(f"  Normal conditions: {normal_count} ({normal_count/len(df)*100:.1f}%)")
print(f"  Extreme conditions: {extreme_count} ({extreme_count/len(df)*100:.1f}%)")

print("\n\nLoad Statistics by Extreme Flag:")
print(df.groupby('extreme_temperature_flag')['Load'].describe())

print("\n" + "="*80)
print("EDA COMPLETED - All visualizations saved in 'results/' directory")
print("="*80)
