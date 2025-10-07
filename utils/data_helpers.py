import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import time
import logging

logger = logging.getLogger(__name__)

def clean_price_column(price_series):
    """Clean price columns by removing currency symbols and converting to numeric"""
    try:
        return pd.to_numeric(price_series.str.replace('$', '').str.replace('€', '').str.replace(',', ''), errors='coerce')
    except Exception:
        return price_series

def calculate_percentage_change(current, previous):
    """Calculate percentage change between two values"""
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return 0
    return ((current - previous) / previous) * 100

def validate_coordinates(lat, lon):
    """Validate latitude and longitude coordinates"""
    try:
        lat_f = float(lat)
        lon_f = float(lon)

        if -90 <= lat_f <= 90 and -180 <= lon_f <= 180:
            return lat_f, lon_f
        else:
            return None, None
    except Exception:
        return None, None

def fetch_with_retry(url, max_retries=3, timeout=10):
    """Fetch URL with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limited
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                break
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)
    
    return None

def safe_divide(numerator, denominator):
    """Safely divide two numbers, return 0 if denominator is 0"""
    try:
        if denominator == 0 or pd.isna(denominator):
            return 0
        return numerator / denominator
    except Exception:
        return 0

def format_currency(amount, currency='EUR'):
    """Format amount as currency"""
    try:
        if currency == 'EUR':
            return f"€{amount:,.0f}"
        elif currency == 'USD':
            return f"${amount:,.0f}"
        else:
            return f"{amount:,.0f} {currency}"
    except Exception:
        return str(amount)

def extract_numeric_from_string(text):
    """Extract first numeric value from string"""
    import re
    try:
        numbers = re.findall(r'\d+\.?\d*', str(text))
        if numbers:
            return float(numbers[0])
        return 0
    except Exception:
        return 0

def create_date_range(start_date, end_date, freq='D'):
    """Create pandas date range"""
    try:
        return pd.date_range(start=start_date, end=end_date, freq=freq)
    except Exception:
        return pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq=freq)

def handle_missing_data(df, strategy='drop'):
    """Handle missing data in DataFrame"""
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill_mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    elif strategy == 'fill_zero':
        return df.fillna(0)
    else:
        return df

def normalize_text(text):
    """Normalize text for analysis"""
    try:
        import re
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())  # Remove extra whitespace
        return text
    except Exception:
        return str(text)

def calculate_correlation_matrix(df, method='pearson'):
    """Calculate correlation matrix for numeric columns"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)
    except Exception:
        return pd.DataFrame()

def get_api_key(key_name, default=None):
    """Safely get API key from environment variables"""
    return os.getenv(key_name, default)

def validate_dataframe(df, required_columns=None):
    """Validate DataFrame structure"""
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def aggregate_by_time_period(df, date_column, value_column, period='M'):
    """Aggregate data by time period"""
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df_grouped = df.groupby(pd.Grouper(key=date_column, freq=period))[value_column].agg(['mean', 'sum', 'count'])
        return df_grouped
    except Exception as e:
        logger.error(f"Error in time aggregation: {e}")
        return pd.DataFrame()
