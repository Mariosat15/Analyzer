import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import os
import requests
import streamlit as st
from alpha_vantage.timeseries import TimeSeries
import pickle
import hashlib

warnings.filterwarnings('ignore')

class DataProcessor:
    """Enhanced data processor with multiple data sources and caching"""
    
    def __init__(self):
        self.months = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        # Alpha Vantage setup
        self.alpha_vantage_key = self._get_alpha_vantage_key()
        self.av_ts = None
        if self.alpha_vantage_key:
            try:
                self.av_ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            except Exception as e:
                st.warning(f"Alpha Vantage initialization failed: {e}")
        
        # Cache settings
        self.cache_dir = "data_cache"
        self.cache_duration = 3600  # 1 hour cache
        self._ensure_cache_dir()
        
        # Rate limiting
        self.last_yahoo_call = 0
        self.last_alpha_call = 0
        self.yahoo_delay = 1.0  # seconds between Yahoo calls
        self.alpha_delay = 12.0  # Alpha Vantage allows 5 calls per minute
    
    def _get_alpha_vantage_key(self):
        """Get Alpha Vantage API key from environment or Streamlit secrets"""
        # Try environment variable first
        key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Try Streamlit secrets
        if not key:
            try:
                key = st.secrets["ALPHA_VANTAGE_API_KEY"]
            except:
                pass
        
        # Use demo key if no key provided (limited but works for testing)
        if not key:
            key = None  # Don't use demo key as it has severe limitations
            
        return key
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, symbol, start_date, end_date, source=""):
        """Generate cache key for data"""
        key_string = f"{symbol}_{start_date}_{end_date}_{source}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _save_to_cache(self, data, cache_key):
        """Save data to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            cache_data = {
                'data': data,
                'timestamp': time.time()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_key):
        """Load data from cache if valid"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Check if cache is still valid
                if time.time() - cache_data['timestamp'] < self.cache_duration:
                    return cache_data['data']
        except Exception as e:
            print(f"Failed to load cache: {e}")
        
        return None
    
    def _rate_limit_yahoo(self):
        """Apply rate limiting for Yahoo Finance"""
        current_time = time.time()
        time_since_last = current_time - self.last_yahoo_call
        if time_since_last < self.yahoo_delay:
            time.sleep(self.yahoo_delay - time_since_last)
        self.last_yahoo_call = time.time()
    
    def _rate_limit_alpha(self):
        """Apply rate limiting for Alpha Vantage"""
        current_time = time.time()
        time_since_last = current_time - self.last_alpha_call
        if time_since_last < self.alpha_delay:
            time.sleep(self.alpha_delay - time_since_last)
        self.last_alpha_call = time.time()
    
    def _fetch_yahoo_data(self, symbol, start_date, end_date):
        """Fetch data from Yahoo Finance with rate limiting"""
        try:
            self._rate_limit_yahoo()
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data if not data.empty else None
        except Exception as e:
            print(f"Yahoo Finance error for {symbol}: {str(e)}")
            return None
    
    def _fetch_alpha_vantage_data(self, symbol, start_date, end_date):
        """Fetch data from Alpha Vantage with rate limiting"""
        if not self.av_ts or not self.alpha_vantage_key:
            return None
            
        try:
            self._rate_limit_alpha()
            
            # Alpha Vantage returns full data, we'll filter by date
            data, meta_data = self.av_ts.get_daily_adjusted(symbol=symbol, outputsize='full')
            
            if data is not None and not data.empty:
                # Rename columns to match Yahoo Finance format
                data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
                data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                
                # Filter by date range
                data.index = pd.to_datetime(data.index)
                data = data.sort_index()
                mask = (data.index >= start_date) & (data.index <= end_date)
                data = data.loc[mask]
                
                return data if not data.empty else None
            
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {str(e)}")
            return None
        
        return None
    
    def _fetch_sample_data(self, symbol, start_date, end_date):
        """Generate realistic sample data with seasonal patterns (for demo purposes)"""
        try:
            # Generate synthetic data for demonstration
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            # Filter out weekends
            date_range = date_range[date_range.weekday < 5]
            
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed based on symbol
            
            # Generate realistic price movements with seasonal patterns
            n_days = len(date_range)
            base_price = 100 + (hash(symbol) % 500)  # Base price between 100-600
            
            # Create seasonal effects (demo patterns)
            seasonal_multipliers = {
                1: 1.02,   # January Effect
                2: 0.99,   # February dip
                3: 1.01,   # March recovery
                4: 0.98,   # April sell-off
                5: 0.97,   # Sell in May
                6: 0.98,   # June weakness
                7: 1.00,   # July neutral
                8: 0.99,   # August doldrums
                9: 0.96,   # September worst month
                10: 1.01,  # October recovery
                11: 1.02,  # November rally
                12: 1.03   # December Santa Rally
            }
            
            # Weekday effects
            weekday_effects = {
                0: 0.98,   # Monday Effect (negative)
                1: 1.00,   # Tuesday neutral
                2: 1.01,   # Wednesday positive
                3: 1.00,   # Thursday neutral
                4: 1.01    # Friday positive
            }
            
            prices = [base_price]
            
            for i, date in enumerate(date_range[1:], 1):
                # Base random return
                base_return = np.random.normal(0.0005, 0.02)
                
                # Apply seasonal effect
                month_effect = seasonal_multipliers.get(date.month, 1.0)
                weekday_effect = weekday_effects.get(date.weekday(), 1.0)
                
                # Combine effects (subtle influence)
                seasonal_factor = (month_effect - 1) * 0.1  # 10% of the seasonal effect
                weekday_factor = (weekday_effect - 1) * 0.05  # 5% of weekday effect
                
                adjusted_return = base_return + seasonal_factor + weekday_factor
                new_price = prices[-1] * (1 + adjusted_return)
                prices.append(new_price)
            
            # Create OHLC data
            opens = []
            highs = []
            lows = []
            closes = prices
            volumes = []
            
            for i, close_price in enumerate(closes):
                # Generate realistic OHLC
                daily_volatility = np.random.uniform(0.005, 0.025)
                
                # Open slightly different from previous close
                if i == 0:
                    open_price = close_price
                else:
                    gap = np.random.normal(0, 0.002)
                    open_price = closes[i-1] * (1 + gap)
                
                opens.append(open_price)
                
                # High and low based on open and close
                high_low_range = max(open_price, close_price) * daily_volatility
                high = max(open_price, close_price) + np.random.uniform(0, high_low_range)
                low = min(open_price, close_price) - np.random.uniform(0, high_low_range)
                
                highs.append(high)
                lows.append(max(low, 0.01))  # Prevent negative prices
                
                # Volume with some pattern
                base_volume = 1000000 + (hash(f"{symbol}_{i}") % 5000000)
                volumes.append(base_volume)
            
            # Create DataFrame
            data = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Adj Close': closes,
                'Volume': volumes
            }, index=date_range)
            
            return data
            
        except Exception as e:
            print(f"Sample data generation error: {e}")
            return None

    def fetch_stock_data(self, symbol, years_back=10, start_date=None, end_date=None):
        """
        Fetch stock data with multiple data sources and fallback
        
        Args:
            symbol (str): Stock symbol
            years_back (int): Years of historical data (if dates not provided)
            start_date (datetime): Custom start date
            end_date (datetime): Custom end date
        
        Returns:
            pd.DataFrame: Stock data with returns and metadata
        """
        # Determine date range
        if start_date is not None and end_date is not None:
            fetch_start = start_date
            fetch_end = end_date
        else:
            fetch_end = datetime.now()
            fetch_start = fetch_end - timedelta(days=years_back * 365)
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, fetch_start.strftime('%Y-%m-%d'), 
                                       fetch_end.strftime('%Y-%m-%d'))
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            st.info(f"📦 Using cached data for {symbol}")
            processed_data = self._process_stock_data(cached_data)
            if not processed_data.empty:
                processed_data.attrs['data_source'] = 'Cache'
            return processed_data
        
        # Try data sources in order of preference
        data_sources = []
        
        # Always try Yahoo Finance first
        data_sources.append(("Yahoo Finance", self._fetch_yahoo_data))
        
        # Add Alpha Vantage if API key is available
        if self.alpha_vantage_key:
            data_sources.append(("Alpha Vantage", self._fetch_alpha_vantage_data))
        else:
            st.info("💡 To avoid rate limits, get a free Alpha Vantage API key at: https://www.alphavantage.co/support/#api-key")
        
        # Always have sample data as fallback
        data_sources.append(("Sample Data (Demo)", self._fetch_sample_data))
        
        for source_name, fetch_func in data_sources:
            try:
                st.info(f"🔄 Fetching {symbol} data from {source_name}...")
                data = fetch_func(symbol, fetch_start, fetch_end)
                
                if data is not None and not data.empty:
                    # Cache successful fetch
                    self._save_to_cache(data, cache_key)
                    st.success(f"✅ Successfully fetched {symbol} data from {source_name}")
                    
                    processed_data = self._process_stock_data(data)
                    if not processed_data.empty:
                        processed_data.attrs['data_source'] = source_name
                    return processed_data
                    
            except Exception as e:
                st.warning(f"⚠️ {source_name} failed: {str(e)}")
                continue
        
        # If all sources fail
        st.error(f"❌ All data sources failed for {symbol}")
        return pd.DataFrame()
    
    def _process_stock_data(self, data):
        """Process raw stock data into analysis format"""
        try:
            if data.empty:
                return pd.DataFrame()
            
            # Calculate returns and add metadata
            data['Returns'] = data['Close'].pct_change()
            data['Month'] = data.index.month
            data['Year'] = data.index.year
            data['Month_Name'] = data.index.month_name()
            data['Weekday'] = data.index.dayofweek  # 0=Monday, 6=Sunday
            data['Weekday_Name'] = data.index.day_name()
            
            # Remove first row with NaN return
            data = data.dropna()
            
            return data
            
        except Exception as e:
            print(f"Error processing stock data: {str(e)}")
            return pd.DataFrame()

    def calculate_seasonal_stats(self, stock_data):
        """
        Calculate seasonal statistics by month
        
        Args:
            stock_data (pd.DataFrame): Stock data with returns
        
        Returns:
            pd.DataFrame: Monthly seasonal statistics
        """
        try:
            # Calculate monthly returns from daily returns
            monthly_data = stock_data.groupby(['Year', 'Month']).agg({
                'Returns': 'sum'  # Sum daily returns within each month
            }).reset_index()
            
            # Keep as decimal for proper percentage formatting
            monthly_data['Monthly_Return'] = monthly_data['Returns']
            
            # Add month names
            monthly_data['Month_Name'] = monthly_data['Month'].apply(
                lambda x: self.months[x-1]
            )
            
            # Calculate statistics by month
            monthly_stats = monthly_data.groupby('Month_Name').agg({
                'Monthly_Return': ['mean', 'std', 'min', 'max', 'count']
            }).round(2)
            
            # Flatten column names
            monthly_stats.columns = ['Avg_Return', 'Volatility', 'Min_Return', 'Max_Return', 'Count']
            
            # Calculate win rate (decimal between 0-1)
            win_rates = monthly_data.groupby('Month_Name')['Monthly_Return'].apply(
                lambda x: (x > 0).sum() / len(x)
            ).round(3)
            
            monthly_stats['Win_Rate'] = win_rates
            
            # Reorder by calendar month
            month_order = self.months
            monthly_stats = monthly_stats.reindex(month_order)
            
            return monthly_stats
            
        except Exception as e:
            print(f"Error calculating seasonal stats: {str(e)}")
            return pd.DataFrame()
    
    def get_quarterly_stats(self, stock_data):
        """
        Calculate quarterly seasonal statistics
        
        Args:
            stock_data (pd.DataFrame): Stock data with returns
        
        Returns:
            pd.DataFrame: Quarterly seasonal statistics
        """
        try:
            # Map months to quarters
            quarter_map = {1: 'Q1', 2: 'Q1', 3: 'Q1',
                          4: 'Q2', 5: 'Q2', 6: 'Q2',
                          7: 'Q3', 8: 'Q3', 9: 'Q3',
                          10: 'Q4', 11: 'Q4', 12: 'Q4'}
            
            stock_data['Quarter'] = stock_data['Month'].map(quarter_map)
            
            # Calculate quarterly returns
            quarterly_data = stock_data.groupby(['Year', 'Quarter']).agg({
                'Returns': 'sum'
            }).reset_index()
            
            quarterly_data['Quarterly_Return'] = quarterly_data['Returns']
            
            # Calculate statistics by quarter
            quarterly_stats = quarterly_data.groupby('Quarter').agg({
                'Quarterly_Return': ['mean', 'std', 'min', 'max', 'count']
            }).round(2)
            
            # Flatten column names
            quarterly_stats.columns = ['Avg_Return', 'Volatility', 'Min_Return', 'Max_Return', 'Count']
            
            # Calculate win rate (decimal between 0-1)
            win_rates = quarterly_data.groupby('Quarter')['Quarterly_Return'].apply(
                lambda x: (x > 0).sum() / len(x)
            ).round(3)
            
            quarterly_stats['Win_Rate'] = win_rates
            
            # Reorder quarters
            quarterly_stats = quarterly_stats.reindex(['Q1', 'Q2', 'Q3', 'Q4'])
            
            return quarterly_stats
            
        except Exception as e:
            print(f"Error calculating quarterly stats: {str(e)}")
            return pd.DataFrame()
    
    def calculate_rolling_seasonality(self, stock_data, window=5):
        """
        Calculate rolling seasonal patterns to detect changes over time
        
        Args:
            stock_data (pd.DataFrame): Stock data with returns
            window (int): Rolling window in years
        
        Returns:
            pd.DataFrame: Rolling seasonal statistics
        """
        try:
            results = []
            unique_years = sorted(stock_data['Year'].unique())
            
            for i in range(window, len(unique_years) + 1):
                year_subset = unique_years[i-window:i]
                data_subset = stock_data[stock_data['Year'].isin(year_subset)]
                
                seasonal_stats = self.calculate_seasonal_stats(data_subset)
                seasonal_stats['Period'] = f"{year_subset[0]}-{year_subset[-1]}"
                
                results.append(seasonal_stats)
            
            return pd.concat(results, keys=[r['Period'].iloc[0] for r in results])
            
        except Exception as e:
            print(f"Error calculating rolling seasonality: {str(e)}")
            return pd.DataFrame()
    
    def detect_earnings_impact(self, symbol, stock_data):
        """
        Analyze impact of earnings announcements on seasonal patterns
        
        Args:
            symbol (str): Stock symbol
            stock_data (pd.DataFrame): Stock data
        
        Returns:
            dict: Earnings impact analysis
        """
        try:
            # This is a simplified version - in practice, you'd fetch actual earnings dates
            # For now, we'll use common earnings months for large cap stocks
            typical_earnings_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
            
            earnings_impact = {}
            
            for month in typical_earnings_months:
                month_name = self.months[month-1]
                month_data = stock_data[stock_data['Month'] == month]
                
                if not month_data.empty:
                    avg_return = month_data['Returns'].mean()
                    volatility = month_data['Returns'].std()
                    
                    earnings_impact[month_name] = {
                        'avg_return': round(avg_return, 2),
                        'volatility': round(volatility, 2),
                        'is_earnings_month': True
                    }
            
            return earnings_impact
            
        except Exception as e:
            print(f"Error analyzing earnings impact: {str(e)}")
            return {}
    
    def calculate_sector_comparison(self, symbols_list, years_back=5):
        """
        Compare seasonal patterns across multiple symbols/sectors
        
        Args:
            symbols_list (list): List of stock symbols
            years_back (int): Years of historical data
        
        Returns:
            pd.DataFrame: Comparison of seasonal patterns
        """
        try:
            comparison_data = {}
            
            for symbol in symbols_list:
                stock_data = self.fetch_stock_data(symbol, years_back)
                if not stock_data.empty:
                    seasonal_stats = self.calculate_seasonal_stats(stock_data)
                    comparison_data[symbol] = seasonal_stats['Avg_Return']
            
            if comparison_data:
                return pd.DataFrame(comparison_data).round(2)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error in sector comparison: {str(e)}")
            return pd.DataFrame()
    
    def get_market_regime_analysis(self, stock_data, market_data):
        """
        Analyze seasonal patterns in different market regimes (bull/bear)
        
        Args:
            stock_data (pd.DataFrame): Individual stock data
            market_data (pd.DataFrame): Market index data (e.g., SPY)
        
        Returns:
            dict: Seasonal patterns by market regime
        """
        try:
            # Define bull/bear markets based on 200-day moving average
            market_data['MA200'] = market_data['Close'].rolling(200).mean()
            market_data['Regime'] = np.where(
                market_data['Close'] > market_data['MA200'], 'Bull', 'Bear'
            )
            
            # Merge with stock data
            merged_data = stock_data.merge(
                market_data[['Regime']], 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            
            regime_analysis = {}
            
            for regime in ['Bull', 'Bear']:
                regime_data = merged_data[merged_data['Regime'] == regime]
                if not regime_data.empty:
                    seasonal_stats = self.calculate_seasonal_stats(regime_data)
                    regime_analysis[regime] = seasonal_stats
            
            return regime_analysis
            
        except Exception as e:
            print(f"Error in market regime analysis: {str(e)}")
            return {}
    
    def calculate_weekday_stats(self, stock_data):
        """
        Calculate seasonal statistics by weekday
        
        Args:
            stock_data (pd.DataFrame): Stock data with returns
        
        Returns:
            pd.DataFrame: Weekday seasonal statistics
        """
        try:
            # Calculate daily returns by weekday (keep as decimal)
            weekday_data = stock_data.copy()
            weekday_data['Daily_Return'] = weekday_data['Returns']
            
            # Calculate statistics by weekday
            weekday_stats = weekday_data.groupby('Weekday_Name').agg({
                'Daily_Return': ['mean', 'std', 'min', 'max', 'count']
            })
            
            # Flatten column names
            weekday_stats.columns = ['Avg_Return', 'Volatility', 'Min_Return', 'Max_Return', 'Count']
            
            # Calculate win rate (decimal between 0-1)
            win_rates = weekday_data.groupby('Weekday_Name')['Daily_Return'].apply(
                lambda x: (x > 0).sum() / len(x)
            ).round(3)
            
            weekday_stats['Win_Rate'] = win_rates
            
            # Reorder by calendar weekday (Monday first)
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            available_weekdays = [day for day in weekday_order if day in weekday_stats.index]
            weekday_stats = weekday_stats.reindex(available_weekdays)
            
            return weekday_stats
            
        except Exception as e:
            print(f"Error calculating weekday stats: {str(e)}")
            return pd.DataFrame()

    def get_intraday_patterns(self, stock_data):
        """
        Analyze intraday patterns and market sessions
        
        Args:
            stock_data (pd.DataFrame): Stock data
        
        Returns:
            dict: Intraday analysis results
        """
        try:
            results = {}
            
            # Calculate first hour effect (first day returns vs rest)
            stock_data['Day_of_Month'] = stock_data.index.day
            first_day_returns = stock_data[stock_data['Day_of_Month'] == 1]['Returns'].mean()
            other_day_returns = stock_data[stock_data['Day_of_Month'] != 1]['Returns'].mean()
            
            results['first_of_month_effect'] = {
                'first_day_avg': round(first_day_returns, 5),
                'other_days_avg': round(other_day_returns, 5),
                'difference': round(first_day_returns - other_day_returns, 5)
            }
            
            # End of month effect (last 3 days vs others)
            stock_data['Days_in_Month'] = stock_data.index.to_series().dt.days_in_month
            stock_data['Days_from_End'] = stock_data['Days_in_Month'] - stock_data['Day_of_Month']
            
            end_month_returns = stock_data[stock_data['Days_from_End'] <= 2]['Returns'].mean()
            mid_month_returns = stock_data[stock_data['Days_from_End'] > 2]['Returns'].mean()
            
            results['end_of_month_effect'] = {
                'end_month_avg': round(end_month_returns, 5),
                'mid_month_avg': round(mid_month_returns, 5),
                'difference': round(end_month_returns - mid_month_returns, 5)
            }
            
            return results
            
        except Exception as e:
            print(f"Error calculating intraday patterns: {str(e)}")
            return {}
    
    def process_uploaded_data(self, df, symbol, start_date=None, end_date=None):
        """Process uploaded CSV data and convert to standard format"""
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Detect and handle different CSV formats
            if self._is_mt5_format(data):
                # MT5/MetaTrader format (like CADJPY_H1)
                data = self._process_mt5_format(data)
            elif self._is_stock_format(data):
                # Stock format (like AAPL daily)
                data = self._process_stock_format(data)
            else:
                # Try to auto-detect and process
                data = self._auto_process_format(data)
            
            # Filter by date range if specified
            if start_date and end_date:
                mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
                data = data.loc[mask]
            
            # Process the data (calculate returns, etc.)
            processed_data = self._process_stock_data(data)
            
            # Add metadata
            processed_data.attrs = {
                'data_source': 'Uploaded CSV File',
                'symbol': symbol,
                'processed_date': datetime.now().isoformat()
            }
            
            return processed_data
            
        except Exception as e:
            st.error(f"Error processing uploaded data: {str(e)}")
            return pd.DataFrame()
    
    def _is_mt5_format(self, df):
        """Check if data is in MT5/MetaTrader format"""
        # MT5 format typically has: <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, <VOL>, <SPREAD>
        cols = [col.upper().replace('<', '').replace('>', '') for col in df.columns]
        mt5_indicators = ['DATE', 'TIME', 'TICKVOL', 'SPREAD']
        return any(indicator in cols for indicator in mt5_indicators)
    
    def _is_stock_format(self, df):
        """Check if data is in stock format"""
        # Stock format typically has: <DATE>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, <VOL>, <SPREAD>
        cols = [col.upper().replace('<', '').replace('>', '') for col in df.columns]
        stock_indicators = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']
        return all(indicator in cols for indicator in stock_indicators)
    
    def _process_mt5_format(self, df):
        """Process MT5/MetaTrader format data"""
        data = df.copy()
        
        # Clean column names
        data.columns = [col.replace('<', '').replace('>', '') for col in data.columns]
        
        # Combine DATE and TIME if both exist
        if 'TIME' in data.columns:
            # Combine date and time
            data['DateTime'] = pd.to_datetime(data['DATE'].astype(str) + ' ' + data['TIME'].astype(str))
            data = data.set_index('DateTime')
        else:
            # Use DATE only
            data['DATE'] = pd.to_datetime(data['DATE'])
            data = data.set_index('DATE')
        
        # Rename columns to standard format
        column_mapping = {
            'OPEN': 'Open',
            'HIGH': 'High', 
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOL': 'Volume',
            'TICKVOL': 'Volume'  # Use TICKVOL as Volume if VOL is not available
        }
        
        # Apply mapping
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data[new_col] = data[old_col]
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in data.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Add Volume if missing
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        
        # Add Adj Close if missing
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        
        # Select only the standard columns
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        data = data[standard_cols]
        
        # Convert to numeric
        for col in standard_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _process_stock_format(self, df):
        """Process stock format data"""
        data = df.copy()
        
        # Clean column names
        data.columns = [col.replace('<', '').replace('>', '') for col in data.columns]
        
        # Set date as index
        if 'DATE' in data.columns:
            data['DATE'] = pd.to_datetime(data['DATE'])
            data = data.set_index('DATE')
        
        # Rename columns to standard format
        column_mapping = {
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low', 
            'CLOSE': 'Close',
            'VOL': 'Volume',
            'TICKVOL': 'Volume'
        }
        
        # Apply mapping
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data[new_col] = data[old_col]
        
        # Add missing columns
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        
        # Select standard columns
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        data = data[standard_cols]
        
        # Convert to numeric
        for col in standard_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _auto_process_format(self, df):
        """Auto-detect and process unknown formats"""
        data = df.copy()
        
        # Try to find date column
        date_cols = []
        for col in data.columns:
            if any(word in col.upper() for word in ['DATE', 'TIME', 'DATETIME']):
                date_cols.append(col)
        
        if date_cols:
            # Use the first date column as index
            data[date_cols[0]] = pd.to_datetime(data[date_cols[0]])
            data = data.set_index(date_cols[0])
        
        # Try to map columns based on common patterns
        column_mapping = {}
        for col in data.columns:
            col_upper = col.upper()
            if 'OPEN' in col_upper:
                column_mapping[col] = 'Open'
            elif 'HIGH' in col_upper:
                column_mapping[col] = 'High'
            elif 'LOW' in col_upper:
                column_mapping[col] = 'Low'
            elif 'CLOSE' in col_upper:
                column_mapping[col] = 'Close'
            elif 'VOL' in col_upper and 'TICK' not in col_upper:
                column_mapping[col] = 'Volume'
            elif 'VOLUME' in col_upper:
                column_mapping[col] = 'Volume'
        
        # Apply mapping
        data = data.rename(columns=column_mapping)
        
        # Add missing columns
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        if 'Adj Close' not in data.columns and 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        
        # Try to get at least OHLC
        required_cols = ['Open', 'High', 'Low', 'Close']
        available_cols = [col for col in required_cols if col in data.columns]
        
        if len(available_cols) < 4:
            # If we don't have OHLC, try to use first 4 numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 4:
                data['Open'] = data[numeric_cols[0]]
                data['High'] = data[numeric_cols[1]]
                data['Low'] = data[numeric_cols[2]]
                data['Close'] = data[numeric_cols[3]]
                if len(numeric_cols) >= 5:
                    data['Volume'] = data[numeric_cols[4]]
                else:
                    data['Volume'] = 0
                data['Adj Close'] = data['Close']
        
        # Select standard columns
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_standard = [col for col in standard_cols if col in data.columns]
        
        if len(available_standard) >= 4:  # At least OHLC
            data = data[available_standard]
            
            # Convert to numeric
            for col in available_standard:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data
        else:
            st.error("Could not identify required price columns (Open, High, Low, Close) in the uploaded file.")
            return pd.DataFrame() 