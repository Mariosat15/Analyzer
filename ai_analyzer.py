import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class AIAnalyzer:
    """Comprehensive AI-powered pattern detection and forecasting for seasonal analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.prophet_models = {}
        self.forecast_horizons = [30, 60, 90, 180, 365]  # Multiple forecast periods
        
    def analyze_patterns(self, stock_data, seasonal_stats, confidence_threshold=0.75):
        """
        Comprehensive AI analysis using multiple advanced techniques
        """
        insights = {
            'high_confidence': [],
            'anomalies': [],
            'predictions': [],
            'pattern_strength': {},
            'seasonal_trends': [],
            'risk_insights': [],
            'trading_strategies': [],
            'prophet_analysis': {},
            'time_series_decomposition': {},
            'market_regimes': [],
            'cyclical_patterns': [],
            'forecast_accuracy': {},
            'structural_breaks': [],
            'volatility_insights': [],
            'correlation_analysis': {},
            'advanced_metrics': {}
        }
        
        try:
            # 1. Advanced Prophet Analysis (if available)
            if PROPHET_AVAILABLE:
                prophet_insights = self._comprehensive_prophet_analysis(stock_data, seasonal_stats)
                insights['prophet_analysis'] = prophet_insights
                insights['predictions'].extend(prophet_insights.get('forecasts', []))
                insights['high_confidence'].extend(prophet_insights.get('high_confidence_patterns', []))
                insights['market_regimes'].extend(prophet_insights.get('regime_changes', []))
            
            # 2. Time Series Decomposition Analysis
            if STATSMODELS_AVAILABLE:
                decomposition = self._advanced_time_series_analysis(stock_data)
                insights['time_series_decomposition'] = decomposition
                insights['cyclical_patterns'].extend(decomposition.get('cyclical_insights', []))
                insights['structural_breaks'].extend(decomposition.get('structural_breaks', []))
            
            # 3. Enhanced Statistical Pattern Detection
            statistical_patterns = self._enhanced_statistical_patterns(seasonal_stats, stock_data, confidence_threshold)
            insights['high_confidence'].extend(statistical_patterns)
            
            # 4. Advanced Machine Learning Analysis
            ml_insights = self._advanced_ml_analysis(stock_data, seasonal_stats, confidence_threshold)
            insights['high_confidence'].extend(ml_insights.get('patterns', []))
            insights['advanced_metrics'].update(ml_insights.get('metrics', {}))
            
            # 5. Comprehensive Anomaly Detection
            anomaly_insights = self._comprehensive_anomaly_detection(stock_data, seasonal_stats)
            insights['anomalies'].extend(anomaly_insights)
            
            # 6. Advanced Seasonal Trend Analysis
            seasonal_insights = self._advanced_seasonal_analysis(seasonal_stats, stock_data)
            insights['seasonal_trends'].extend(seasonal_insights)
            
            # 7. Comprehensive Risk Analysis
            risk_insights = self._comprehensive_risk_analysis(seasonal_stats, stock_data)
            insights['risk_insights'].extend(risk_insights)
            insights['volatility_insights'].extend(risk_insights)
            
            # 8. Advanced Trading Strategy Generation
            strategy_insights = self._advanced_strategy_generation(seasonal_stats, stock_data, insights)
            insights['trading_strategies'].extend(strategy_insights)
            
            # 9. Market Regime Detection
            regime_insights = self._detect_market_regimes(stock_data)
            insights['market_regimes'].extend(regime_insights)
            
            # 10. Cross-Asset Correlation Analysis
            correlation_insights = self._analyze_correlations(stock_data)
            insights['correlation_analysis'] = correlation_insights
            
            # 11. Enhanced Pattern Strength Assessment
            insights['pattern_strength'] = self._comprehensive_pattern_assessment(seasonal_stats, stock_data, insights)
            
            # 12. Forecast Accuracy Assessment
            if PROPHET_AVAILABLE:
                accuracy_metrics = self._assess_forecast_accuracy(stock_data)
                insights['forecast_accuracy'] = accuracy_metrics
            
            return insights
            
        except Exception as e:
            print(f"Error in comprehensive AI analysis: {str(e)}")
            return None
    
    def _detect_statistical_patterns(self, seasonal_stats, confidence_threshold):
        """Detect statistically significant seasonal patterns"""
        patterns = []
        
        try:
            for month in seasonal_stats.index:
                stats = seasonal_stats.loc[month]
                
                # High win rate with sufficient data
                if stats['Win_Rate'] >= 0.7 and stats['Count'] >= 5:
                    confidence = min(0.95, stats['Win_Rate'] * (stats['Count'] / 10))
                    
                    if confidence >= confidence_threshold:
                        patterns.append({
                            'pattern': f"Strong {month} Pattern",
                            'confidence': confidence,
                            'description': f"{month} shows {stats['Win_Rate']:.1%} win rate with {stats['Avg_Return']:.1%} average return over {stats['Count']} years"
                        })
                
                # Consistent negative performance
                elif stats['Win_Rate'] <= 0.3 and stats['Count'] >= 5:
                    confidence = min(0.95, (1 - stats['Win_Rate']) * (stats['Count'] / 10))
                    
                    if confidence >= confidence_threshold:
                        patterns.append({
                            'pattern': f"Weak {month} Pattern",
                            'confidence': confidence,
                            'description': f"{month} shows {stats['Win_Rate']:.1%} win rate with {stats['Avg_Return']:.1%} average return - consider avoiding"
                        })
                
                # High volatility months
                if stats.get('Volatility', 0) > seasonal_stats['Volatility'].mean() * 1.5:
                    patterns.append({
                        'pattern': f"High Volatility in {month}",
                        'confidence': 0.8,
                        'description': f"{month} shows above-average volatility ({stats.get('Volatility', 0):.1%}) - higher risk/reward potential"
                    })
            
            return patterns
            
        except Exception as e:
            print(f"Error in statistical pattern detection: {str(e)}")
            return []
    
    def _detect_ml_patterns(self, stock_data, seasonal_stats, confidence_threshold):
        """Use machine learning to detect complex patterns"""
        patterns = []
        
        try:
            # Feature engineering
            features = self._engineer_features(stock_data)
            
            if features.empty:
                return patterns
            
            # Random Forest for pattern classification
            if len(features) > 50:  # Minimum data requirement
                rf_patterns = self._random_forest_analysis(features, confidence_threshold)
                patterns.extend(rf_patterns)
            
            # XGBoost analysis (if available)
            if XGBOOST_AVAILABLE and len(features) > 100:
                xgb_patterns = self._xgboost_analysis(features, confidence_threshold)
                patterns.extend(xgb_patterns)
            
            return patterns
            
        except Exception as e:
            print(f"Error in ML pattern detection: {str(e)}")
            return []
    
    def _engineer_features(self, stock_data):
        """Create features for machine learning analysis"""
        try:
            features = stock_data.copy()
            
            # Time-based features
            features['Month'] = features.index.month
            features['Quarter'] = features.index.quarter
            features['Week_of_Year'] = features.index.isocalendar().week
            features['Day_of_Week'] = features.index.dayofweek
            features['Day_of_Month'] = features.index.day
            features['Day_of_Year'] = features.index.dayofyear
            features['Is_Month_Start'] = features.index.is_month_start.astype(int)
            features['Is_Month_End'] = features.index.is_month_end.astype(int)
            features['Is_Quarter_Start'] = features.index.is_quarter_start.astype(int)
            features['Is_Quarter_End'] = features.index.is_quarter_end.astype(int)
            
            # Technical indicators
            features['SMA_20'] = features['Close'].rolling(20).mean()
            features['SMA_50'] = features['Close'].rolling(50).mean()
            features['Price_vs_SMA20'] = features['Close'] / features['SMA_20'] - 1
            features['Price_vs_SMA50'] = features['Close'] / features['SMA_50'] - 1
            
            # Volatility features
            features['Volatility_20'] = features['Returns'].rolling(20).std()
            features['Volatility_Regime'] = np.where(
                features['Volatility_20'] > features['Volatility_20'].median(), 1, 0
            )
            
            # Return features
            features['Return_5d'] = features['Returns'].rolling(5).sum()
            features['Return_20d'] = features['Returns'].rolling(20).sum()
            features['Positive_Return'] = (features['Returns'] > 0).astype(int)
            
            # Lag features
            features['Return_lag1'] = features['Returns'].shift(1)
            features['Return_lag5'] = features['Returns'].shift(5)
            features['Return_lag20'] = features['Returns'].shift(20)
            
            # Drop rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            print(f"Error in feature engineering: {str(e)}")
            return pd.DataFrame()
    
    def _random_forest_analysis(self, features, confidence_threshold):
        """Use Random Forest to detect patterns"""
        patterns = []
        
        try:
            # Prepare target variable (next day positive return)
            target = (features['Returns'].shift(-1) > 0).astype(int)
            
            # Select features for modeling
            feature_cols = ['Month', 'Day_of_Year', 'Week_of_Year', 'Quarter',
                           'Is_Month_Start', 'Is_Month_End', 'Is_Quarter_Start', 'Is_Quarter_End',
                           'Price_vs_SMA20', 'Price_vs_SMA50', 'Volatility_Regime',
                           'Return_5d', 'Return_20d', 'Return_lag1', 'Return_lag5']
            
            # Filter available columns
            available_cols = [col for col in feature_cols if col in features.columns]
            
            if len(available_cols) < 5:
                return patterns
            
            X = features[available_cols]
            y = target
            
            # Remove NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                return patterns
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Feature importance analysis
            feature_importance = pd.DataFrame({
                'feature': available_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Identify most important seasonal features
            seasonal_features = ['Month', 'Quarter', 'Is_Month_Start', 'Is_Month_End', 
                               'Is_Quarter_Start', 'Is_Quarter_End']
            
            top_seasonal = feature_importance[
                feature_importance['feature'].isin(seasonal_features)
            ].head(3)
            
            accuracy = accuracy_score(y_test, rf.predict(X_test))
            
            if accuracy > 0.55:  # Better than random
                for _, row in top_seasonal.iterrows():
                    if row['importance'] > 0.1:
                        patterns.append({
                            'pattern': f"ML-Detected {row['feature']} Pattern",
                            'confidence': min(0.95, accuracy + row['importance']),
                            'description': f"Machine learning identified {row['feature']} as important seasonal factor (accuracy: {accuracy:.1%})"
                        })
            
            return patterns
            
        except Exception as e:
            print(f"Error in Random Forest analysis: {str(e)}")
            return []
    
    def _xgboost_analysis(self, features, confidence_threshold):
        """Use XGBoost for advanced pattern detection"""
        patterns = []
        
        try:
            if not XGBOOST_AVAILABLE:
                return patterns
            
            # Similar to Random Forest but with XGBoost
            target = (features['Returns'].shift(-1) > 0).astype(int)
            
            feature_cols = ['Month', 'Day_of_Year', 'Week_of_Year', 'Quarter',
                           'Price_vs_SMA20', 'Price_vs_SMA50', 'Volatility_Regime',
                           'Return_5d', 'Return_20d', 'Return_lag1']
            
            available_cols = [col for col in feature_cols if col in features.columns]
            
            if len(available_cols) < 5:
                return patterns
            
            X = features[available_cols]
            y = target
            
            # Remove NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:
                return patterns
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train, y_train)
            
            accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
            
            # Feature importance
            importance_dict = xgb_model.get_booster().get_score(importance_type='weight')
            
            if accuracy > 0.57 and importance_dict:
                patterns.append({
                    'pattern': "XGBoost Advanced Pattern",
                    'confidence': min(0.95, accuracy),
                    'description': f"XGBoost detected complex seasonal interactions (accuracy: {accuracy:.1%})"
                })
            
            return patterns
            
        except Exception as e:
            print(f"Error in XGBoost analysis: {str(e)}")
            return []
    
    def _detect_anomalies(self, stock_data, seasonal_stats):
        """Detect seasonal anomalies using Isolation Forest"""
        anomalies = []
        
        try:
            # Create monthly aggregated data
            monthly_data = stock_data.groupby(['Year', 'Month']).agg({
                'Returns': 'sum',
                'Close': 'last'
            }).reset_index()
            
            monthly_data['Monthly_Return'] = monthly_data['Returns'] * 100
            
            # For each month, detect anomalies
            for month in range(1, 13):
                month_data = monthly_data[monthly_data['Month'] == month]
                
                if len(month_data) < 5:
                    continue
                
                # Apply Isolation Forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(month_data[['Monthly_Return']].values)
                
                anomaly_years = month_data[outliers == -1]['Year'].tolist()
                
                if anomaly_years:
                    month_name = ['January', 'February', 'March', 'April', 'May', 'June',
                                 'July', 'August', 'September', 'October', 'November', 'December'][month-1]
                    
                    anomalies.append({
                        'type': f"Anomalous {month_name} Performance",
                        'description': f"Unusual returns detected in {month_name} for years: {', '.join(map(str, anomaly_years))}"
                    })
            
            return anomalies
            
        except Exception as e:
            print(f"Error in anomaly detection: {str(e)}")
            return []
    
    def _generate_forecasts(self, stock_data):
        """Advanced Prophet forecasting with multiple horizons and comprehensive analysis"""
        predictions = []
        
        try:
            from prophet import Prophet
            from prophet.diagnostics import cross_validation, performance_metrics
            from prophet.plot import add_changepoints_to_plot
            
            # Prepare data for Prophet
            stock_data_reset = stock_data.reset_index()
            ds_series = pd.to_datetime(stock_data_reset['DATE']).reset_index(drop=True)
            
            # Multiple forecasting targets
            forecasting_targets = {
                'price': stock_data['Close'].pct_change().cumsum().reset_index(drop=True),
                'returns': stock_data['Returns'].reset_index(drop=True),
                'volatility': stock_data['Returns'].rolling(22).std().reset_index(drop=True)
            }
            
            for target_name, y_series in forecasting_targets.items():
                try:
                    prophet_data = pd.DataFrame({
                        'ds': ds_series,
                        'y': y_series
                    }).dropna()
                    
                    if len(prophet_data) < 100:
                        continue
                    
                    # Advanced Prophet model with holidays and multiple seasonalities
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10.0,
                        holidays_prior_scale=10.0,
                        changepoint_range=0.8,
                        uncertainty_samples=1000
                    )
                    
                    # Add custom seasonalities
                    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    
                    # Add economic calendar holidays (simplified)
                    holidays = pd.DataFrame({
                        'holiday': 'market_close',
                        'ds': pd.to_datetime(['2020-01-01', '2020-07-04', '2020-12-25', 
                                              '2021-01-01', '2021-07-04', '2021-12-25',
                                              '2022-01-01', '2022-07-04', '2022-12-25',
                                              '2023-01-01', '2023-07-04', '2023-12-25',
                                              '2024-01-01', '2024-07-04', '2024-12-25']),
                        'lower_window': -1,
                        'upper_window': 1,
                    })
                    
                    # Fit model
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(prophet_data)
                    
                    self.prophet_models[target_name] = model
                    
                    # Generate forecasts for multiple horizons
                    for horizon in self.forecast_horizons:
                        future = model.make_future_dataframe(periods=horizon)
                        forecast = model.predict(future)
                        
                        # Extract key insights
                        latest_forecast = forecast.tail(horizon)
                        trend_direction = "Bullish" if latest_forecast['trend'].iloc[-1] > latest_forecast['trend'].iloc[0] else "Bearish"
                        
                        # Calculate confidence based on uncertainty
                        uncertainty = (latest_forecast['yhat_upper'] - latest_forecast['yhat_lower']).mean()
                        confidence = max(0.5, min(0.95, 1 - (uncertainty / abs(latest_forecast['yhat'].mean()))))
                        
                        # Seasonal component analysis
                        seasonal_strength = abs(latest_forecast['yearly'].std()) if 'yearly' in latest_forecast else 0
                        
                        predictions.append({
                            'Target': target_name.title(),
                            'Horizon': f"{horizon} days",
                            'Trend_Direction': trend_direction,
                            'Confidence': f"{confidence:.1%}",
                            'Seasonal_Strength': f"{seasonal_strength:.3f}",
                            'Expected_Value': f"{latest_forecast['yhat'].iloc[-1]:.4f}",
                            'Lower_Bound': f"{latest_forecast['yhat_lower'].iloc[-1]:.4f}",
                            'Upper_Bound': f"{latest_forecast['yhat_upper'].iloc[-1]:.4f}",
                            'Details': f"Prophet {target_name} forecast with {len(prophet_data)} historical points"
                        })
                    
                    # Changepoint analysis
                    changepoints = model.changepoints
                    if len(changepoints) > 0:
                        recent_changepoints = changepoints[-3:]  # Last 3 changepoints
                        for cp in recent_changepoints:
                            predictions.append({
                                'Target': 'Regime Change',
                                'Horizon': 'Historical',
                                'Trend_Direction': 'Structural Break',
                                'Confidence': '85%',
                                'Details': f"Significant trend change detected on {cp.strftime('%Y-%m-%d')}",
                                'Type': 'changepoint'
                            })
                    
                    # Cross-validation for model validation
                    if len(prophet_data) > 365:  # Need sufficient data for CV
                        try:
                            df_cv = cross_validation(
                                model, 
                                initial='365 days', 
                                period='90 days', 
                                horizon='30 days'
                            )
                            df_p = performance_metrics(df_cv)
                            
                            # Extract performance metrics
                            mape = df_p['mape'].mean()
                            mae = df_p['mae'].mean()
                            
                            predictions.append({
                                'Target': f"{target_name.title()} Validation",
                                'Horizon': 'Cross-Validation',
                                'MAPE': f"{mape:.2%}",
                                'MAE': f"{mae:.4f}",
                                'Details': f"Cross-validation over {len(df_cv)} periods",
                                'Type': 'validation'
                            })
                        except Exception as cv_e:
                            print(f"Cross-validation failed for {target_name}: {str(cv_e)}")
                
                except Exception as model_e:
                    print(f"Prophet model failed for {target_name}: {str(model_e)}")
                    continue
            
            # Component analysis across all models
            if self.prophet_models:
                predictions.append({
                    'Target': 'Model Summary',
                    'Horizon': 'Analysis Complete',
                    'Models_Built': str(len(self.prophet_models)),
                    'Confidence': 'High',
                    'Details': f"Successfully built {len(self.prophet_models)} Prophet models with advanced seasonality detection"
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error in advanced Prophet forecasting: {str(e)}")
            return []
    
    def _assess_pattern_strength(self, seasonal_stats):
        """Assess overall strength of seasonal patterns"""
        try:
            # 1. Consistency score (low volatility across months)
            volatility_range = seasonal_stats['Volatility'].max() - seasonal_stats['Volatility'].min()
            consistency_score = max(0, 1 - (volatility_range / 20))  # Normalize to 0-1
            
            # 2. Win rate score (higher win rates = stronger patterns)
            avg_win_rate = seasonal_stats['Win_Rate'].mean()
            win_rate_score = min(1, avg_win_rate / 0.7)  # 70% win rate = perfect score
            
            # 3. Reliability score (based on data points)
            min_count = seasonal_stats['Count'].min()
            reliability_score = min(1, min_count / 10)
            
            # 4. Return magnitude score
            avg_abs_return = seasonal_stats['Avg_Return'].abs().mean()
            return_score = min(1, avg_abs_return / 0.05)  # 5% average return = perfect score
            
            # Overall strength
            overall_strength = (consistency_score + win_rate_score + reliability_score + return_score) / 4
            
            return {
                'overall_strength': overall_strength,
                'consistency': consistency_score,
                'win_rate_quality': win_rate_score,
                'reliability': reliability_score,
                'return_magnitude': return_score,
                'interpretation': self._interpret_strength(overall_strength)
            }
            
        except Exception as e:
            print(f"Error in pattern strength assessment: {str(e)}")
            return {}
    
    def _interpret_strength(self, strength_score):
        """Interpret the pattern strength score"""
        if strength_score >= 0.8:
            return "Very Strong - Highly reliable seasonal patterns detected"
        elif strength_score >= 0.6:
            return "Strong - Reliable seasonal patterns with good consistency"
        elif strength_score >= 0.4:
            return "Moderate - Some seasonal patterns but with limitations"
        elif strength_score >= 0.2:
            return "Weak - Limited seasonal patterns, use with caution"
        else:
            return "Very Weak - No significant seasonal patterns detected"
    
    def _analyze_seasonal_trends(self, seasonal_stats):
        """Analyze seasonal trends and patterns"""
        trends = []
        
        try:
            # Identify the best performing months
            best_months = seasonal_stats.nlargest(3, 'Avg_Return')
            worst_months = seasonal_stats.nsmallest(3, 'Avg_Return')
            
            for idx, (month, stats) in enumerate(best_months.iterrows()):
                trends.append({
                    'type': 'seasonal_strength',
                    'month': month,
                    'description': f"{month} is the #{idx+1} best performing month with {stats['Avg_Return']:.3%} average return and {stats['Win_Rate']:.1%} win rate",
                    'confidence': min(0.95, stats['Win_Rate']),
                    'recommendation': 'Consider increasing position size during this month'
                })
            
            for idx, (month, stats) in enumerate(worst_months.iterrows()):
                trends.append({
                    'type': 'seasonal_weakness',
                    'month': month,
                    'description': f"{month} is the #{idx+1} worst performing month with {stats['Avg_Return']:.3%} average return and {stats['Win_Rate']:.1%} win rate",
                    'confidence': min(0.95, (1 - stats['Win_Rate'])),
                    'recommendation': 'Consider reducing exposure or hedging during this month'
                })
            
            # Identify high volatility periods
            high_vol_months = seasonal_stats[seasonal_stats['Volatility'] > seasonal_stats['Volatility'].mean() * 1.3]
            for month, stats in high_vol_months.iterrows():
                trends.append({
                    'type': 'high_volatility',
                    'month': month,
                    'description': f"{month} shows high volatility ({stats['Volatility']:.3%}) - expect larger price swings",
                    'confidence': 0.8,
                    'recommendation': 'Adjust position sizing for increased volatility'
                })
            
            return trends
            
        except Exception as e:
            print(f"Error in seasonal trend analysis: {str(e)}")
            return []
    
    def _generate_risk_insights(self, seasonal_stats, stock_data):
        """Generate risk management insights"""
        insights = []
        
        try:
            # Risk concentration analysis
            total_positive_months = seasonal_stats[seasonal_stats['Avg_Return'] > 0].shape[0]
            risk_concentration = (12 - total_positive_months) / 12
            
            if risk_concentration > 0.5:
                insights.append({
                    'type': 'high_risk_concentration',
                    'description': f"High risk concentration: {total_positive_months}/12 months are positive on average",
                    'recommendation': 'Consider diversifying across different assets or time periods',
                    'severity': 'high'
                })
            
            # Drawdown risk analysis
            worst_month = seasonal_stats.loc[seasonal_stats['Min_Return'].idxmin()]
            if worst_month['Min_Return'] < -0.15:
                insights.append({
                    'type': 'severe_drawdown_risk',
                    'description': f"Severe drawdown risk in {worst_month.name}: worst month was {worst_month['Min_Return']:.3%}",
                    'recommendation': 'Implement stop-loss mechanisms during high-risk periods',
                    'severity': 'high'
                })
            
            # Volatility clustering
            high_vol_count = seasonal_stats[seasonal_stats['Volatility'] > seasonal_stats['Volatility'].mean() * 1.5].shape[0]
            if high_vol_count >= 3:
                insights.append({
                    'type': 'volatility_clustering',
                    'description': f"{high_vol_count} months show significantly higher volatility",
                    'recommendation': 'Prepare for volatility clustering - adjust position sizing accordingly',
                    'severity': 'medium'
                })
            
            return insights
            
        except Exception as e:
            print(f"Error generating risk insights: {str(e)}")
            return []
    
    def _generate_trading_strategies(self, seasonal_stats, stock_data):
        """Generate actionable trading strategies"""
        strategies = []
        
        try:
            # Seasonal momentum strategy
            strong_months = seasonal_stats[(seasonal_stats['Avg_Return'] > 0.02) & (seasonal_stats['Win_Rate'] > 0.6)]
            if not strong_months.empty:
                strategy_months = list(strong_months.index)
                avg_return = strong_months['Avg_Return'].mean()
                avg_win_rate = strong_months['Win_Rate'].mean()
                
                strategies.append({
                    'name': 'Seasonal Momentum Strategy',
                    'type': 'momentum',
                    'description': f"Increase allocation during {', '.join(strategy_months)}",
                    'expected_return': f"{avg_return:.1%} per month",
                    'win_rate': f"{avg_win_rate:.0%}",
                    'implementation': f"Consider 1.5x normal position size during {', '.join(strategy_months)}"
                })
            
            # Seasonal contrarian strategy
            weak_months = seasonal_stats[(seasonal_stats['Avg_Return'] < -0.01) & (seasonal_stats['Win_Rate'] < 0.4)]
            if not weak_months.empty:
                strategy_months = list(weak_months.index)
                
                strategies.append({
                    'name': 'Seasonal Contrarian Strategy',
                    'type': 'contrarian',
                    'description': f"Reduce exposure or hedge during {', '.join(strategy_months)}",
                    'implementation': f"Consider 0.5x normal position size or protective puts during {', '.join(strategy_months)}"
                })
            
            # Mean reversion strategy
            volatile_months = seasonal_stats[seasonal_stats['Volatility'] > seasonal_stats['Volatility'].mean() * 1.3]
            if not volatile_months.empty:
                strategies.append({
                    'name': 'Volatility Timing Strategy',
                    'type': 'volatility',
                    'description': f"Capitalize on higher volatility during {', '.join(volatile_months.index)}",
                    'implementation': "Consider options strategies or shorter holding periods during high volatility months"
                })
            
            # Calendar spread strategy
            best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
            worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
            
            if best_month['Avg_Return'] - worst_month['Avg_Return'] > 0.05:
                spread_value = best_month['Avg_Return'] - worst_month['Avg_Return']
                strategies.append({
                    'name': 'Calendar Spread Strategy',
                    'type': 'spread',
                    'description': f"Long {best_month.name} / Short {worst_month.name} seasonal spread",
                    'expected_spread': f"{spread_value:.1%}",
                    'implementation': f"Enter long positions in {best_month.name}, exit before {worst_month.name}"
                })
            
            return strategies
            
        except Exception as e:
            print(f"Error generating trading strategies: {str(e)}")
            return []
    
    def _comprehensive_prophet_analysis(self, stock_data, seasonal_stats):
        """Perform comprehensive Prophet analysis"""
        insights = {
            'forecasts': [],
            'high_confidence_patterns': [],
            'regime_changes': [],
            'cyclical_patterns': [],
            'structural_breaks': [],
            'market_regimes': []
        }
        
        try:
            # 1. Time Series Decomposition Analysis
            decomposition = self._advanced_time_series_analysis(stock_data)
            insights['time_series_decomposition'] = decomposition
            insights['cyclical_patterns'].extend(decomposition.get('cyclical_insights', []))
            insights['structural_breaks'].extend(decomposition.get('structural_breaks', []))
            
            # 2. Prophet Forecasting
            forecasts = self._generate_forecasts(stock_data)
            insights['forecasts'].extend(forecasts)
            
            # 3. Market Regime Detection
            regime_insights = self._detect_market_regimes(stock_data)
            insights['market_regimes'].extend(regime_insights)
            
            # 4. High Confidence Patterns
            high_confidence_patterns = self._enhanced_statistical_patterns(seasonal_stats, stock_data, 0.95)
            insights['high_confidence_patterns'].extend(high_confidence_patterns)
            
            # 5. Regime Changes
            regime_changes = self._detect_market_regimes(stock_data)
            insights['regime_changes'].extend(regime_changes)
            
            return insights
            
        except Exception as e:
            print(f"Error in comprehensive Prophet analysis: {str(e)}")
            return insights
    
    def _calculate_metrics(self, stock_data):
        """Calculate advanced metrics for analysis"""
        metrics = {}
        
        try:
            returns = stock_data['Returns'].dropna()
            
            # Basic metrics
            metrics['annual_return'] = returns.mean() * 252
            metrics['annual_volatility'] = returns.std() * np.sqrt(252)
            metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility'] if metrics['annual_volatility'] > 0 else 0
            
            # Risk metrics
            metrics['skewness'] = stats.skew(returns)
            metrics['kurtosis'] = stats.kurtosis(returns)
            metrics['max_drawdown'] = (returns.cumsum().cummax() - returns.cumsum()).max()
            
            # Time series properties
            if STATSMODELS_AVAILABLE:
                # Stationarity test
                adf_stat, adf_pvalue, _, _, _, _ = adfuller(returns.dropna())
                metrics['adf_statistic'] = adf_stat
                metrics['adf_pvalue'] = adf_pvalue
                metrics['is_stationary'] = adf_pvalue < 0.05
                
                # KPSS test
                kpss_stat, kpss_pvalue, _, _ = kpss(returns.dropna())
                metrics['kpss_statistic'] = kpss_stat
                metrics['kpss_pvalue'] = kpss_pvalue
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _advanced_time_series_analysis(self, stock_data):
        """Perform advanced time series decomposition analysis"""
        insights = {
            'cyclical_insights': [],
            'structural_breaks': [],
            'trend_analysis': {},
            'seasonality_strength': {}
        }
        
        try:
            if not STATSMODELS_AVAILABLE:
                return insights
            
            # Ensure returns column exists and is numeric
            if 'Returns' not in stock_data.columns:
                return insights
            
            # Convert to numeric and handle any non-numeric values
            returns = pd.to_numeric(stock_data['Returns'], errors='coerce').dropna()
            
            # Ensure we have sufficient data
            if len(returns) < 20:
                return insights
            
            # 1. Time Series Decomposition (using longer period for better seasonal detection)
            if len(returns) >= 104:  # Need at least 2 years of data
                try:
                    # Ensure we have a clean numeric series for decomposition
                    clean_returns = returns[pd.notnull(returns) & np.isfinite(returns)]
                    
                    if len(clean_returns) >= 104:
                        decomposition = seasonal_decompose(clean_returns, model='additive', period=52)  # Weekly seasonality
                        
                        # Extract components with additional validation
                        trend_component = decomposition.trend.dropna()
                        seasonal_component = decomposition.seasonal.dropna()
                        residual_component = decomposition.resid.dropna()
                        
                        # Trend analysis with numeric validation
                        if len(trend_component) > 0 and all(pd.notnull(trend_component) & np.isfinite(trend_component)):
                            trend_slope = np.polyfit(range(len(trend_component)), trend_component, 1)[0]
                            if pd.notnull(trend_slope) and np.isfinite(trend_slope):
                                insights['trend_analysis'] = {
                                    'slope': trend_slope,
                                    'direction': 'upward' if trend_slope > 0 else 'downward',
                                    'strength': abs(trend_slope),
                                    'description': f"Long-term trend is {('upward' if trend_slope > 0 else 'downward')} with slope {trend_slope:.6f}"
                                }
                        
                        # Seasonality strength with numeric validation
                        if len(seasonal_component) > 0 and all(pd.notnull(seasonal_component) & np.isfinite(seasonal_component)):
                            seasonal_variance = seasonal_component.var()
                            total_variance = clean_returns.var()
                            
                            if (pd.notnull(seasonal_variance) and np.isfinite(seasonal_variance) and 
                                pd.notnull(total_variance) and np.isfinite(total_variance) and total_variance > 0):
                                
                                seasonality_strength = seasonal_variance / total_variance
                                
                                if pd.notnull(seasonality_strength) and np.isfinite(seasonality_strength):
                                    insights['seasonality_strength'] = {
                                        'strength': seasonality_strength,
                                        'description': f"Seasonality explains {seasonality_strength:.1%} of total variance",
                                        'interpretation': 'Strong' if seasonality_strength > 0.1 else 'Moderate' if seasonality_strength > 0.05 else 'Weak'
                                    }
                        
                        # Cyclical patterns detection with validation
                        if len(seasonal_component) > 0 and all(pd.notnull(seasonal_component) & np.isfinite(seasonal_component)):
                            seasonal_std = seasonal_component.std()
                            if pd.notnull(seasonal_std) and np.isfinite(seasonal_std) and seasonal_std > 0:
                                # Find peaks in seasonal component
                                peaks, _ = find_peaks(seasonal_component, height=seasonal_std)
                                troughs, _ = find_peaks(-seasonal_component, height=seasonal_std)
                                
                                if len(peaks) > 0 or len(troughs) > 0:
                                    insights['cyclical_insights'].append({
                                        'type': 'seasonal_peaks_troughs',
                                        'peaks_count': len(peaks),
                                        'troughs_count': len(troughs),
                                        'description': f"Detected {len(peaks)} seasonal peaks and {len(troughs)} troughs in the data"
                                    })
                except Exception as decomp_error:
                    # Skip decomposition if it fails
                    pass
            
            # 2. Structural break detection using rolling statistics
            window = min(252, len(returns) // 4)  # 1 year or 1/4 of data
            if window >= 20:
                try:
                    # Ensure clean numeric data for rolling calculations
                    clean_returns = returns[pd.notnull(returns) & np.isfinite(returns)]
                    
                    if len(clean_returns) >= window:
                        rolling_mean = clean_returns.rolling(window).mean().dropna()
                        rolling_std = clean_returns.rolling(window).std().dropna()
                        
                        # Additional validation for rolling statistics
                        rolling_mean = rolling_mean[pd.notnull(rolling_mean) & np.isfinite(rolling_mean)]
                        rolling_std = rolling_std[pd.notnull(rolling_std) & np.isfinite(rolling_std)]
                        
                        if len(rolling_mean) > 0 and len(rolling_std) > 0:
                            # Detect significant changes in mean
                            mean_diff = rolling_mean.diff().dropna()
                            mean_diff = mean_diff[pd.notnull(mean_diff) & np.isfinite(mean_diff)]
                            
                            std_threshold = rolling_mean.std()
                            
                            # Only compare numeric values
                            if (not mean_diff.empty and 
                                pd.notnull(std_threshold) and 
                                np.isfinite(std_threshold) and 
                                std_threshold > 0):
                                mean_changes = abs(mean_diff) > 2 * std_threshold
                            else:
                                mean_changes = pd.Series([], dtype=bool)
                            
                            # Detect significant changes in volatility
                            vol_diff = rolling_std.diff().dropna()
                            vol_diff = vol_diff[pd.notnull(vol_diff) & np.isfinite(vol_diff)]
                            
                            vol_threshold = rolling_std.std()
                            
                            if (not vol_diff.empty and 
                                pd.notnull(vol_threshold) and 
                                np.isfinite(vol_threshold) and 
                                vol_threshold > 0):
                                vol_changes = abs(vol_diff) > 2 * vol_threshold
                            else:
                                vol_changes = pd.Series([], dtype=bool)
                            
                            structural_breaks = []
                            # Process mean changes
                            for i, (idx, change) in enumerate(mean_changes.items()):
                                if change and i < 5:  # Limit to 5 breaks
                                    try:
                                        structural_breaks.append({
                                            'date': idx,
                                            'type': 'mean_change',
                                            'description': f"Mean structural break detected on {idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)}"
                                        })
                                    except:
                                        continue
                            
                            # Process volatility changes  
                            for i, (idx, change) in enumerate(vol_changes.items()):
                                if change and len(structural_breaks) < 5:  # Limit total to 5
                                    try:
                                        structural_breaks.append({
                                            'date': idx,
                                            'type': 'volatility_change',
                                            'description': f"Volatility structural break detected on {idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)}"
                                        })
                                    except:
                                        continue
                            
                            insights['structural_breaks'] = structural_breaks
                except Exception as struct_error:
                    # Skip structural break detection if it fails
                    pass
            
            # 3. Autocorrelation analysis
            if STATSMODELS_AVAILABLE and len(returns) > 20:
                try:
                    # Ensure returns are all numeric
                    numeric_returns = returns[pd.notnull(returns) & np.isfinite(returns)]
                    
                    if len(numeric_returns) > 20:
                        # Ljung-Box test for autocorrelation
                        lbvalue, lbpvalue = acorr_ljungbox(numeric_returns, lags=[10], return_df=False)
                        
                        # Validate p-value is numeric before comparison
                        if (len(lbpvalue) > 0 and 
                            pd.notnull(lbpvalue[0]) and 
                            np.isfinite(lbpvalue[0]) and 
                            isinstance(lbpvalue[0], (int, float)) and 
                            lbpvalue[0] < 0.05):
                            
                            insights['cyclical_insights'].append({
                                'type': 'autocorrelation',
                                'ljung_box_pvalue': float(lbpvalue[0]),
                                'description': f"Significant autocorrelation detected (p-value: {lbpvalue[0]:.3f})"
                            })
                except Exception as autocorr_error:
                    # Skip autocorrelation analysis if it fails
                    pass
            
            return insights
            
        except Exception as e:
            print(f"Error in advanced time series analysis: {str(e)}")
            return insights
    
    def _enhanced_statistical_patterns(self, seasonal_stats, stock_data, confidence_threshold):
        """Enhanced statistical pattern detection"""
        patterns = []
        
        try:
            # 1. Statistical significance testing
            statistical_patterns = self._detect_statistical_patterns(seasonal_stats, confidence_threshold)
            patterns.extend(statistical_patterns)
            
            # 2. Machine learning pattern detection
            ml_patterns = self._detect_ml_patterns(stock_data, seasonal_stats, confidence_threshold)
            patterns.extend(ml_patterns)
            
            return patterns
            
        except Exception as e:
            print(f"Error in enhanced statistical pattern detection: {str(e)}")
            return []
    
    def _advanced_ml_analysis(self, stock_data, seasonal_stats, confidence_threshold):
        """Advanced machine learning analysis"""
        insights = {
            'patterns': [],
            'metrics': {}
        }
        
        try:
            # 1. Feature engineering
            features = self._engineer_features(stock_data)
            
            if features.empty:
                return insights
            
            # 2. Random Forest analysis
            if len(features) > 50:  # Minimum data requirement
                rf_patterns = self._random_forest_analysis(features, confidence_threshold)
                insights['patterns'].extend(rf_patterns)
            
            # 3. XGBoost analysis (if available)
            if XGBOOST_AVAILABLE and len(features) > 100:
                xgb_patterns = self._xgboost_analysis(features, confidence_threshold)
                insights['patterns'].extend(xgb_patterns)
            
            # 4. Metrics
            metrics = self._calculate_metrics(stock_data)
            insights['metrics'].update(metrics)
            
            return insights
            
        except Exception as e:
            print(f"Error in advanced machine learning analysis: {str(e)}")
            return None
    
    def _comprehensive_anomaly_detection(self, stock_data, seasonal_stats):
        """Comprehensive anomaly detection"""
        anomalies = []
        
        try:
            # 1. Statistical anomalies
            statistical_anomalies = self._detect_statistical_patterns(seasonal_stats, 0.95)
            anomalies.extend(statistical_anomalies)
            
            # 2. Machine learning anomalies
            ml_anomalies = self._detect_ml_patterns(stock_data, seasonal_stats, 0.95)
            anomalies.extend(ml_anomalies)
            
            # 3. Seasonal anomalies
            seasonal_anomalies = self._detect_anomalies(stock_data, seasonal_stats)
            anomalies.extend(seasonal_anomalies)
            
            return anomalies
            
        except Exception as e:
            print(f"Error in comprehensive anomaly detection: {str(e)}")
            return []
    
    def _advanced_seasonal_analysis(self, seasonal_stats, stock_data):
        """Advanced seasonal trend analysis"""
        trends = []
        
        try:
            # 1. Analyze seasonal trends
            trends.extend(self._analyze_seasonal_trends(seasonal_stats))
            
            # 2. Risk insights
            risk_insights = self._generate_risk_insights(seasonal_stats, stock_data)
            trends.extend(risk_insights)
            
            return trends
            
        except Exception as e:
            print(f"Error in advanced seasonal analysis: {str(e)}")
            return []
    
    def _comprehensive_risk_analysis(self, seasonal_stats, stock_data):
        """Comprehensive risk analysis"""
        insights = []
        
        try:
            # 1. Risk concentration analysis
            total_positive_months = seasonal_stats[seasonal_stats['Avg_Return'] > 0].shape[0]
            risk_concentration = (12 - total_positive_months) / 12
            
            if risk_concentration > 0.5:
                insights.append({
                    'type': 'high_risk_concentration',
                    'description': f"High risk concentration: {total_positive_months}/12 months are positive on average",
                    'recommendation': 'Consider diversifying across different assets or time periods',
                    'severity': 'high'
                })
            
            # 2. Drawdown risk analysis
            worst_month = seasonal_stats.loc[seasonal_stats['Min_Return'].idxmin()]
            if worst_month['Min_Return'] < -0.15:
                insights.append({
                    'type': 'severe_drawdown_risk',
                    'description': f"Severe drawdown risk in {worst_month.name}: worst month was {worst_month['Min_Return']:.3%}",
                    'recommendation': 'Implement stop-loss mechanisms during high-risk periods',
                    'severity': 'high'
                })
            
            # 3. Volatility clustering
            high_vol_count = seasonal_stats[seasonal_stats['Volatility'] > seasonal_stats['Volatility'].mean() * 1.5].shape[0]
            if high_vol_count >= 3:
                insights.append({
                    'type': 'volatility_clustering',
                    'description': f"{high_vol_count} months show significantly higher volatility",
                    'recommendation': 'Prepare for volatility clustering - adjust position sizing accordingly',
                    'severity': 'medium'
                })
            
            return insights
            
        except Exception as e:
            print(f"Error in comprehensive risk analysis: {str(e)}")
            return []
    
    def _advanced_strategy_generation(self, seasonal_stats, stock_data, insights):
        """Advanced trading strategy generation"""
        strategies = []
        
        try:
            # 1. Seasonal momentum strategy
            strong_months = seasonal_stats[(seasonal_stats['Avg_Return'] > 0.02) & (seasonal_stats['Win_Rate'] > 0.6)]
            if not strong_months.empty:
                strategy_months = list(strong_months.index)
                avg_return = strong_months['Avg_Return'].mean()
                avg_win_rate = strong_months['Win_Rate'].mean()
                
                strategies.append({
                    'name': 'Seasonal Momentum Strategy',
                    'type': 'momentum',
                    'description': f"Increase allocation during {', '.join(strategy_months)}",
                    'expected_return': f"{avg_return:.1%} per month",
                    'win_rate': f"{avg_win_rate:.0%}",
                    'implementation': f"Consider 1.5x normal position size during {', '.join(strategy_months)}"
                })
            
            # 2. Seasonal contrarian strategy
            weak_months = seasonal_stats[(seasonal_stats['Avg_Return'] < -0.01) & (seasonal_stats['Win_Rate'] < 0.4)]
            if not weak_months.empty:
                strategy_months = list(weak_months.index)
                
                strategies.append({
                    'name': 'Seasonal Contrarian Strategy',
                    'type': 'contrarian',
                    'description': f"Reduce exposure or hedge during {', '.join(strategy_months)}",
                    'implementation': f"Consider 0.5x normal position size or protective puts during {', '.join(strategy_months)}"
                })
            
            # 3. Mean reversion strategy
            volatile_months = seasonal_stats[seasonal_stats['Volatility'] > seasonal_stats['Volatility'].mean() * 1.3]
            if not volatile_months.empty:
                strategies.append({
                    'name': 'Volatility Timing Strategy',
                    'type': 'volatility',
                    'description': f"Capitalize on higher volatility during {', '.join(volatile_months.index)}",
                    'implementation': "Consider options strategies or shorter holding periods during high volatility months"
                })
            
            # 4. Calendar spread strategy
            best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
            worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
            
            if best_month['Avg_Return'] - worst_month['Avg_Return'] > 0.05:
                spread_value = best_month['Avg_Return'] - worst_month['Avg_Return']
                strategies.append({
                    'name': 'Calendar Spread Strategy',
                    'type': 'spread',
                    'description': f"Long {best_month.name} / Short {worst_month.name} seasonal spread",
                    'expected_spread': f"{spread_value:.1%}",
                    'implementation': f"Enter long positions in {best_month.name}, exit before {worst_month.name}"
                })
            
            return strategies
            
        except Exception as e:
            print(f"Error in advanced strategy generation: {str(e)}")
            return []
    
    def _detect_market_regimes(self, stock_data):
        """Detect market regimes using volatility clustering"""
        regimes = []
        
        try:
            returns = stock_data['Returns'].dropna()
            
            if len(returns) < 60:  # Need minimum data
                return regimes
            
            # 1. Volatility-based regime detection
            rolling_vol = returns.rolling(window=20).std()
            vol_threshold = rolling_vol.quantile(0.7)
            
            high_vol_periods = rolling_vol > vol_threshold
            regime_changes = high_vol_periods.diff().fillna(False)
            
            # 2. Identify regime change points
            for i, change in enumerate(regime_changes):
                if change and i > 0:
                    regime_type = 'high_volatility' if high_vol_periods.iloc[i] else 'low_volatility'
                    regimes.append({
                        'type': 'volatility_regime_change',
                        'regime': regime_type,
                        'date': returns.index[i],
                        'volatility': rolling_vol.iloc[i],
                        'description': f"Market entered {regime_type} regime on {returns.index[i].strftime('%Y-%m-%d')}"
                    })
            
            # 3. Return-based regime detection
            rolling_returns = returns.rolling(window=20).mean()
            return_threshold = 0.001  # 0.1% daily threshold
            
            bull_periods = rolling_returns > return_threshold
            bear_periods = rolling_returns < -return_threshold
            
            return_regime_changes = bull_periods.diff().fillna(False) | bear_periods.diff().fillna(False)
            
            for i, change in enumerate(return_regime_changes):
                if change and i > 0:
                    if bull_periods.iloc[i]:
                        regime_type = 'bullish_trend'
                    elif bear_periods.iloc[i]:
                        regime_type = 'bearish_trend'
                    else:
                        regime_type = 'neutral_trend'
                    
                    regimes.append({
                        'type': 'trend_regime_change',
                        'regime': regime_type,
                        'date': returns.index[i],
                        'avg_return': rolling_returns.iloc[i],
                        'description': f"Market entered {regime_type} on {returns.index[i].strftime('%Y-%m-%d')}"
                    })
            
            return regimes[:10]  # Limit to most recent 10 regime changes
            
        except Exception as e:
            print(f"Error in market regime detection: {str(e)}")
            return []
    
    def _analyze_correlations(self, stock_data):
        """Analyze internal correlations within the dataset"""
        insights = {}
        
        try:
            # 1. Select only numeric columns for correlation analysis
            numeric_columns = stock_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) < 2:
                return {'correlation_analysis': 'Insufficient numeric data for correlation analysis'}
            
            # 2. Calculate correlation matrix for numeric columns only
            correlation_data = stock_data[numeric_columns]
            correlation_matrix = correlation_data.corr()
            
            # 3. Analyze price-volume relationships
            if 'Close' in numeric_columns and 'Volume' in numeric_columns:
                price_volume_corr = correlation_matrix.loc['Close', 'Volume']
                insights['price_volume_correlation'] = {
                    'correlation': price_volume_corr,
                    'interpretation': 'Strong' if abs(price_volume_corr) > 0.7 else 'Moderate' if abs(price_volume_corr) > 0.3 else 'Weak',
                    'description': f"Price-volume correlation: {price_volume_corr:.3f}"
                }
            
            # 4. Analyze OHLC relationships
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            available_ohlc = [col for col in ohlc_cols if col in numeric_columns]
            
            if len(available_ohlc) >= 2:
                ohlc_correlations = {}
                for i in range(len(available_ohlc)):
                    for j in range(i+1, len(available_ohlc)):
                        col1, col2 = available_ohlc[i], available_ohlc[j]
                        corr_value = correlation_matrix.loc[col1, col2]
                        ohlc_correlations[f"{col1}-{col2}"] = corr_value
                
                insights['ohlc_correlations'] = ohlc_correlations
            
            # 5. Return correlation strength summary
            all_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        all_correlations.append(abs(corr_val))
            
            if all_correlations:
                avg_correlation = np.mean(all_correlations)
                max_correlation = np.max(all_correlations)
                
                insights['correlation_summary'] = {
                    'average_correlation': avg_correlation,
                    'maximum_correlation': max_correlation,
                    'interpretation': f"Average internal correlation: {avg_correlation:.3f}, Maximum: {max_correlation:.3f}"
                }
            
            return insights
            
        except Exception as e:
            print(f"Error in cross-asset correlation analysis: {str(e)}")
            return {}
    
    def _comprehensive_pattern_assessment(self, seasonal_stats, stock_data, insights):
        """Comprehensive pattern strength assessment"""
        try:
            # 1. Assess overall strength of seasonal patterns
            overall_strength = self._assess_pattern_strength(seasonal_stats)
            
            # 2. Interpret strength score
            interpretation = self._interpret_strength(overall_strength['overall_strength'])
            
            # 3. Generate insights
            insights['pattern_strength'] = {
                'overall_strength': overall_strength['overall_strength'],
                'consistency': overall_strength['consistency'],
                'win_rate_quality': overall_strength['win_rate_quality'],
                'reliability': overall_strength['reliability'],
                'return_magnitude': overall_strength['return_magnitude'],
                'interpretation': interpretation
            }
            
            return insights['pattern_strength']
            
        except Exception as e:
            print(f"Error in comprehensive pattern assessment: {str(e)}")
            return {}
    
    def _assess_forecast_accuracy(self, stock_data):
        """Assess forecast accuracy using simple statistical methods"""
        insights = {}
        
        try:
            returns = stock_data['Returns'].dropna()
            
            if len(returns) < 100:
                return {'forecast_accuracy': 'Insufficient data for accuracy assessment'}
            
            # 1. Simple moving average accuracy
            sma_20 = stock_data['Close'].rolling(20).mean()
            sma_50 = stock_data['Close'].rolling(50).mean()
            
            # Calculate directional accuracy for next day
            actual_direction = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)
            sma20_direction = (sma_20 > stock_data['Close']).astype(int)
            sma50_direction = (sma_50 > stock_data['Close']).astype(int)
            
            # Remove NaN values for comparison
            valid_data = ~(actual_direction.isna() | sma20_direction.isna() | sma50_direction.isna())
            
            if valid_data.sum() > 0:
                sma20_accuracy = (actual_direction[valid_data] == sma20_direction[valid_data]).mean()
                sma50_accuracy = (actual_direction[valid_data] == sma50_direction[valid_data]).mean()
                
                insights['moving_average_accuracy'] = {
                    'sma_20_directional_accuracy': sma20_accuracy,
                    'sma_50_directional_accuracy': sma50_accuracy,
                    'description': f"SMA20 directional accuracy: {sma20_accuracy:.1%}, SMA50: {sma50_accuracy:.1%}"
                }
            
            # 2. Volatility forecast accuracy (using rolling standard deviation)
            actual_volatility = returns.rolling(20).std().shift(-20)
            predicted_volatility = returns.rolling(20).std()
            
            vol_valid = ~(actual_volatility.isna() | predicted_volatility.isna())
            if vol_valid.sum() > 0:
                vol_mae = np.mean(np.abs(actual_volatility[vol_valid] - predicted_volatility[vol_valid]))
                vol_mape = np.mean(np.abs((actual_volatility[vol_valid] - predicted_volatility[vol_valid]) / actual_volatility[vol_valid])) * 100
                
                insights['volatility_forecast_accuracy'] = {
                    'volatility_mae': vol_mae,
                    'volatility_mape': vol_mape,
                    'description': f"Volatility forecast MAPE: {vol_mape:.1f}%"
                }
            
            # 3. Return persistence analysis
            return_autocorr = returns.autocorr(lag=1)
            if not np.isnan(return_autocorr):
                insights['return_predictability'] = {
                    'autocorrelation_lag1': return_autocorr,
                    'predictability': 'High' if abs(return_autocorr) > 0.1 else 'Moderate' if abs(return_autocorr) > 0.05 else 'Low',
                    'description': f"One-day return autocorrelation: {return_autocorr:.3f}"
                }
            
            # 4. Overall forecast quality assessment
            if len(insights) > 0:
                forecast_quality_score = 0
                
                if 'moving_average_accuracy' in insights:
                    ma_score = max(insights['moving_average_accuracy']['sma_20_directional_accuracy'],
                                 insights['moving_average_accuracy']['sma_50_directional_accuracy'])
                    forecast_quality_score += ma_score * 0.4
                
                if 'volatility_forecast_accuracy' in insights:
                    # Lower MAPE is better, convert to score
                    vol_score = max(0, 1 - insights['volatility_forecast_accuracy']['volatility_mape'] / 100)
                    forecast_quality_score += vol_score * 0.3
                
                if 'return_predictability' in insights:
                    pred_score = min(1, abs(insights['return_predictability']['autocorrelation_lag1']) * 10)
                    forecast_quality_score += pred_score * 0.3
                
                insights['overall_forecast_quality'] = {
                    'quality_score': forecast_quality_score,
                    'interpretation': 'High' if forecast_quality_score > 0.7 else 'Moderate' if forecast_quality_score > 0.4 else 'Low',
                    'description': f"Overall forecast quality score: {forecast_quality_score:.2f}"
                }
            
            return insights
            
        except Exception as e:
            print(f"Error in forecast accuracy assessment: {str(e)}")
            return {} 