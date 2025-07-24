import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    """Creates interactive visualizations for seasonal analysis"""
    
    def __init__(self, dark_theme=False):
        self.dark_theme = dark_theme
        self.light_palette = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4',   # Steel Blue
            'background': '#F8F9FA', # Light Gray
            'accent': '#FF6B35'     # Orange
        }
        self.dark_palette = {
            'positive': '#48BB78',  # Lighter green for dark mode
            'negative': '#F56565',  # Lighter red for dark mode
            'neutral': '#63B3ED',   # Lighter blue for dark mode
            'background': '#1A202C', # Dark gray
            'accent': '#ED8936'     # Orange
        }
        self.color_palette = self.dark_palette if dark_theme else self.light_palette
    
    def get_chart_layout(self, title="", height=400):
        """Get consistent chart layout based on theme"""
        if self.dark_theme:
            return {
                'title': title,
                'title_font_size': 18,
                'title_x': 0.5,
                'height': height,
                'font': dict(size=12, color='white'),
                'paper_bgcolor': '#1A202C',
                'plot_bgcolor': '#2D3748',
                'title_font_color': 'white'
            }
        else:
            return {
                'title': title,
                'title_font_size': 18,
                'title_x': 0.5,
                'height': height,
                'font': dict(size=12),
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white'
            }
    
    def update_axes_style(self, fig):
        """Update axes styling based on theme"""
        if self.dark_theme:
            fig.update_xaxes(gridcolor='#4A5568', color='white')
            fig.update_yaxes(gridcolor='#4A5568', color='white')
        else:
            fig.update_xaxes(gridcolor='lightgray')
            fig.update_yaxes(gridcolor='lightgray')
    
    def create_seasonal_heatmap(self, seasonal_stats, symbol):
        """Create an interactive heatmap of seasonal patterns"""
        try:
            # Prepare data for heatmap
            months = seasonal_stats.index.tolist()
            
            # Create a matrix for the heatmap
            heatmap_data = []
            metrics = ['Avg_Return', 'Win_Rate']
            
            for metric in metrics:
                row = seasonal_stats[metric].tolist()
                heatmap_data.append(row)
            
            # Create the heatmap
            text_color = 'white' if self.dark_theme else 'black'
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=months,
                y=['Average Return (%)', 'Win Rate (%)'],
                colorscale='RdYlGn',
                text=[[f"{val:.1f}%" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 12, "color": text_color},
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                             'Month: %{x}<br>' +
                             'Value: %{text}<br>' +
                             '<extra></extra>'
            ))
            
            layout = self.get_chart_layout(f'📅 Seasonal Heatmap for {symbol}', 300)
            layout.update({
                'xaxis_title': "Months",
                'yaxis_title': "Metrics"
            })
            fig.update_layout(**layout)
            self.update_axes_style(fig)
            
            return fig
            
        except Exception as e:
            print(f"Error creating seasonal heatmap: {str(e)}")
            return go.Figure()
    
    def create_monthly_returns_chart(self, seasonal_stats, symbol, height=400):
        """Create bar chart of monthly returns"""
        try:
            months = seasonal_stats.index.tolist()
            returns = seasonal_stats['Avg_Return'].tolist()
            
            # Color bars based on positive/negative returns
            colors = [self.color_palette['positive'] if r > 0 else self.color_palette['negative'] for r in returns]
            
            fig = go.Figure()
            
            # Add return bars (convert to percentage for display)
            returns_pct = [r * 100 for r in returns]  # Convert to percentage
            fig.add_trace(go.Bar(
                x=months,
                y=returns_pct,
                name='Average Return (%)',
                marker_color=colors,
                text=[f"{r:.1f}%" for r in returns_pct],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Average Return: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ))
            
            layout = self.get_chart_layout(f'📊 Monthly Average Returns for {symbol}', height)
            layout.update({
                'xaxis_title': "Months",
                'yaxis_title': "Average Return (%)",
                'showlegend': False
            })
            fig.update_layout(**layout)
            self.update_axes_style(fig)
            
            # Add zero line
            line_color = "#4A5568" if self.dark_theme else "gray"
            fig.add_hline(y=0, line_dash="dash", line_color=line_color, opacity=0.5)
            
            return fig
            
        except Exception as e:
            print(f"Error creating monthly returns chart: {str(e)}")
            return go.Figure()
    
    def create_win_rate_chart(self, seasonal_stats, symbol):
        """Create win rate visualization"""
        try:
            months = seasonal_stats.index.tolist()
            win_rates = seasonal_stats['Win_Rate'].tolist()
            
            # Color based on win rate thresholds (convert to percentage for comparison)
            colors = []
            for wr in win_rates:
                wr_pct = wr * 100  # Convert to percentage for threshold comparison
                if wr_pct >= 70:
                    colors.append(self.color_palette['positive'])
                elif wr_pct <= 30:
                    colors.append(self.color_palette['negative'])
                else:
                    colors.append(self.color_palette['neutral'])
            
            fig = go.Figure()
            
            # Convert win rates to percentage for display
            win_rates_pct = [wr * 100 for wr in win_rates]
            fig.add_trace(go.Bar(
                x=months,
                y=win_rates_pct,
                name='Win Rate (%)',
                marker_color=colors,
                text=[f"{wr:.0f}%" for wr in win_rates_pct],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Win Rate: %{y:.0f}%<br>' +
                             '<extra></extra>'
            ))
            
            layout = self.get_chart_layout(f'🎯 Monthly Win Rates for {symbol}', 400)
            layout.update({
                'xaxis_title': "Months",
                'yaxis_title': "Win Rate (%)",
                'showlegend': False
            })
            fig.update_layout(**layout)
            self.update_axes_style(fig)
            
            # Add reference lines with theme-appropriate colors
            line_color = "#4A5568" if self.dark_theme else "gray"
            pos_color = "#48BB78" if self.dark_theme else "green"
            neg_color = "#F56565" if self.dark_theme else "red"
            
            fig.add_hline(y=50, line_dash="dash", line_color=line_color, opacity=0.5, 
                         annotation_text="50% (Random)")
            fig.add_hline(y=70, line_dash="dot", line_color=pos_color, opacity=0.7,
                         annotation_text="70% (Strong)")
            fig.add_hline(y=30, line_dash="dot", line_color=neg_color, opacity=0.7,
                         annotation_text="30% (Weak)")
            
            return fig
            
        except Exception as e:
            print(f"Error creating win rate chart: {str(e)}")
            return go.Figure()
    
    def create_price_chart(self, stock_data, symbol):
        """Create interactive price chart with seasonal highlights"""
        try:
            subplot_titles = [f'{symbol} Price Chart', 'Daily Returns']
            if self.dark_theme:
                subplot_title_color = 'white'
            else:
                subplot_title_color = 'black'
                
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=self.color_palette['neutral'], width=2),
                    hovertemplate='Date: %{x}<br>' +
                                 'Price: $%{y:.2f}<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if len(stock_data) > 50:
                ma50 = stock_data['Close'].rolling(50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=ma50,
                        mode='lines',
                        name='50-day MA',
                        line=dict(color=self.color_palette['accent'], width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Returns chart
            colors = [self.color_palette['positive'] if r > 0 else self.color_palette['negative'] 
                     for r in stock_data['Returns']]
            
            fig.add_trace(
                go.Bar(
                    x=stock_data.index,
                    y=stock_data['Returns'] * 100,
                    name='Daily Returns (%)',
                    marker_color=colors,
                    opacity=0.7,
                    hovertemplate='Date: %{x}<br>' +
                                 'Return: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
            
            layout = self.get_chart_layout(f'📈 {symbol} Price Analysis', 600)
            layout['showlegend'] = True
            fig.update_layout(**layout)
            self.update_axes_style(fig)
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
            
            # Update subplot title colors
            for annotation in fig['layout']['annotations']:
                annotation['font'] = dict(color=subplot_title_color, size=14)
            
            return fig
            
        except Exception as e:
            print(f"Error creating price chart: {str(e)}")
            return go.Figure()
    
    def create_weekday_returns_chart(self, weekday_stats, symbol, height=400):
        """Create bar chart of weekday returns"""
        try:
            weekdays = weekday_stats.index.tolist()
            returns = weekday_stats['Avg_Return'].tolist()
            
            # Color bars based on positive/negative returns
            colors = [self.color_palette['positive'] if r > 0 else self.color_palette['negative'] for r in returns]
            
            fig = go.Figure()
            
            # Add return bars (convert to percentage for display)
            returns_pct = [r * 100 for r in returns]  # Convert to percentage
            fig.add_trace(go.Bar(
                x=weekdays,
                y=returns_pct,
                name='Average Return (%)',
                marker_color=colors,
                text=[f"{r:.3f}%" for r in returns_pct],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Average Return: %{y:.3f}%<br>' +
                             '<extra></extra>'
            ))
            
            layout = self.get_chart_layout(f'📅 Weekday Returns for {symbol}', height)
            layout.update({
                'xaxis_title': "Day of Week",
                'yaxis_title': "Average Return (%)",
                'showlegend': False
            })
            fig.update_layout(**layout)
            self.update_axes_style(fig)
            
            # Add zero line
            line_color = "#4A5568" if self.dark_theme else "gray"
            fig.add_hline(y=0, line_dash="dash", line_color=line_color, opacity=0.5)
            
            return fig
            
        except Exception as e:
            print(f"Error creating weekday returns chart: {str(e)}")
            return go.Figure()

    def create_combined_seasonality_chart(self, seasonal_stats, weekday_stats, symbol):
        """Create a combined chart showing both monthly and weekday patterns"""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f'Monthly Patterns', 'Weekday Patterns'],
                horizontal_spacing=0.1
            )
            
            # Monthly data
            months = seasonal_stats.index.tolist()
            monthly_returns = seasonal_stats['Avg_Return'].tolist()
            monthly_returns_pct = [r * 100 for r in monthly_returns]  # Convert to percentage
            monthly_colors = [self.color_palette['positive'] if r > 0 else self.color_palette['negative'] for r in monthly_returns]
            
            # Weekday data
            weekdays = weekday_stats.index.tolist()
            weekday_returns = weekday_stats['Avg_Return'].tolist()
            weekday_returns_pct = [r * 100 for r in weekday_returns]  # Convert to percentage
            weekday_colors = [self.color_palette['positive'] if r > 0 else self.color_palette['negative'] for r in weekday_returns]
            
            # Add monthly bars
            fig.add_trace(
                go.Bar(
                    x=months,
                    y=monthly_returns_pct,
                    name='Monthly Returns',
                    marker_color=monthly_colors,
                    text=[f"{r:.1f}%" for r in monthly_returns_pct],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add weekday bars
            fig.add_trace(
                go.Bar(
                    x=weekdays,
                    y=weekday_returns_pct,
                    name='Weekday Returns',
                    marker_color=weekday_colors,
                    text=[f"{r:.3f}%" for r in weekday_returns_pct],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Return: %{y:.3f}%<extra></extra>'
                ),
                row=1, col=2
            )
            
            layout = self.get_chart_layout(f'📊 Combined Seasonality Analysis for {symbol}', 500)
            layout['showlegend'] = False
            fig.update_layout(**layout)
            self.update_axes_style(fig)
            
            # Update axis titles
            fig.update_xaxes(title_text="Months", row=1, col=1)
            fig.update_xaxes(title_text="Weekdays", row=1, col=2)
            fig.update_yaxes(title_text="Average Return (%)", row=1, col=1)
            fig.update_yaxes(title_text="Average Return (%)", row=1, col=2)
            
            # Add zero lines
            line_color = "#4A5568" if self.dark_theme else "gray"
            fig.add_hline(y=0, line_dash="dash", line_color=line_color, opacity=0.5, row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color=line_color, opacity=0.5, row=1, col=2)
            
            # Update subplot title colors
            if self.dark_theme:
                for annotation in fig['layout']['annotations']:
                    annotation['font'] = dict(color='white', size=14)
            
            return fig
            
        except Exception as e:
            print(f"Error creating combined seasonality chart: {str(e)}")
            return go.Figure() 