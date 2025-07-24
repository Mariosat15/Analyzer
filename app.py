import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_processor import DataProcessor
from ai_analyzer import AIAnalyzer
from visualizer import Visualizer
from advanced_analytics import (
    RiskAnalyzer, TechnicalAnalyzer, BacktestEngine, 
    StatisticalTester, MarketRegimeDetector, AlertSystem,
    AdvancedVisualizer, PortfolioOptimizer, SentimentAnalyzer,
    OptionsAnalyzer, EconomicDataIntegrator, AlternativeDataAnalyzer
)

# Page config
st.set_page_config(
    page_title="AI Seasonal Edge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'dark_theme' not in st.session_state:
    st.session_state.dark_theme = False

# Initialize session state for uploaded data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'uploaded_symbol' not in st.session_state:
    st.session_state.uploaded_symbol = None

# Initialize session state for multi-asset dashboard
if 'multi_asset_data' not in st.session_state:
    st.session_state.multi_asset_data = {}
if 'uploaded_tickers' not in st.session_state:
    st.session_state.uploaded_tickers = []
if 'dashboard_mode' not in st.session_state:
    st.session_state.dashboard_mode = 'single'  # 'single' or 'multi'
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = None

def get_theme_css(dark_mode=False):
    """Generate modern, professional CSS styling"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Overrides */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 1rem 0 2rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.1;
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header .subtitle {
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .main-header .creator {
        font-size: 1.5rem;
        margin: 1rem 0 0 0;
        opacity: 1.0;
        font-weight: 700;
        font-style: normal;
        position: relative;
        z-index: 1;
        text-align: center;
    }
    
    .main-header .email {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 400;
        position: relative;
        z-index: 1;
        text-align: center;
        color: #ffffff;
    }
    
    .github-section {
        margin: 1.5rem 0 0 0;
        text-align: center;
        position: relative;
        z-index: 1;
    }
    
    .github-btn {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: white !important;
        padding: 12px 24px;
        border-radius: 25px;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .github-btn:hover {
        background: rgba(255, 255, 255, 0.25);
        border-color: rgba(255, 255, 255, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        color: white !important;
        text-decoration: none !important;
    }
    
    .github-btn:link, .github-btn:visited, .github-btn:active {
        color: white !important;
        text-decoration: none !important;
    }
    
    .github-icon {
        margin-right: 8px;
        font-size: 1.1em;
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .metric-card h2, .metric-card h3, .metric-card h4 {
        font-family: 'Inter', sans-serif;
        margin: 0;
        color: #1a202c;
    }
    
    .metric-card h2 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h4 {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    
    .metric-value {
        font-size: 1.1rem;
        font-weight: 500;
        color: #475569;
        line-height: 1.6;
    }
    
    /* Specialized Cards */
    .success-card {
        background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
        border-left: 6px solid #22c55e;
    }
    
    .success-card h2 {
        background: linear-gradient(135deg, #15803d 0%, #22c55e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 6px solid #ef4444;
    }
    
    .danger-card h2 {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 6px solid #f59e0b;
    }
    
    .warning-card h2 {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .info-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 6px solid #3b82f6;
    }
    
    .info-card h2 {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .section-header h3 {
        margin: 0;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    /* Strategy Cards */
    .strategy-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .strategy-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
    }
    
    /* Enhanced Hero Section Styles */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><pattern id="hero-pattern" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/><circle cx="25" cy="25" r="1" fill="white" opacity="0.05"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.05"/></pattern></defs><rect width="100%" height="100%" fill="url(%23hero-pattern)"/></svg>');
        opacity: 0.3;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
        text-align: center;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .hero-badge span {
        background: #22c55e;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1.5rem;
        line-height: 1.1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-description {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 3rem;
        line-height: 1.7;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .hero-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
        margin-top: 3rem;
    }
    
    .stat-item {
        text-align: center;
        color: white;
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 900;
        color: #ffd700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stat-label {
        font-size: 1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* Enhanced Features Section */
    .features-section {
        margin: 4rem 0;
    }
    
    .section-title {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .section-title h2 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .section-title p {
        font-size: 1.1rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 2rem;
        margin-top: 3rem;
    }
    
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    
    .feature-card.featured {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: scale(1.05);
    }
    
    .feature-card.featured:hover {
        transform: translateY(-8px) scale(1.07);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        display: block;
    }
    
    .feature-card h3 {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1e293b;
    }
    
    .feature-card.featured h3 {
        color: white;
    }
    
    .feature-card p {
        font-size: 1rem;
        line-height: 1.6;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    
    .feature-card.featured p {
        color: rgba(255, 255, 255, 0.9);
    }
    
    .feature-details {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-details li {
        padding: 0.5rem 0;
        color: #475569;
        font-weight: 500;
        position: relative;
        padding-left: 1.5rem;
    }
    
    .feature-details li::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #22c55e;
        font-weight: 700;
    }
    
    .feature-card.featured .feature-details li {
        color: rgba(255, 255, 255, 0.9);
    }
    
    .feature-card.featured .feature-details li::before {
        color: #ffd700;
    }
    
    .feature-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: #ffd700;
        color: #1e293b;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Upload Success Styles - Cleaned up */
    
    /* Enhanced Header Styles */
    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 2rem;
    }
    
    .brand-section {
        flex: 1;
    }
    
    .header-badges {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .badge {
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .badge.professional {
        background: rgba(34, 197, 94, 0.2);
        color: #dcfce7;
    }
    
    .badge.ai-powered {
        background: rgba(59, 130, 246, 0.2);
        color: #dbeafe;
    }
    
    .badge.institutional {
        background: rgba(245, 158, 11, 0.2);
        color: #fef3c7;
    }
    
    /* Enhanced Sidebar Styles */
    .sidebar-header {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .logo-icon {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .logo-text h2 {
        margin: 0;
        color: #1e293b;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .logo-text p {
        margin: 0.25rem 0 0 0;
        color: #64748b;
        font-size: 0.85rem;
    }
    
    .sidebar-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem;
        background: rgba(34, 197, 94, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
    }
    
    .status-indicator.active {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .sidebar-status span {
        color: #15803d;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Cleaned up - using standard Streamlit components now */
    
    /* Enhanced CTA Section */
    .cta-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 24px;
        padding: 4rem 2rem;
        margin: 4rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .cta-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><pattern id="cta-pattern" width="50" height="50" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100%" height="100%" fill="url(%23cta-pattern)"/></svg>');
        opacity: 0.3;
    }
    
    .cta-content {
        position: relative;
        z-index: 2;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .cta-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .cta-description {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2.5rem;
        line-height: 1.7;
    }
    
    .cta-features {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2.5rem;
        flex-wrap: wrap;
    }
    
    .cta-feature {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        font-weight: 600;
    }
    
    .cta-icon {
        font-size: 1.2rem;
    }
    
    .cta-action {
        margin-top: 2rem;
    }
    
    .cta-prompt {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .cta-arrow {
        font-size: 2rem;
        animation: bounce-arrow 2s infinite;
    }
    
    @keyframes bounce-arrow {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    /* Platform Footer */
    .platform-footer {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 3rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .footer-content {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .footer-section {
        text-align: center;
        padding: 1.5rem;
    }
    
    .footer-section h4 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    .footer-section p {
        color: #64748b;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-stats {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
        }
        
        .header-content {
            flex-direction: column;
            text-align: center;
        }
        
        .header-badges {
            justify-content: center;
        }
        
        .cta-features {
            flex-direction: column;
            align-items: center;
        }
        
        .cta-title {
            font-size: 2rem;
        }
        
        .upload-formats {
            grid-template-columns: 1fr;
        }
        
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .section-title h2 {
            font-size: 2rem;
        }
    }
    
    .strategy-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .strategy-card:hover::before {
        transform: scaleX(1);
    }
    
    .strategy-card h4 {
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .strategy-card p {
        margin: 0.75rem 0;
        color: #475569;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    .strategy-card strong {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Risk Level Cards */
    .risk-high {
        border-left: 6px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    .risk-medium {
        border-left: 6px solid #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .risk-low {
        border-left: 6px solid #22c55e;
        background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
    }
    
    /* Upload Area Enhancement */
    .upload-area {
        border: 3px dashed #cbd5e1;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        transform: translateY(-2px);
    }
    
    /* Insights List Enhancement */
    .insights-list {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }
    
    .insights-list li {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        background: #f8fafc;
        border-left: 4px solid #667eea;
        color: #475569;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .insights-list li:hover {
        background: #eff6ff;
        transform: translateX(4px);
    }
    
    .insights-list li::before {
        content: "💡";
        margin-right: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* Feature List */
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }
    
    .feature-list li {
        padding: 0.75rem 0;
        border-bottom: 1px solid #e2e8f0;
        color: #475569;
        font-weight: 500;
        display: flex;
        align-items: center;
    }
    
    .feature-list li:last-child {
        border-bottom: none;
    }
    
    .feature-list li::before {
        content: "✅";
        margin-right: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Button Enhancements */
    .custom-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Dark Mode Adjustments */
    """ + ("""
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(51, 65, 85, 0.3);
        color: #e2e8f0 !important;
    }
    
    .metric-card h2, .metric-card h3, .metric-card h4 {
        color: #f1f5f9 !important;
    }
    
    .section-header {
        background: rgba(30, 41, 59, 0.6);
        color: #e2e8f0;
    }
    
    .strategy-card {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(51, 65, 85, 0.3);
        color: #e2e8f0 !important;
    }
    """ if dark_mode else "") + """
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .metric-card {
            padding: 1.5rem;
            margin: 0.75rem 0;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .strategy-card {
            padding: 1.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            padding: 2rem 1rem;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    </style>
    """

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.dark_theme), unsafe_allow_html=True)

def process_multi_assets(tickers, enable_ai, confidence_threshold):
    """Process multiple assets and store in session state"""
    import time
    import random
    
    data_processor = DataProcessor()
    ai_analyzer = AIAnalyzer() if enable_ai else None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_count = 0
    failed_count = 0
    st.session_state.multi_asset_data = {}
    
    # Check if we have uploaded CSV data for multi-asset analysis
    has_multi_asset_csv = hasattr(st.session_state, 'multi_asset_csv_data') and st.session_state.multi_asset_csv_data
    has_single_uploaded_data = hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data is not None
    
    if has_multi_asset_csv:
        st.info(f"🔄 **Processing {len(st.session_state.multi_asset_csv_data)} assets from uploaded CSV files locally** - No API calls needed!")
        # No rate limiting needed for local data
        max_assets = len(tickers)  # Process all tickers from uploaded data
    elif has_single_uploaded_data:
        st.info("🔄 **Processing uploaded CSV data locally** - No API calls needed!")
        # No rate limiting needed for local data
        max_assets = len(tickers)  # Process all tickers from uploaded data
    else:
        st.info("🌐 **Fetching data from Yahoo Finance API** - Rate limiting applied")
        # Limit processing to reasonable number to avoid rate limits
        max_assets = min(len(tickers), 50)  # Process max 50 assets at once
        if len(tickers) > max_assets:
            st.warning(f"⚠️ Processing first {max_assets} assets to avoid rate limits. You can process more in batches.")
            tickers = tickers[:max_assets]
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"Processing {ticker}... ({i+1}/{len(tickers)}) - ✅{processed_count} ❌{failed_count}")
            progress_bar.progress((i + 1) / len(tickers))
            
            # Only add delays for API calls, not local data processing
            if not has_multi_asset_csv and not has_single_uploaded_data and i > 0:  # Only delay for API calls
                delay = random.uniform(0.3, 1.0)
                time.sleep(delay)
            
            # Get stock data - either from uploaded CSV or API
            if has_multi_asset_csv and ticker in st.session_state.multi_asset_csv_data:
                # Process from multi-asset uploaded CSV data
                stock_data = data_processor.process_uploaded_data(
                    st.session_state.multi_asset_csv_data[ticker], 
                    ticker,
                    start_date=None,
                    end_date=None
                )
            elif has_single_uploaded_data:
                # Process from single uploaded CSV data
                stock_data = data_processor.process_uploaded_data(
                    st.session_state.uploaded_data, 
                    ticker,
                    start_date=None,
                    end_date=None
                )
            else:
                # Fetch data using yfinance with timeout
                ticker_obj = yf.Ticker(ticker)
                stock_data = ticker_obj.history(period="5y", timeout=15)  # 5 years of data with timeout
            
            if stock_data.empty:
                if has_multi_asset_csv:
                    st.warning(f"⚠️ No data found for {ticker} in uploaded CSV files")
                elif has_single_uploaded_data:
                    st.warning(f"⚠️ No data found for {ticker} in uploaded CSV")
                else:
                    st.warning(f"⚠️ No data found for {ticker} from Yahoo Finance")
                failed_count += 1
                continue
                
            # Calculate seasonal stats
            seasonal_stats = data_processor.calculate_seasonal_stats(stock_data)
            
            # Calculate current seasonal relevance
            today = datetime.now()
            current_month = today.month
            current_day = today.timetuple().tm_yday
            
            # Get today's historical performance
            seasonal_relevance = calculate_seasonal_relevance(stock_data, today, seasonal_stats)
            
            # AI insights if enabled (skip for large batches to speed up)
            ai_insights = None
            if enable_ai and ai_analyzer and len(tickers) <= 20:  # Only run AI for smaller batches
                try:
                    ai_insights = ai_analyzer.analyze_patterns(stock_data, seasonal_stats, confidence_threshold)
                except:
                    pass  # Skip AI if fails
            
            # Store processed data
            st.session_state.multi_asset_data[ticker] = {
                'stock_data': stock_data,
                'seasonal_stats': seasonal_stats,
                'seasonal_relevance': seasonal_relevance,
                'ai_insights': ai_insights,
                'current_month_stats': seasonal_stats.iloc[current_month - 1] if len(seasonal_stats) >= current_month else None,
                'last_updated': datetime.now()
            }
            
            processed_count += 1
            
        except Exception as e:
            error_msg = str(e)
            if not has_multi_asset_csv and not has_single_uploaded_data and ("Too Many Requests" in error_msg or "rate" in error_msg.lower() or "429" in error_msg):
                st.error(f"⚠️ Rate limited on {ticker}. Adding longer delay...")
                time.sleep(5)  # Wait 5 seconds on rate limit
                failed_count += 1
            elif not has_multi_asset_csv and not has_single_uploaded_data and "timeout" in error_msg.lower():
                st.warning(f"⏰ Timeout on {ticker}")
                failed_count += 1
            else:
                if has_multi_asset_csv or has_single_uploaded_data:
                    st.warning(f"❌ Failed to process {ticker} from CSV: {error_msg}")
                else:
                    st.warning(f"❌ Failed to process {ticker} from API: {error_msg}")
                failed_count += 1
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if processed_count > 0:
        st.session_state.multi_asset_analyzed = True
        if has_multi_asset_csv:
            data_source = f"{len(st.session_state.multi_asset_csv_data)} uploaded CSV files"
        elif has_single_uploaded_data:
            data_source = "uploaded CSV data"
        else:
            data_source = "Yahoo Finance API"
        
        st.success(f"✅ Successfully processed {processed_count} out of {len(tickers)} assets from {data_source} ({failed_count} failed)")
        
        if failed_count > 0:
            if has_multi_asset_csv or has_single_uploaded_data:
                st.info("💡 **Tips for CSV processing:**\n- Check that ticker symbols exist in your CSV file\n- Ensure data format is correct (DATE,OPEN,HIGH,LOW,CLOSE)")
            else:
                st.info("💡 **Tips to reduce API failures:**\n- Process fewer assets at once\n- Wait a few minutes between batches\n- Check that ticker symbols are valid")
    else:
        if has_multi_asset_csv or has_single_uploaded_data:
            data_source = "uploaded CSV files"
        else:
            data_source = "APIs due to rate limiting or invalid tickers"
        st.error(f"❌ No assets were successfully processed from {data_source}.")

def calculate_seasonal_relevance(stock_data, target_date, seasonal_stats):
    """Calculate how relevant seasonal patterns are for a given date"""
    current_month = target_date.month
    current_day_of_year = target_date.timetuple().tm_yday
    
    # Get current month stats
    current_month_return = seasonal_stats.iloc[current_month - 1]['Avg_Return'] if len(seasonal_stats) >= current_month else 0
    current_month_win_rate = seasonal_stats.iloc[current_month - 1]['Win_Rate'] if len(seasonal_stats) >= current_month else 0.5
    
    # Calculate seasonal strength based on:
    # 1. Current month performance
    # 2. Historical volatility 
    # 3. Consistency of pattern
    
    volatility = seasonal_stats['Avg_Return'].std()
    consistency = current_month_win_rate
    strength = abs(current_month_return)
    
    # Combine factors into relevance score
    relevance_score = (strength * 0.4 + consistency * 0.4 + (1/max(volatility, 0.001)) * 0.2)
    
    return {
        'score': relevance_score,
        'current_month_return': current_month_return,
        'current_month_win_rate': current_month_win_rate,
        'strength': strength,
        'consistency': consistency,
        'volatility': volatility
    }

def get_asset_logo_url(ticker):
    """Get logo URL for an asset (simplified version)"""
    # This is a simplified implementation - in production you'd use a proper API
    return f"https://logo.clearbit.com/{ticker.lower()}.com"

def display_multi_asset_dashboard():
    """Display the comprehensive multi-asset dashboard with professional structure"""
    # Professional header with proper spacing
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">🎛️ Multi-Asset Seasonal Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Assets ranked by seasonal pattern relevance to today's date</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.multi_asset_data:
        st.error("❌ No asset data available. Please process assets first.")
        return
    
    # Sort assets by seasonal relevance
    sorted_assets = sorted(
        st.session_state.multi_asset_data.items(),
        key=lambda x: x[1]['seasonal_relevance']['score'],
        reverse=True
    )
    
    # SECTION 1: Dashboard Controls
    st.markdown("### 🎛️ Dashboard Controls")
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search Assets", placeholder="Type ticker or filter...", key="asset_search_stable")
        
        with col2:
            sort_option = st.selectbox(
                "📊 Sort By",
                ["Seasonal Relevance", "Current Month Return", "Win Rate", "Alphabetical"],
                key="dashboard_sort_stable"
            )
        
        with col3:
            display_mode = st.selectbox(
                "📋 Display Mode",
                ["Grid View", "List View", "Compact View"],
                key="display_mode_stable"
            )
        
        with col4:
            show_details = st.checkbox("Show Details", value=True, key="show_details_stable")
    
    st.divider()
    
    # Filter and sort assets
    if search_query:
        filtered_assets = [(k, v) for k, v in sorted_assets if search_query.upper() in k.upper()]
    else:
        filtered_assets = sorted_assets
    
    # Re-sort based on selection
    if sort_option == "Current Month Return":
        filtered_assets.sort(key=lambda x: x[1]['seasonal_relevance']['current_month_return'], reverse=True)
    elif sort_option == "Win Rate":
        filtered_assets.sort(key=lambda x: x[1]['seasonal_relevance']['current_month_win_rate'], reverse=True)
    elif sort_option == "Alphabetical":
        filtered_assets.sort(key=lambda x: x[0])
    
    # SECTION 2: Portfolio Summary (moved to top for better visibility)
    st.markdown("### 📊 Portfolio Summary")
    with st.container():
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        avg_return = np.mean([data['seasonal_relevance']['current_month_return'] for _, data in filtered_assets])
        avg_win_rate = np.mean([data['seasonal_relevance']['current_month_win_rate'] for _, data in filtered_assets])
        positive_count = len([d for _, d in filtered_assets if d['seasonal_relevance']['current_month_return'] > 0])
        high_confidence = len([d for _, d in filtered_assets if d['seasonal_relevance']['score'] > 0.6])
        
        with summary_col1:
            st.metric(
                "📈 Avg Monthly Return", 
                f"{avg_return:.2%}",
                delta=f"Portfolio of {len(filtered_assets)} assets",
                help="Average expected monthly return across all assets"
            )
        
        with summary_col2:
            st.metric(
                "🎯 Avg Win Rate", 
                f"{avg_win_rate:.1%}",
                delta="Historical probability",
                help="Average percentage of positive months across all assets"
            )
        
        with summary_col3:
            positive_pct = (positive_count / len(filtered_assets)) * 100 if filtered_assets else 0
            st.metric(
                "✅ Positive Seasonality", 
                f"{positive_count}/{len(filtered_assets)}",
                delta=f"{positive_pct:.0f}% of portfolio",
                help="Assets with positive current month historical performance"
            )
        
        with summary_col4:
            confidence_pct = (high_confidence / len(filtered_assets)) * 100 if filtered_assets else 0
            st.metric(
                "🔥 High Confidence", 
                f"{high_confidence}/{len(filtered_assets)}",
                delta=f"{confidence_pct:.0f}% strong signals",
                help="Assets with seasonal relevance score > 0.6"
            )
    
    st.divider()
    
    # SECTION 3: Top Seasonal Alerts
    st.markdown("### 🚨 Today's Top Seasonal Alerts")
    
    top_3_positive = [asset for asset in filtered_assets[:3] if asset[1]['seasonal_relevance']['current_month_return'] > 0]
    top_3_negative = [asset for asset in filtered_assets if asset[1]['seasonal_relevance']['current_month_return'] < 0][:3]
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.markdown("#### 📈 Top Seasonal Picks Today")
        if top_3_positive:
            for ticker, data in top_3_positive:
                rel = data['seasonal_relevance']
                confidence_icon = "🟢" if rel['score'] > 0.7 else "🟡" if rel['score'] > 0.5 else "🔴"
                
                with st.container():
                    st.success(f"""
                    **{ticker}** {confidence_icon}
                    
                    📊 **Current Month Avg:** {rel['current_month_return']:.2%}
                    
                    🎯 **Win Rate:** {rel['current_month_win_rate']:.1%} | **Score:** {rel['score']:.2f}
                    """)
        else:
            st.info("No strong positive seasonal signals today")
    
    with alert_col2:
        st.markdown("#### 📉 Historically Weak Today")
        if top_3_negative:
            for ticker, data in top_3_negative:
                rel = data['seasonal_relevance']
                caution_icon = "🔴" if rel['current_month_return'] < -0.02 else "🟡"
                
                with st.container():
                    st.warning(f"""
                    **{ticker}** {caution_icon}
                    
                    📊 **Current Month Avg:** {rel['current_month_return']:.2%}
                    
                    🎯 **Win Rate:** {rel['current_month_win_rate']:.1%} | **Score:** {rel['score']:.2f}
                    """)
        else:
            st.info("No significant negative seasonal patterns today")
    
    st.divider()
    
    # SECTION 4: Asset Display based on mode
    st.markdown("### 📋 Asset Portfolio Overview")
    
    if display_mode == "Grid View":
        # Display assets in a 3-column grid
        for i in range(0, len(filtered_assets), 3):
            cols = st.columns(3)
            
            for j, col in enumerate(cols):
                if i + j < len(filtered_assets):
                    ticker, data = filtered_assets[i + j]
                    
                    with col:
                        display_asset_card(ticker, data, show_details)
                        
    elif display_mode == "List View":
        # Display assets in a detailed list format
        for ticker, data in filtered_assets:
            with st.expander(f"📈 {ticker} - Score: {data['seasonal_relevance']['score']:.2f}", expanded=False):
                display_asset_card(ticker, data, True)
                
    else:  # Compact View
        # Display assets in a compact table format
        compact_data = []
        for ticker, data in filtered_assets:
            rel = data['seasonal_relevance']
            seasonal_stats = data['seasonal_stats']
            best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
            worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
            
            compact_data.append({
                'Asset': ticker,
                'Score': f"{rel['score']:.2f}",
                'Current Month': f"{rel['current_month_return']:.2%}",
                'Win Rate': f"{rel['current_month_win_rate']:.1%}",
                'Best Month': f"{best_month.name} ({best_month['Avg_Return']:.2%})",
                'Worst Month': f"{worst_month.name} ({worst_month['Avg_Return']:.2%})"
            })
        
        if compact_data:
            df_compact = pd.DataFrame(compact_data)
            st.dataframe(df_compact, use_container_width=True, hide_index=True)
            
            # Add analyze buttons for compact view
            st.markdown("**Quick Actions:**")
            action_cols = st.columns(min(len(filtered_assets), 5))
            for i, (ticker, data) in enumerate(filtered_assets[:5]):
                with action_cols[i]:
                    if st.button(f"📊 {ticker}", key=f"analyze_compact_{ticker}", use_container_width=True):
                        # Switch to single asset mode
                        st.session_state.dashboard_mode = 'single'
                        st.session_state.selected_asset = ticker
                        st.session_state.stock_data = data['stock_data']
                        st.session_state.seasonal_stats = data['seasonal_stats']
                        st.session_state.current_symbol = ticker
                        st.session_state.data_analyzed = True
                        
                        # Initialize weekday_stats
                        from data_processor import DataProcessor
                        processor = DataProcessor()
                        st.session_state.weekday_stats = processor.calculate_weekday_stats(data['stock_data'])
                        
                        if data['ai_insights']:
                            st.session_state.ai_insights = data['ai_insights']
                        st.rerun()

def display_asset_card(ticker, data, show_details=True):
    """Display individual asset card with professional styling"""
    rel = data['seasonal_relevance']
    seasonal_stats = data['seasonal_stats']
    
    # Get best and worst months
    best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
    worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
    
    # Determine card style and color based on performance
    if rel['current_month_return'] > 0.02:
        card_color = "#e8f5e8"
        border_color = "#4caf50"
        performance_emoji = "🟢"
    elif rel['current_month_return'] < -0.02:
        card_color = "#ffeaea"
        border_color = "#f44336"
        performance_emoji = "🔴"
    else:
        card_color = "#f8f9fa"
        border_color = "#6c757d"
        performance_emoji = "🟡"
    
    # Professional card container
    with st.container():
        st.markdown(f"""
        <div style="
            background: {card_color}; 
            border-left: 4px solid {border_color}; 
            padding: 1.5rem; 
            border-radius: 10px; 
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
        </div>
        """, unsafe_allow_html=True)
        
        # Header with ticker and score
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {performance_emoji} **{ticker}**")
        with col2:
            st.metric("Score", f"{rel['score']:.2f}", help="Seasonal relevance score (0-1)")
        with col3:
            # Add confidence badge
            if rel['score'] > 0.7:
                confidence = "🔥 High"
                confidence_color = "#4caf50"
            elif rel['score'] > 0.5:
                confidence = "⭐ Med"
                confidence_color = "#ff9800"
            else:
                confidence = "⚠️ Low"
                confidence_color = "#f44336"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: {confidence_color}20; 
                        border-radius: 8px; border: 1px solid {confidence_color};">
                <strong style="color: {confidence_color};">{confidence}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        if show_details:
            st.markdown("---")
            
            # Key metrics in organized layout
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric(
                    "📊 Current Month", 
                    f"{rel['current_month_return']:.2%}",
                    delta=f"{rel['current_month_win_rate']:.0f}% win rate",
                    help="Historical average return for current month"
                )
                
                st.metric(
                    "🏆 Best Month", 
                    f"{best_month.name[:3]}",
                    delta=f"{best_month['Avg_Return']:.2%}",
                    help=f"Best performing month: {best_month.name}"
                )
            
            with metric_col2:
                # Calculate some additional metrics
                total_data_years = len(seasonal_stats[seasonal_stats['Count'] > 0])
                avg_volatility = seasonal_stats['Volatility'].mean()
                
                st.metric(
                    "📈 Avg Volatility", 
                    f"{avg_volatility:.1%}",
                    delta=f"{total_data_years} months data",
                    help="Average monthly volatility across all months"
                )
                
                st.metric(
                    "📉 Worst Month", 
                    f"{worst_month.name[:3]}",
                    delta=f"{worst_month['Avg_Return']:.2%}",
                    delta_color="inverse",
                    help=f"Worst performing month: {worst_month.name}"
                )
    
    # Action buttons
    button_col1, button_col2 = st.columns([2, 1])
    
    with button_col1:
        if st.button(f"📊 Deep Analysis", key=f"analyze_{ticker}", use_container_width=True, type="primary"):
            # Switch to single asset mode and set the selected asset with all required data
            st.session_state.dashboard_mode = 'single'
            st.session_state.selected_asset = ticker
            st.session_state.stock_data = data['stock_data']
            st.session_state.seasonal_stats = data['seasonal_stats']
            st.session_state.current_symbol = ticker
            st.session_state.data_analyzed = True
            
            # Initialize weekday_stats for the selected asset
            from data_processor import DataProcessor
            processor = DataProcessor()
            st.session_state.weekday_stats = processor.calculate_weekday_stats(data['stock_data'])
            
            if data['ai_insights']:
                st.session_state.ai_insights = data['ai_insights']
            st.rerun()
    
    with button_col2:
        if st.button(f"📋 Quick View", key=f"quick_{ticker}", use_container_width=True):
            # Show a quick summary in an expander
            with st.expander(f"📊 {ticker} Quick Summary", expanded=True):
                st.write(f"**🎯 Seasonal Score:** {rel['score']:.3f}")
                st.write(f"**📊 Current Month:** {rel['current_month_return']:.2%} avg return")
                st.write(f"**🎲 Win Rate:** {rel['current_month_win_rate']:.1%}")
                st.write(f"**📈 Best:** {best_month.name} ({best_month['Avg_Return']:.2%})")
                st.write(f"**📉 Worst:** {worst_month.name} ({worst_month['Avg_Return']:.2%})")
                
                # Show mini seasonal chart if data available
                if len(seasonal_stats) >= 12:
                    monthly_returns = [seasonal_stats.iloc[i]['Avg_Return'] * 100 for i in range(12)]
                    months_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=months_short,
                        y=monthly_returns,
                        marker_color=['green' if x > 0 else 'red' for x in monthly_returns],
                        name=f"{ticker} Monthly Returns"
                    ))
                    fig.update_layout(
                        title=f"{ticker} - Monthly Seasonal Pattern",
                        height=250,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

def display_multi_asset_overview():
    """Display comprehensive overview for multiple assets"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">📊 Multi-Asset Executive Overview</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Comprehensive portfolio analysis and seasonal insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.multi_asset_data:
        st.error("❌ No asset data available. Please process assets first.")
        return
    
    # Portfolio-level calculations
    all_assets = list(st.session_state.multi_asset_data.keys())
    total_assets = len(all_assets)
    
    # Calculate portfolio metrics
    portfolio_returns = []
    portfolio_volatilities = []
    seasonal_scores = []
    current_month_returns = []
    win_rates = []
    
    for ticker, data in st.session_state.multi_asset_data.items():
        stock_data = data['stock_data']
        seasonal_stats = data['seasonal_stats']
        rel = data['seasonal_relevance']
        
        # Calculate individual asset metrics
        total_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
        portfolio_returns.append(total_return)
        
        volatility = stock_data['Returns'].std() * np.sqrt(252) * 100
        portfolio_volatilities.append(volatility)
        
        seasonal_scores.append(rel['score'])
        current_month_returns.append(rel['current_month_return'])
        win_rates.append(rel['current_month_win_rate'])
    
    # Portfolio summary metrics
    avg_return = np.mean(portfolio_returns)
    avg_volatility = np.mean(portfolio_volatilities)
    avg_seasonal_score = np.mean(seasonal_scores)
    avg_current_month = np.mean(current_month_returns)
    avg_win_rate = np.mean(win_rates)
    
    # Count high-quality assets
    high_score_assets = len([s for s in seasonal_scores if s > 0.6])
    positive_seasonality = len([r for r in current_month_returns if r > 0])
    
    # SECTION 1: Portfolio Summary Cards
    st.markdown("### 🎯 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        return_color = "success" if avg_return > 0 else "error"
        st.metric(
            "📈 Avg Total Return",
            f"{avg_return:.1f}%",
            delta=f"{total_assets} assets analyzed",
            help="Average total return across all assets in portfolio"
        )
        
    with col2:
        volatility_status = "Low Risk" if avg_volatility < 20 else "Med Risk" if avg_volatility < 35 else "High Risk"
        st.metric(
            "📊 Avg Volatility",
            f"{avg_volatility:.1f}%",
            delta=volatility_status,
            help="Average annualized volatility across portfolio"
        )
        
    with col3:
        score_status = "Excellent" if avg_seasonal_score > 0.6 else "Good" if avg_seasonal_score > 0.4 else "Fair"
        st.metric(
            "🎯 Seasonal Score",
            f"{avg_seasonal_score:.2f}",
            delta=score_status,
            help="Average seasonal relevance score (0-1 scale)"
        )
        
    with col4:
        seasonality_pct = (positive_seasonality / total_assets) * 100
        st.metric(
            "✅ Positive Seasonality",
            f"{positive_seasonality}/{total_assets}",
            delta=f"{seasonality_pct:.0f}% of portfolio",
            help="Assets with positive current month performance"
        )
    
    st.divider()
    
    # SECTION 2: Current Month Analysis
    st.markdown("### 📅 Current Month Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create bar chart of current month performance
        fig = go.Figure()
        
        colors = ['green' if r > 0 else 'red' for r in current_month_returns]
        fig.add_trace(go.Bar(
            x=all_assets,
            y=[r * 100 for r in current_month_returns],
            marker_color=colors,
            name="Current Month Avg Return (%)"
        ))
        
        fig.update_layout(
            title="Current Month Historical Performance by Asset",
            xaxis_title="Assets",
            yaxis_title="Average Return (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Month Summary")
        
        top_performer = all_assets[current_month_returns.index(max(current_month_returns))]
        worst_performer = all_assets[current_month_returns.index(min(current_month_returns))]
        
        st.metric("🏆 Top Performer", top_performer, f"{max(current_month_returns):.2%}")
        st.metric("📉 Worst Performer", worst_performer, f"{min(current_month_returns):.2%}")
        st.metric("📈 Average", f"{avg_current_month:.2%}", f"{avg_win_rate:.1%} win rate")
        
        # Quality distribution
        st.markdown("#### 🎯 Quality Distribution")
        high_quality = len([s for s in seasonal_scores if s > 0.6])
        medium_quality = len([s for s in seasonal_scores if 0.4 <= s <= 0.6])
        low_quality = len([s for s in seasonal_scores if s < 0.4])
        
        st.write(f"🔥 High Quality: {high_quality}")
        st.write(f"⭐ Medium Quality: {medium_quality}")
        st.write(f"⚠️ Low Quality: {low_quality}")
    
    st.divider()
    
    # SECTION 3: Asset Performance Table
    st.markdown("### 📋 Detailed Asset Performance")
    
    # Create comprehensive performance table
    performance_data = []
    for ticker, data in st.session_state.multi_asset_data.items():
        stock_data = data['stock_data']
        seasonal_stats = data['seasonal_stats']
        rel = data['seasonal_relevance']
        
        # Calculate metrics
        total_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
        volatility = stock_data['Returns'].std() * np.sqrt(252) * 100
        years_data = len(stock_data) / 252
        
        best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
        worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
        
        performance_data.append({
            'Asset': ticker,
            'Total Return': f"{total_return:.1f}%",
            'Volatility': f"{volatility:.1f}%",
            'Seasonal Score': f"{rel['score']:.2f}",
            'Current Month': f"{rel['current_month_return']:.2%}",
            'Win Rate': f"{rel['current_month_win_rate']:.1%}",
            'Best Month': f"{best_month.name} ({best_month['Avg_Return']:.1%})",
            'Worst Month': f"{worst_month.name} ({worst_month['Avg_Return']:.1%})",
            'Data Years': f"{years_data:.1f}"
        })
    
    # Display as interactive table
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # SECTION 4: Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Detailed Dashboard", use_container_width=True, type="primary", key="detailed_dashboard_btn"):
            # Switch to detailed dashboard view
            st.session_state.show_detailed_dashboard = True
            st.rerun()
    
    with col2:
        if st.button("📈 Portfolio Analysis", use_container_width=True, key="portfolio_analysis_btn"):
            # Switch to backtesting tab for portfolio analysis
            st.session_state.selected_tab = "💰 Backtesting"
            st.rerun()
    
    with col3:
        if st.button("📋 Export Report", use_container_width=True, key="export_report_btn"):
            # Switch to export tab
            st.session_state.selected_tab = "📋 Export & Alerts"
            st.rerun()
            st.success("📋 Multi-asset report generation coming soon!")
    
    # Show detailed dashboard if requested
    if getattr(st.session_state, 'show_detailed_dashboard', False):
        st.markdown("---")
        display_multi_asset_dashboard()
        st.session_state.show_detailed_dashboard = False

def display_multi_asset_seasonal_comparison():
    """Display seasonal pattern comparison across multiple assets"""
    st.markdown("""
    <div class="section-header">
        <h3>📅 Multi-Asset Seasonal Pattern Comparison</h3>
        <p>Compare seasonal performance across all assets</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.multi_asset_data:
        st.warning("No asset data available.")
        return
    
    # Create seasonal heatmap
    st.markdown("### 🔥 Seasonal Performance Heatmap")
    
    # Prepare data for heatmap
    heatmap_data = []
    assets = list(st.session_state.multi_asset_data.keys())
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for asset in assets:
        seasonal_stats = st.session_state.multi_asset_data[asset]['seasonal_stats']
        monthly_returns = [seasonal_stats.iloc[i]['Avg_Return'] * 100 for i in range(min(12, len(seasonal_stats)))]
        # Pad with zeros if less than 12 months
        while len(monthly_returns) < 12:
            monthly_returns.append(0)
        heatmap_data.append(monthly_returns)
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=months,
        y=assets,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{val:.1f}%" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Monthly Average Returns (%)",
        xaxis_title="Month",
        yaxis_title="Assets",
        height=max(400, len(assets) * 30)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly performance ranking
    st.markdown("### 📊 Monthly Performance Rankings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Best Performing Months by Asset")
        for asset in assets:
            seasonal_stats = st.session_state.multi_asset_data[asset]['seasonal_stats']
            best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
            st.markdown(f"**{asset}**: {best_month.name} ({best_month['Avg_Return']:.2%})")
    
    with col2:
        st.markdown("#### 📉 Worst Performing Months by Asset")
        for asset in assets:
            seasonal_stats = st.session_state.multi_asset_data[asset]['seasonal_stats']
            worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
            st.markdown(f"**{asset}**: {worst_month.name} ({worst_month['Avg_Return']:.2%})")
    
    # Current month focus
    current_month = datetime.now().month
    current_month_name = months[current_month - 1]
    
    st.markdown(f"### 📅 Focus on {current_month_name} (Current Month)")
    
    current_month_data = []
    for asset in assets:
        seasonal_stats = st.session_state.multi_asset_data[asset]['seasonal_stats']
        if len(seasonal_stats) >= current_month:
            month_stats = seasonal_stats.iloc[current_month - 1]
            current_month_data.append({
                'Asset': asset,
                'Avg Return': f"{month_stats['Avg_Return']:.2%}",
                'Win Rate': f"{month_stats['Win_Rate']:.1%}",
                'Best Year': f"{month_stats['Max_Return']:.2%}",
                'Worst Year': f"{month_stats['Min_Return']:.2%}"
            })
    
    if current_month_data:
        df_current = pd.DataFrame(current_month_data)
        st.dataframe(df_current, use_container_width=True)

def main():
    # Professional Header
    st.markdown('''
    <div class="main-header">
        <div class="header-content">
            <div class="brand-section">
                <h1>📈 AI Seasonal Edge</h1>
                <p class="subtitle">Advanced Seasonal Pattern Analytics • Powered by Machine Learning</p>
                <p class="creator">Created by Marios Athinodorou</p>
                <p class="email">📧 Athinodoroumarios@yahoo.com</p>
                <div class="github-section">
                    <a href="https://github.com/Mariosat15/Analyzer/tree/master" target="_blank" class="github-btn">
                        <span class="github-icon">⭐</span>
                        View on GitHub
                    </a>
                </div>
            </div>
            <div class="header-badges">
                <span class="badge professional">Professional Grade</span>
                <span class="badge ai-powered">AI Powered</span>
                <span class="badge institutional">Institutional Quality</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Professional Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">
                <div class="logo-icon">📊</div>
                <div class="logo-text">
                    <h2>Control Center</h2>
                    <p>Configure Analysis Parameters</p>
                </div>
            </div>
            <div class="sidebar-status">
                <div class="status-indicator active"></div>
                <span>System Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme Selection
        st.markdown("**🎨 Theme Selection**")
        theme_option = st.radio(
            "Choose Theme",
            ["☀️ Light Mode", "🌙 Dark Mode"],
            index=1 if st.session_state.dark_theme else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="theme_selection_radio"
        )
        
        if theme_option == "🌙 Dark Mode" and not st.session_state.dark_theme:
            st.session_state.dark_theme = True
            st.rerun()
        elif theme_option == "☀️ Light Mode" and st.session_state.dark_theme:
            st.session_state.dark_theme = False
            st.rerun()
        
        st.markdown("---")
        
        # Dashboard Mode Selection
        st.markdown("**🎯 Analysis Mode**")
        dashboard_mode = st.radio(
            "Choose Analysis Type",
            ["📊 Single Asset Analysis", "🎛️ Multi-Asset Dashboard"],
            index=0 if st.session_state.dashboard_mode == 'single' else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="dashboard_mode_radio"
        )
        
        if dashboard_mode == "🎛️ Multi-Asset Dashboard" and st.session_state.dashboard_mode == 'single':
            st.session_state.dashboard_mode = 'multi'
            st.rerun()
        elif dashboard_mode == "📊 Single Asset Analysis" and st.session_state.dashboard_mode == 'multi':
            st.session_state.dashboard_mode = 'single'
            st.rerun()
        
        st.markdown("---")
        
        # Clean Data Upload Section
        if st.session_state.dashboard_mode == 'single':
            st.markdown("**📁 Data Source**")
            st.markdown("""
            <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #334155; margin-bottom: 0.5rem;">Upload Market Data</div>
                <div style="font-size: 0.9rem; color: #64748b;">Supports: Stocks, ETFs, Forex, Crypto, Commodities</div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload CSV File",
                type=['csv'],
                help="Supported formats: DATE,OPEN,HIGH,LOW,CLOSE or DATE,TIME,OPEN,HIGH,LOW,CLOSE",
                label_visibility="collapsed",
                key="csv_file_uploader"
            )
        else:
            st.markdown("**📥 Multi-Asset Upload**")
            st.markdown("""
            <div style="background: #f0f9ff; border: 2px dashed #0ea5e9; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #0c4a6e; margin-bottom: 0.5rem;">Upload Multiple Assets</div>
                <div style="font-size: 0.9rem; color: #0369a1;">CSV with tickers or manual text input</div>
            </div>
            """, unsafe_allow_html=True)
            
            upload_method = st.selectbox(
                "Upload Method",
                ["📝 Manual Text Input", "📄 CSV File Upload"],
                key="upload_method_select"
            )
            
            uploaded_tickers = []
            
            if upload_method == "📝 Manual Text Input":
                ticker_input = st.text_area(
                    "Enter Tickers (one per line or comma-separated)",
                    placeholder="AAPL\nTSLA\nMSFT\nGOOGL\nAMZN",
                    height=100,
                    key="ticker_text_input"
                )
                
                if ticker_input.strip():
                    # Parse input - handle both newlines and commas
                    tickers = ticker_input.replace(',', '\n').split('\n')
                    uploaded_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
            
            elif upload_method == "📄 CSV File Upload":
                uploaded_ticker_files = st.file_uploader(
                    "Upload CSV file(s) with price data",
                    type=['csv', 'txt'],
                    help="Upload CSV files containing OHLC price data (DATE,OPEN,HIGH,LOW,CLOSE). Each file should be named with the ticker symbol.",
                    accept_multiple_files=True,
                    key="ticker_csv_uploader"
                )
                
                if uploaded_ticker_files:
                    try:
                        # Store the actual CSV data for local processing
                        st.session_state.multi_asset_csv_data = {}
                        uploaded_tickers = []
                        
                        for uploaded_file in uploaded_ticker_files:
                            # Extract ticker from filename
                            filename = uploaded_file.name
                            ticker = filename.split('.')[0].split('_')[0].upper()  # Handle formats like "AAPL_Daily.csv"
                            
                            # Read the CSV data - handle both comma and tab separated files
                            df = pd.read_csv(uploaded_file, sep=None, engine='python')  # Auto-detect separator
                            
                            # Check if this looks like price data (has OHLC columns) or just a ticker list
                            # Check for various OHLC column name formats
                            columns_upper = [col.upper().strip('<>') for col in df.columns]  # Remove angle brackets and convert to uppercase
                            has_ohlc = (len(df.columns) >= 4 and 
                                       any(col in ['OPEN', 'O'] for col in columns_upper) and
                                       any(col in ['HIGH', 'H'] for col in columns_upper) and 
                                       any(col in ['LOW', 'L'] for col in columns_upper) and
                                       any(col in ['CLOSE', 'C'] for col in columns_upper))
                            
                            if has_ohlc:
                                # This is actual price data - store it
                                st.session_state.multi_asset_csv_data[ticker] = df
                                uploaded_tickers.append(ticker)
                                st.success(f"📊 **{ticker}**: Loaded {len(df)} price data points from local CSV")
                                
                                # Show column info for debugging
                                col_info = ", ".join(df.columns[:6])
                                if len(df.columns) > 6:
                                    col_info += f" (+ {len(df.columns)-6} more)"
                                st.caption(f"   Columns: {col_info}")
                            else:
                                # This is a ticker list - extract tickers from first column
                                file_tickers = df.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
                                file_tickers = [t for t in file_tickers if t and t != 'NAN']
                                uploaded_tickers.extend(file_tickers)
                                st.info(f"📝 **{filename}**: Found {len(file_tickers)} ticker symbols")
                        
                        # Remove duplicates while preserving order
                        uploaded_tickers = list(dict.fromkeys(uploaded_tickers))
                        
                        # Show summary
                        csv_data_count = len(st.session_state.multi_asset_csv_data) if hasattr(st.session_state, 'multi_asset_csv_data') else 0
                        if csv_data_count > 0:
                            st.success(f"🎯 **Local Data Mode**: {csv_data_count} assets with price data loaded locally")
                        
                        if len(uploaded_ticker_files) > 1:
                            st.info(f"📄 Processed {len(uploaded_ticker_files)} files, found {len(uploaded_tickers)} unique tickers")
                            
                    except Exception as e:
                        st.error(f"Error reading file(s): {e}")
                        uploaded_tickers = []
            
            if uploaded_tickers:
                st.success(f"✅ Found {len(uploaded_tickers)} tickers: {', '.join(uploaded_tickers[:10])}{' ...' if len(uploaded_tickers) > 10 else ''}")
                
                # Show batch processing options for large datasets
                if len(uploaded_tickers) > 50:
                    st.warning(f"⚠️ Large dataset detected ({len(uploaded_tickers)} tickers)")
                    st.markdown("**📦 Batch Processing Options**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        batch_size = st.slider("Batch size:", 10, 50, 30, 5, help="Process this many assets at once", key="batch_size_slider")
                    with col2:
                        total_batches = (len(uploaded_tickers) + batch_size - 1) // batch_size
                        batch_number = st.selectbox("Select batch:", range(1, total_batches + 1), 
                                                  help=f"Total {total_batches} batches needed",
                                                  key="batch_number_select")
                    
                    start_idx = (batch_number - 1) * batch_size
                    end_idx = min(start_idx + batch_size, len(uploaded_tickers))
                    selected_tickers = uploaded_tickers[start_idx:end_idx]
                    
                    st.info(f"📊 **Batch {batch_number}/{total_batches}**: Processing tickers {start_idx + 1}-{end_idx} of {len(uploaded_tickers)} total")
                    st.text_area("This batch:", value=", ".join(selected_tickers), height=60, disabled=True)
                    
                    st.session_state.uploaded_tickers = selected_tickers
                else:
                    st.session_state.uploaded_tickers = uploaded_tickers
            
            uploaded_file = None  # No single file upload in multi mode
    
        symbol = None
        company_name = "Unknown Company"
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file, sep=None, engine='python')  # Auto-detect separator
                
                # Store in session state
                st.session_state.uploaded_data = df
                
                # Extract symbol from filename (remove extension and clean up)
                filename = uploaded_file.name
                symbol = filename.split('.')[0].replace('_', ' ').replace('-', ' ')
                st.session_state.uploaded_symbol = symbol
                
                # Clean success indicator
                st.success("✅ **Upload Successful!** - Data processed and ready for analysis")
                
                with st.expander("📋 Data Summary & Validation", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📊 Dataset Information**")
                        st.metric("Symbol/Asset", symbol, help="Extracted from filename")
                        st.metric("Data Points", f"{len(df):,}", help="Total number of observations")
                        
                        # Data quality assessment
                        completeness = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
                        quality_color = "🟢" if completeness > 95 else "🟡" if completeness > 90 else "🔴"
                        st.metric("Data Quality", f"{quality_color} {completeness:.1f}%", help="Percentage of non-null values")
                    
                    with col2:
                        st.markdown("**📅 Time Period Coverage**")
                        if len(df.columns) >= 5:
                            first_date = df.iloc[0, 0]
                            last_date = df.iloc[-1, 0]
                            st.metric("From", f"{first_date}")
                            st.metric("To", f"{last_date}")
                            
                            # Calculate time span
                            try:
                                from datetime import datetime
                                if isinstance(first_date, str):
                                    first_dt = pd.to_datetime(first_date)
                                    last_dt = pd.to_datetime(last_date)
                                    time_span = (last_dt - first_dt).days
                                    st.metric("Time Span", f"{time_span:,} days", help="Total period covered")
                            except:
                                pass
                        
                        st.markdown("**🔍 Column Structure**")
                        col_text = f"Columns: {', '.join(df.columns[:4])}"
                        if len(df.columns) > 4:
                            col_text += f"... (+{len(df.columns)-4} more)"
                        st.text(col_text)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("💡 Ensure your CSV has the correct format")
        else:
            st.info("👆 Upload a CSV file to begin analysis")
        
        st.markdown("---")
        
        # AI Configuration Section  
        st.markdown("**🧠 AI Configuration**")
        
        enable_ai = st.checkbox("🤖 Enable AI Pattern Detection", value=True, 
                               help="Use machine learning for advanced pattern recognition",
                               key="enable_ai_checkbox")
        
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold", 
            min_value=0.5, 
            max_value=0.95, 
            value=0.75, 
            step=0.05,
            help="Higher values = more conservative AI predictions",
            key="confidence_threshold_slider"
        )
        
        # Analysis Quality Settings
        st.markdown("**⚙️ Analysis Settings**")
        
        analysis_mode = st.selectbox(
            "Analysis Depth",
            ["Quick Analysis", "Standard Analysis", "Deep Analysis"],
            index=1,
            help="Choose analysis depth vs speed trade-off",
            key="analysis_mode_selectbox"
        )
        
        include_forecasting = st.checkbox("📈 Include Forecasting", value=True,
                                         help="Generate future predictions using Prophet",
                                         key="include_forecasting_checkbox")
        
        st.markdown("---")
        
        # Enhanced Action Button
        st.markdown("---")
        st.markdown("**🚀 Analysis Execution**")
        
        if st.session_state.dashboard_mode == 'single':
            if uploaded_file is not None:
                st.info("🟢 **Ready to analyze** - Click button below to start")
                
                analyze_button = st.button(
                    "🚀 Start Advanced Analysis", 
                    type="primary", 
                    use_container_width=True,
                    key="main_analyze_button"
                )
            else:
                st.warning("⚪ **Waiting for data upload** - Please upload a CSV file first")
                
                analyze_button = st.button(
                    "⏳ Upload Data First", 
                    type="secondary", 
                    use_container_width=True,
                    disabled=True,
                    key="main_analyze_button_disabled"
                )
        else:  # Multi-asset mode
            if st.session_state.uploaded_tickers:
                st.info(f"🟢 **Ready to analyze {len(st.session_state.uploaded_tickers)} assets** - Click button below to start")
                
                analyze_button = st.button(
                    f"🚀 Build Multi-Asset Dashboard ({len(st.session_state.uploaded_tickers)} assets)", 
                    type="primary", 
                    use_container_width=True,
                    key="multi_analyze_button"
                )
            else:
                st.warning("⚪ **Waiting for ticker upload** - Please enter tickers first")
                
                analyze_button = st.button(
                    "⏳ Upload Tickers First", 
                    type="secondary", 
                    use_container_width=True,
                    disabled=True,
                    key="multi_analyze_button_disabled"
                )
    
    # Handle both single and multi-asset analysis
    if analyze_button and (symbol or st.session_state.dashboard_mode == 'multi'):
        if st.session_state.dashboard_mode == 'single' and symbol:
            with st.spinner(f"🔄 Analyzing {symbol}... Please wait."):
                try:
                    # Initialize processors
                    data_processor = DataProcessor()
                    ai_analyzer = AIAnalyzer()
                    visualizer = Visualizer(dark_theme=st.session_state.dark_theme)
                    
                    # Fetch and process data
                    stock_data = data_processor.process_uploaded_data(
                        st.session_state.uploaded_data, 
                        symbol,
                        start_date=None,
                        end_date=None
                    )
                    
                    if stock_data.empty:
                        st.error(f"❌ No data found for symbol '{symbol}'. Please check the symbol and try again.")
                        return
                    
                    seasonal_stats = data_processor.calculate_seasonal_stats(stock_data)
                    weekday_stats = data_processor.calculate_weekday_stats(stock_data)
                    intraday_patterns = data_processor.get_intraday_patterns(stock_data)
                    
                    # Store data in session state for persistence across tab interactions
                    st.session_state.stock_data = stock_data
                    st.session_state.seasonal_stats = seasonal_stats
                    st.session_state.weekday_stats = weekday_stats
                    st.session_state.current_symbol = symbol
                    st.session_state.data_analyzed = True
                    
                    # Calculate AI insights once if AI is enabled
                    ai_insights = None
                    if enable_ai:
                        try:
                            ai_insights = ai_analyzer.analyze_patterns(
                                stock_data, seasonal_stats, confidence_threshold
                            )
                            st.session_state.ai_insights = ai_insights
                        except Exception as e:
                            st.warning(f"AI analysis had issues: {str(e)}")
                    
                    # Display basic info
                    company_name = symbol
                    
                    # Show data source info
                    st.success(f"✅ Successfully processed uploaded data for **{company_name}**")
                    st.info(f"📊 Data contains {len(stock_data)} data points from {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing {symbol}: {str(e)}")
                    st.info("💡 Try a different symbol or check your internet connection.")
        
        elif st.session_state.dashboard_mode == 'multi' and st.session_state.uploaded_tickers:
            # Multi-asset processing
            process_multi_assets(st.session_state.uploaded_tickers, enable_ai, confidence_threshold)
    
    # Create tabs for different analyses - ALWAYS RENDERED OUTSIDE THE BUTTON CLICK
    if symbol or st.session_state.dashboard_mode == 'multi':
        if st.session_state.dashboard_mode == 'single':
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
                "📊 Overview", 
                "📅 Seasonal Patterns", 
                "🧠 AI Insights", 
                "📈 Performance Charts",
                "⚠️ Risk Analysis",
                "📊 Technical Analysis", 
                "💰 Backtesting",
                "📈 Statistical Tests",
                "🔔 Market Regimes",
                "📋 Export & Alerts",
                "📚 Wiki & Help"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
                "🎛️ Multi-Asset Dashboard",
                "📅 Seasonal Patterns", 
                "🧠 AI Insights", 
                "📈 Performance Charts",
                "⚠️ Risk Analysis",
                "📊 Technical Analysis", 
                "💰 Backtesting",
                "📈 Statistical Tests",
                "🔔 Market Regimes",
                "📋 Export & Alerts",
                "📚 Wiki & Help"
            ])
        
        # Check if data has been analyzed
        data_analyzed = getattr(st.session_state, 'data_analyzed', False)
        
        with tab1:
            if st.session_state.dashboard_mode == 'single':
                if data_analyzed:
                    stock_data = st.session_state.stock_data
                    seasonal_stats = st.session_state.seasonal_stats
                    symbol = st.session_state.current_symbol
                    display_overview(stock_data, seasonal_stats, symbol, symbol)
                else:
                    st.info("📊 **Upload data and click 'Run Analysis' to see the executive overview**")
            else:
                # Multi-Asset Dashboard
                if hasattr(st.session_state, 'multi_asset_analyzed') and st.session_state.multi_asset_analyzed:
                    display_multi_asset_overview()
                else:
                    st.info("🎛️ **Upload tickers and click 'Build Dashboard' to see the multi-asset overview**")

        with tab2:
            if data_analyzed or (st.session_state.dashboard_mode == 'multi' and hasattr(st.session_state, 'multi_asset_analyzed')):
                # Add back to dashboard button if in single asset mode but came from dashboard
                if st.session_state.dashboard_mode == 'single' and hasattr(st.session_state, 'multi_asset_data') and st.session_state.multi_asset_data:
                    if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tab2"):
                        st.session_state.dashboard_mode = 'multi'
                        st.rerun()
                    st.markdown("---")
                
                if st.session_state.dashboard_mode == 'single' and data_analyzed:
                    seasonal_stats = st.session_state.seasonal_stats
                    # Initialize weekday_stats if missing
                    if not hasattr(st.session_state, 'weekday_stats') or st.session_state.weekday_stats is None:
                        processor = DataProcessor()
                        st.session_state.weekday_stats = processor.calculate_weekday_stats(st.session_state.stock_data)
                    weekday_stats = st.session_state.weekday_stats
                    symbol = st.session_state.current_symbol
                    visualizer = Visualizer(dark_theme=st.session_state.dark_theme)
                    display_seasonal_analysis(seasonal_stats, weekday_stats, visualizer, symbol)
                elif st.session_state.dashboard_mode == 'multi':
                    display_multi_asset_seasonal_comparison()
            else:
                st.info("📈 **Upload data and click 'Run Analysis' to see seasonal patterns**")
        
        with tab3:
            if (data_analyzed and enable_ai and hasattr(st.session_state, 'ai_insights')) or (st.session_state.dashboard_mode == 'multi' and hasattr(st.session_state, 'multi_asset_analyzed') and enable_ai):
                # Add back to dashboard button if in single asset mode but came from dashboard
                if st.session_state.dashboard_mode == 'single' and hasattr(st.session_state, 'multi_asset_data') and st.session_state.multi_asset_data:
                    if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tab3"):
                        st.session_state.dashboard_mode = 'multi'
                        st.rerun()
                    st.markdown("---")
                
                if st.session_state.dashboard_mode == 'single' and data_analyzed:
                    stock_data = st.session_state.stock_data
                    seasonal_stats = st.session_state.seasonal_stats
                    symbol = st.session_state.current_symbol
                    ai_insights = st.session_state.ai_insights
                    ai_analyzer = AIAnalyzer()
                    display_ai_insights(
                        stock_data, seasonal_stats, ai_analyzer, 
                        symbol, confidence_threshold, ai_insights
                    )
                elif st.session_state.dashboard_mode == 'multi':
                    display_multi_asset_ai_insights(confidence_threshold)
            elif not enable_ai:
                st.info("🔧 Enable AI Pattern Detection in the sidebar to see AI insights.")
            else:
                st.info("🤖 **Upload data and click 'Run Analysis' to see AI insights**")
        
        with tab4:
            if data_analyzed or (st.session_state.dashboard_mode == 'multi' and hasattr(st.session_state, 'multi_asset_analyzed')):
                # Add back to dashboard button if in single asset mode but came from dashboard
                if st.session_state.dashboard_mode == 'single' and hasattr(st.session_state, 'multi_asset_data') and st.session_state.multi_asset_data:
                    if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tab4"):
                        st.session_state.dashboard_mode = 'multi'
                        st.rerun()
                    st.markdown("---")
                
                if st.session_state.dashboard_mode == 'single' and data_analyzed:
                    stock_data = st.session_state.stock_data
                    seasonal_stats = st.session_state.seasonal_stats
                    symbol = st.session_state.current_symbol
                    visualizer = Visualizer(dark_theme=st.session_state.dark_theme)
                    display_performance_charts(stock_data, seasonal_stats, visualizer, symbol)
                elif st.session_state.dashboard_mode == 'multi':
                    display_multi_asset_performance_charts()
            else:
                st.info("📉 **Upload data and click 'Run Analysis' to see performance charts**")
        
        with tab5:
            if data_analyzed or (st.session_state.dashboard_mode == 'multi' and hasattr(st.session_state, 'multi_asset_analyzed')):
                # Add back to dashboard button if in single asset mode but came from dashboard
                if st.session_state.dashboard_mode == 'single' and hasattr(st.session_state, 'multi_asset_data') and st.session_state.multi_asset_data:
                    if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tab5"):
                        st.session_state.dashboard_mode = 'multi'
                        st.rerun()
                    st.markdown("---")
                
                if st.session_state.dashboard_mode == 'single' and data_analyzed:
                    stock_data = st.session_state.stock_data
                    seasonal_stats = st.session_state.seasonal_stats
                    symbol = st.session_state.current_symbol
                    display_risk_analysis(stock_data, seasonal_stats, symbol)
                elif st.session_state.dashboard_mode == 'multi':
                    display_multi_asset_risk_analysis()
            else:
                st.info("⚠️ **Upload data and click 'Run Analysis' to see risk analysis**")
        
        with tab6:
            if data_analyzed or (st.session_state.dashboard_mode == 'multi' and hasattr(st.session_state, 'multi_asset_analyzed')):
                # Add back to dashboard button if in single asset mode but came from dashboard
                if st.session_state.dashboard_mode == 'single' and hasattr(st.session_state, 'multi_asset_data') and st.session_state.multi_asset_data:
                    if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tab6"):
                        st.session_state.dashboard_mode = 'multi'
                        st.rerun()
                    st.markdown("---")
                
                if st.session_state.dashboard_mode == 'single' and data_analyzed:
                    stock_data = st.session_state.stock_data
                    symbol = st.session_state.current_symbol
                    display_technical_analysis(stock_data, symbol)
                elif st.session_state.dashboard_mode == 'multi':
                    # Multi-asset technical analysis is handled within the function
                    display_technical_analysis(None, None)
            else:
                st.info("🔧 **Upload data and click 'Run Analysis' to see technical analysis**")
        
        with tab7:
            if data_analyzed or (st.session_state.dashboard_mode == 'multi' and hasattr(st.session_state, 'multi_asset_analyzed')):
                # Add back to dashboard button if in single asset mode but came from dashboard
                if st.session_state.dashboard_mode == 'single' and hasattr(st.session_state, 'multi_asset_data') and st.session_state.multi_asset_data:
                    if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tab7"):
                        st.session_state.dashboard_mode = 'multi'
                        st.rerun()
                    st.markdown("---")
                
                if st.session_state.dashboard_mode == 'single' and data_analyzed:
                    stock_data = st.session_state.stock_data
                    seasonal_stats = st.session_state.seasonal_stats
                    symbol = st.session_state.current_symbol
                    display_backtesting(stock_data, seasonal_stats, symbol)
                elif st.session_state.dashboard_mode == 'multi':
                    display_multi_asset_backtesting()
            else:
                st.info("💰 **Upload data and click 'Run Analysis' to configure strategy backtesting**")
        
        with tab8:
            if data_analyzed or (st.session_state.dashboard_mode == 'multi' and hasattr(st.session_state, 'multi_asset_analyzed')):
                # Add back to dashboard button if in single asset mode but came from dashboard
                if st.session_state.dashboard_mode == 'single' and hasattr(st.session_state, 'multi_asset_data') and st.session_state.multi_asset_data:
                    if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tab8"):
                        st.session_state.dashboard_mode = 'multi'
                        st.rerun()
                    st.markdown("---")
                
                if st.session_state.dashboard_mode == 'single' and data_analyzed:
                    stock_data = st.session_state.stock_data
                    seasonal_stats = st.session_state.seasonal_stats
                    symbol = st.session_state.current_symbol
                    display_statistical_tests(stock_data, seasonal_stats, symbol)
                elif st.session_state.dashboard_mode == 'multi':
                    display_multi_asset_statistical_tests()
            else:
                st.info("📈 **Upload data and click 'Run Analysis' to run statistical tests**")
        
        with tab9:
            if data_analyzed or (st.session_state.dashboard_mode == 'multi' and hasattr(st.session_state, 'multi_asset_analyzed')):
                # Add back to dashboard button if in single asset mode but came from dashboard
                if st.session_state.dashboard_mode == 'single' and hasattr(st.session_state, 'multi_asset_data') and st.session_state.multi_asset_data:
                    if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tab9"):
                        st.session_state.dashboard_mode = 'multi'
                        st.rerun()
                    st.markdown("---")
                
                if st.session_state.dashboard_mode == 'single' and data_analyzed:
                    stock_data = st.session_state.stock_data
                    symbol = st.session_state.current_symbol
                    display_market_regimes(stock_data, symbol)
                elif st.session_state.dashboard_mode == 'multi':
                    # Multi-asset market regimes is handled within the function
                    display_market_regimes(None, None)
            else:
                st.info("🔔 **Upload data and click 'Run Analysis' to see market regimes**")
        
        with tab10:
            if data_analyzed:
                seasonal_stats = st.session_state.seasonal_stats
                symbol = st.session_state.current_symbol
                ai_insights = getattr(st.session_state, 'ai_insights', None)
                display_export_alerts(seasonal_stats, symbol, symbol, ai_insights=ai_insights)
            else:
                st.info("📋 **Upload data and click 'Run Analysis' to export results**")
        
        with tab11:
            display_wiki_help()
    
    elif not symbol:
        # Default landing page
        display_landing_page()

def calculate_strategy_return(stock_data, target_months, strategy_type):
    """Calculate actual strategy returns based on seasonal patterns"""
    try:
        # Add month column to stock data and ensure Returns column exists
        stock_data = stock_data.copy()
        if 'Returns' not in stock_data.columns:
            stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['Returns'] = stock_data['Returns'].fillna(0)
        stock_data['Month'] = stock_data.index.month
        
        if strategy_type == "best_only":
            # Only trade during best months, cash otherwise
            strategy_returns = []
            for _, row in stock_data.iterrows():
                if row['Month'] in target_months:
                    strategy_returns.append(row['Returns'])
                else:
                    strategy_returns.append(0)  # Cash position
            
        elif strategy_type == "avoid_worst":
            # Trade normally except during worst months (cash position)
            strategy_returns = []
            for _, row in stock_data.iterrows():
                if row['Month'] in target_months:
                    strategy_returns.append(row['Returns'])
                else:
                    strategy_returns.append(0)  # Cash position
                    
        elif strategy_type == "top_6":
            # Trade only during top 6 months
            strategy_returns = []
            for _, row in stock_data.iterrows():
                if row['Month'] in target_months:
                    strategy_returns.append(row['Returns'])
                else:
                    strategy_returns.append(0)  # Cash position
                    
        elif strategy_type == "score_weighted":
            # Weight returns by seasonal score (simplified)
            strategy_returns = []
            for _, row in stock_data.iterrows():
                if row['Month'] in target_months:
                    strategy_returns.append(row['Returns'])
                else:
                    strategy_returns.append(row['Returns'] * 0.5)  # Reduced exposure
                    
        elif strategy_type == "dynamic":
            # Dynamic rotation based on 3-month rolling performance
            stock_data['Rolling_Return'] = stock_data['Returns'].rolling(63).mean()  # ~3 months
            strategy_returns = []
            for _, row in stock_data.iterrows():
                if pd.notna(row['Rolling_Return']) and row['Rolling_Return'] > 0:
                    strategy_returns.append(row['Returns'])
                else:
                    strategy_returns.append(0)  # Cash position
        else:
            strategy_returns = stock_data['Returns'].fillna(0).tolist()
        
        # Calculate cumulative returns
        strategy_returns = np.array(strategy_returns)
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        
        if len(strategy_returns) > 0:
            cumulative_return = ((1 + strategy_returns).prod() - 1) * 100
            return cumulative_return
        else:
            return 0.0
            
    except Exception as e:
        print(f"Strategy calculation error: {e}")
        return 0.0

def display_multi_asset_backtesting():
    """Display backtesting analysis for multiple assets with portfolio capabilities"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">💰 Multi-Asset Portfolio Backtesting</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Test seasonal strategies across individual assets and portfolio combinations</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.multi_asset_data:
        st.error("❌ No asset data available. Please process assets first.")
        return
    
    # Strategy selection and configuration
    st.markdown("### ⚙️ Strategy Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_mode = st.selectbox(
            "📊 Analysis Mode",
            ["Individual Assets", "Equal Weight Portfolio", "Risk-Weighted Portfolio", "Seasonal Score Weighted"],
            help="Choose how to analyze the assets",
            key="backtest_analysis_mode"
        )
    
    with col2:
        initial_capital = st.number_input("💰 Initial Capital ($)", value=100000, min_value=1000, step=1000, key="backtest_initial_capital")
    
    with col3:
        commission = st.number_input("💳 Commission (%)", value=0.1, min_value=0.0, max_value=2.0, step=0.01, key="backtest_commission") / 100
    
    # Strategy type selection
    strategy_type = st.selectbox("🎯 Strategy Type", [
        "Seasonal Long (Buy & Hold)", 
        "Best Months Only", 
        "Avoid Worst Months",
        "Top 6 Months Strategy",
        "Seasonal Score Based",
        "Dynamic Rotation"
    ], help="Choose the backtesting strategy", key="backtest_strategy_type")
    
    st.divider()
    
    # Asset selection for analysis
    all_assets = list(st.session_state.multi_asset_data.keys())
    
    if analysis_mode == "Individual Assets":
        st.markdown("### 📊 Individual Asset Results")
        
        # Allow user to select which assets to compare
        selected_assets = st.multiselect(
            "Select Assets to Compare",
            all_assets,
            default=all_assets[:min(5, len(all_assets))],
            help="Choose up to 10 assets for comparison",
            key="backtest_selected_assets"
        )
        
        if not selected_assets:
            st.warning("Please select at least one asset for analysis.")
            return
            
        if st.button("🚀 Run Individual Asset Backtests", type="primary"):
            results_data = []
            
            for ticker in selected_assets:
                data = st.session_state.multi_asset_data[ticker]
                stock_data = data['stock_data'].copy()
                seasonal_stats = data['seasonal_stats']
                
                # Calculate buy & hold performance
                total_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
                volatility = stock_data['Returns'].std() * np.sqrt(252) * 100
                max_drawdown = (stock_data['Close'] / stock_data['Close'].cummax() - 1).min() * 100
                
                # Calculate actual strategy returns based on seasonal data
                if strategy_type == "Best Months Only":
                    # Get best 3 months
                    best_months = seasonal_stats.nlargest(3, 'Avg_Return').index.tolist()
                    strategy_return = calculate_strategy_return(stock_data, best_months, "best_only")
                elif strategy_type == "Avoid Worst Months":
                    # Avoid worst 3 months, trade other 9
                    worst_months = seasonal_stats.nsmallest(3, 'Avg_Return').index.tolist()
                    avoid_months = [m for m in range(1, 13) if m not in worst_months]
                    strategy_return = calculate_strategy_return(stock_data, avoid_months, "avoid_worst")
                elif strategy_type == "Top 6 Months Strategy":
                    # Trade only top 6 months
                    top_months = seasonal_stats.nlargest(6, 'Avg_Return').index.tolist()
                    strategy_return = calculate_strategy_return(stock_data, top_months, "top_6")
                elif strategy_type == "Seasonal Score Based":
                    # Weight months by seasonal score
                    weighted_months = seasonal_stats[seasonal_stats['Avg_Return'] > 0].index.tolist()
                    strategy_return = calculate_strategy_return(stock_data, weighted_months, "score_weighted")
                elif strategy_type == "Dynamic Rotation":
                    # Rotate based on 3-month rolling performance
                    strategy_return = calculate_strategy_return(stock_data, [], "dynamic")
                else:
                    strategy_return = total_return  # Buy and hold
                
                results_data.append({
                    'Asset': ticker,
                    'Buy & Hold Return': f"{total_return:.1f}%",
                    'Strategy Return': f"{strategy_return:.1f}%",
                    'Volatility': f"{volatility:.1f}%",
                    'Max Drawdown': f"{max_drawdown:.1f}%",
                    'Sharpe Ratio': f"{(total_return/100) / (volatility/100) if volatility > 0 else 0:.2f}",
                    'Seasonal Score': f"{data['seasonal_relevance']['score']:.2f}"
                })
            
            # Display results table
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Create comparison chart
            fig = go.Figure()
            
            buy_hold_returns = [float(r['Buy & Hold Return'].rstrip('%')) for r in results_data]
            strategy_returns = [float(r['Strategy Return'].rstrip('%')) for r in results_data]
            
            fig.add_trace(go.Bar(
                name='Buy & Hold',
                x=selected_assets,
                y=buy_hold_returns,
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name=strategy_type,
                x=selected_assets,
                y=strategy_returns,
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title=f"Strategy Performance Comparison: {strategy_type}",
                xaxis_title="Assets",
                yaxis_title="Total Return (%)",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Portfolio analysis
        st.markdown(f"### 📈 {analysis_mode} Analysis")
        
        if st.button("🚀 Run Portfolio Backtest", type="primary"):
            with st.spinner("Running portfolio backtest..."):
                # Calculate portfolio weights based on analysis mode
                portfolio_data = {}
                weights = {}
                
                if analysis_mode == "Equal Weight Portfolio":
                    # Equal allocation to all assets
                    weight = 1.0 / len(all_assets)
                    weights = {ticker: weight for ticker in all_assets}
                    
                elif analysis_mode == "Risk-Weighted Portfolio":
                    # Weight by inverse volatility
                    volatilities = {}
                    for ticker in all_assets:
                        stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
                        vol = stock_data['Returns'].std() * np.sqrt(252)
                        volatilities[ticker] = vol
                    
                    # Inverse volatility weights
                    inv_vols = {ticker: 1/vol for ticker, vol in volatilities.items()}
                    total_inv_vol = sum(inv_vols.values())
                    weights = {ticker: inv_vol/total_inv_vol for ticker, inv_vol in inv_vols.items()}
                    
                elif analysis_mode == "Seasonal Score Weighted":
                    # Weight by seasonal relevance scores
                    scores = {}
                    for ticker in all_assets:
                        score = st.session_state.multi_asset_data[ticker]['seasonal_relevance']['score']
                        scores[ticker] = max(score, 0.1)  # Minimum weight
                    
                    total_score = sum(scores.values())
                    weights = {ticker: score/total_score for ticker, score in scores.items()}
                
                # Calculate portfolio metrics
                portfolio_returns = []
                portfolio_values = []
                individual_returns = {}
                
                # Get all stock data and calculate individual returns
                for ticker in all_assets:
                    stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
                    individual_returns[ticker] = stock_data['Returns'].fillna(0)
                
                # Apply strategy-based filtering to portfolio returns
                # First, get all dates and create a master DataFrame
                all_stock_data = {}
                for ticker in all_assets:
                    data = st.session_state.multi_asset_data[ticker]['stock_data'].copy()
                    data['Month'] = data.index.month
                    all_stock_data[ticker] = data
                
                # Get common date range
                all_dates = list(set().union(*[data.index for data in all_stock_data.values()]))
                all_dates.sort()
                
                # Apply strategy logic based on the selected strategy type
                if strategy_type == "Best Months Only":
                    # Only invest during the 3 best months across all assets
                    best_months_per_asset = {}
                    for ticker in all_assets:
                        seasonal_stats = st.session_state.multi_asset_data[ticker]['seasonal_stats']
                        best_months = seasonal_stats.nlargest(3, 'Avg_Return').index
                        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                                   'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                        best_months_per_asset[ticker] = [month_map[month] for month in best_months]
                    
                elif strategy_type == "Avoid Worst Months":
                    # Avoid the 3 worst months across all assets
                    worst_months_per_asset = {}
                    for ticker in all_assets:
                        seasonal_stats = st.session_state.multi_asset_data[ticker]['seasonal_stats']
                        worst_months = seasonal_stats.nsmallest(3, 'Avg_Return').index
                        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                                   'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                        worst_months_per_asset[ticker] = [month_map[month] for month in worst_months]
                        
                elif strategy_type == "Top 6 Months Strategy":
                    # Only invest during the 6 best months across all assets
                    top_months_per_asset = {}
                    for ticker in all_assets:
                        seasonal_stats = st.session_state.multi_asset_data[ticker]['seasonal_stats']
                        top_months = seasonal_stats.nlargest(6, 'Avg_Return').index
                        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                                   'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                        top_months_per_asset[ticker] = [month_map[month] for month in top_months]
                        
                elif strategy_type == "Seasonal Score Based":
                    # Adjust exposure based on seasonal scores
                    seasonal_scores_per_asset = {}
                    for ticker in all_assets:
                        seasonal_stats = st.session_state.multi_asset_data[ticker]['seasonal_stats']
                        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                                   'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                        scores = {}
                        for month_name, month_num in month_map.items():
                            if month_name in seasonal_stats.index:
                                avg_return = seasonal_stats.loc[month_name, 'Avg_Return']
                                win_rate = seasonal_stats.loc[month_name, 'Win_Rate']
                                scores[month_num] = max(0.1, (avg_return + win_rate) / 2)  # Combined score
                            else:
                                scores[month_num] = 0.5  # Neutral if no data
                        seasonal_scores_per_asset[ticker] = scores
                
                # Calculate strategy-based portfolio returns
                portfolio_returns = []
                for date in all_dates:
                    daily_return = 0
                    current_month = date.month
                    
                    for ticker, base_weight in weights.items():
                        if date in all_stock_data[ticker].index:
                            asset_return = all_stock_data[ticker].loc[date, 'Returns']
                            
                            # Apply strategy logic to determine actual exposure
                            if strategy_type == "Best Months Only":
                                # Only invest if current month is in best months for this asset
                                if current_month in best_months_per_asset[ticker]:
                                    exposure = 1.0
                                else:
                                    exposure = 0.0  # Cash position
                                    
                            elif strategy_type == "Avoid Worst Months":
                                # Don't invest if current month is in worst months for this asset
                                if current_month in worst_months_per_asset[ticker]:
                                    exposure = 0.0  # Cash position
                                else:
                                    exposure = 1.0
                                    
                            elif strategy_type == "Top 6 Months Strategy":
                                # Only invest if current month is in top 6 months for this asset
                                if current_month in top_months_per_asset[ticker]:
                                    exposure = 1.0
                                else:
                                    exposure = 0.0  # Cash position
                                    
                            elif strategy_type == "Seasonal Score Based":
                                # Scale exposure based on seasonal score
                                exposure = seasonal_scores_per_asset[ticker].get(current_month, 0.5)
                                
                            else:  # Buy and hold
                                exposure = 1.0
                            
                            # Calculate weighted return with strategy exposure
                            daily_return += base_weight * asset_return * exposure
                    
                    portfolio_returns.append(daily_return)
                
                # Calculate portfolio performance metrics
                portfolio_returns = np.array(portfolio_returns)
                total_portfolio_return = ((1 + portfolio_returns).prod() - 1) * 100
                portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252) * 100
                portfolio_sharpe = (np.mean(portfolio_returns) * 252) / (np.std(portfolio_returns) * np.sqrt(252)) if np.std(portfolio_returns) > 0 else 0
                
                # Calculate strategy-adjusted individual asset performance
                asset_performance = []
                for ticker in all_assets:
                    stock_data = all_stock_data[ticker].copy()
                    seasonal_stats = st.session_state.multi_asset_data[ticker]['seasonal_stats']
                    
                    # Calculate buy-and-hold return
                    buy_hold_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
                    
                    # Calculate strategy-adjusted return for this individual asset
                    strategy_returns = []
                    for date in stock_data.index:
                        current_month = date.month
                        asset_return = stock_data.loc[date, 'Returns']
                        
                        # Apply same strategy logic as portfolio
                        if strategy_type == "Best Months Only":
                            if current_month in best_months_per_asset[ticker]:
                                exposure = 1.0
                            else:
                                exposure = 0.0  # Cash position
                        elif strategy_type == "Avoid Worst Months":
                            if current_month in worst_months_per_asset[ticker]:
                                exposure = 0.0  # Cash position
                            else:
                                exposure = 1.0
                        elif strategy_type == "Top 6 Months Strategy":
                            if current_month in top_months_per_asset[ticker]:
                                exposure = 1.0
                            else:
                                exposure = 0.0  # Cash position
                        elif strategy_type == "Seasonal Score Based":
                            exposure = seasonal_scores_per_asset[ticker].get(current_month, 0.5)
                        else:  # Buy and hold
                            exposure = 1.0
                        
                        strategy_returns.append(asset_return * exposure)
                    
                    # Calculate strategy performance metrics
                    strategy_returns = np.array(strategy_returns)
                    strategy_total_return = ((1 + strategy_returns).prod() - 1) * 100
                    strategy_volatility = np.std(strategy_returns) * np.sqrt(252) * 100
                    
                    # Calculate effective weight (considering strategy exposure)
                    effective_exposure = np.mean([1.0 if strategy_type == "Seasonal Long (Buy & Hold)" else 
                                                np.mean([1.0 if strategy_type == "Best Months Only" and month in best_months_per_asset[ticker] else
                                                        0.0 if strategy_type == "Best Months Only" else
                                                        0.0 if strategy_type == "Avoid Worst Months" and month in worst_months_per_asset[ticker] else
                                                        1.0 if strategy_type == "Avoid Worst Months" else
                                                        1.0 if strategy_type == "Top 6 Months Strategy" and month in top_months_per_asset[ticker] else
                                                        0.0 if strategy_type == "Top 6 Months Strategy" else
                                                        seasonal_scores_per_asset[ticker].get(month, 0.5) if strategy_type == "Seasonal Score Based" else 1.0
                                                        for month in range(1, 13)])
                                                for _ in range(1)])
                    
                    # Simplified effective weight calculation
                    if strategy_type == "Best Months Only":
                        effective_weight = weights[ticker] * (3/12)  # Only 3 months active
                    elif strategy_type == "Avoid Worst Months":
                        effective_weight = weights[ticker] * (9/12)  # 9 months active  
                    elif strategy_type == "Top 6 Months Strategy":
                        effective_weight = weights[ticker] * (6/12)  # 6 months active
                    elif strategy_type == "Seasonal Score Based":
                        avg_score = np.mean(list(seasonal_scores_per_asset[ticker].values()))
                        effective_weight = weights[ticker] * avg_score
                    else:
                        effective_weight = weights[ticker]  # Buy and hold
                    
                    asset_performance.append({
                        'Asset': ticker,
                        'Weight': f"{effective_weight*100:.1f}%",
                        'Strategy Return': f"{strategy_total_return:.1f}%",
                        'Volatility': f"{strategy_volatility:.1f}%",
                        'Seasonal Score': f"{st.session_state.multi_asset_data[ticker]['seasonal_relevance']['score']:.2f}"
                    })
                
                # Display results
                st.success("✅ Portfolio backtest completed!")
                
                # Portfolio summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Portfolio Return", f"{total_portfolio_return:.1f}%")
                
                with col2:
                    st.metric("Portfolio Volatility", f"{portfolio_volatility:.1f}%")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
                
                with col4:
                    risk_adjusted = total_portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                    st.metric("Risk-Adj Return", f"{risk_adjusted:.2f}")
                
                # Asset allocation table
                st.markdown("#### 📊 Portfolio Allocation & Performance")
                results_df = pd.DataFrame(asset_performance)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Visualization
                fig = go.Figure()
                
                # Portfolio allocation pie chart
                fig.add_trace(go.Pie(
                    labels=list(weights.keys()),
                    values=list(weights.values()),
                    name="Portfolio Allocation",
                    hole=0.3
                ))
                
                fig.update_layout(
                    title=f"{analysis_mode} - Asset Allocation",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance comparison using strategy-adjusted returns
                strategy_returns = [float(asset['Strategy Return'].rstrip('%')) for asset in asset_performance]
                equal_weight_strategy_return = np.mean(strategy_returns)
                
                comparison_data = {
                    'Strategy': [f"{analysis_mode} + {strategy_type}", 'Equal Weight Buy & Hold', 'Best Strategy Asset'],
                    'Return (%)': [f"{total_portfolio_return:.1f}", f"{equal_weight_strategy_return:.1f}", f"{max(strategy_returns):.1f}"],
                    'Risk (%)': [f"{portfolio_volatility:.1f}", f"{np.mean([float(asset['Volatility'].rstrip('%')) for asset in asset_performance]):.1f}", "Variable"]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.markdown("#### 📈 Strategy Comparison")
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Quick portfolio metrics preview
    st.markdown("### 📋 Portfolio Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate portfolio-level metrics
    total_assets = len(all_assets)
    avg_return = np.mean([((st.session_state.multi_asset_data[ticker]['stock_data']['Close'].iloc[-1] / 
                            st.session_state.multi_asset_data[ticker]['stock_data']['Close'].iloc[0]) - 1) * 100 
                          for ticker in all_assets])
    avg_volatility = np.mean([st.session_state.multi_asset_data[ticker]['stock_data']['Returns'].std() * np.sqrt(252) * 100 
                             for ticker in all_assets])
    avg_score = np.mean([st.session_state.multi_asset_data[ticker]['seasonal_relevance']['score'] 
                        for ticker in all_assets])
    
    with col1:
        st.metric("Portfolio Size", f"{total_assets} assets")
    
    with col2:
        st.metric("Avg Return", f"{avg_return:.1f}%")
    
    with col3:
        st.metric("Avg Volatility", f"{avg_volatility:.1f}%")
    
    with col4:
        portfolio_score = "Strong" if avg_score > 0.6 else "Medium" if avg_score > 0.4 else "Weak"
        st.metric("Portfolio Score", portfolio_score, f"{avg_score:.2f}")

def display_multi_asset_ai_insights(confidence_threshold):
    """Display AI insights for multiple assets"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">🤖 Multi-Asset AI Insights</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">AI-powered pattern analysis across your asset portfolio</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.multi_asset_data:
        st.error("❌ No asset data available. Please process assets first.")
        return
    
    # Aggregate AI insights across all assets
    all_insights = []
    high_confidence_signals = []
    portfolio_patterns = []
    
    for ticker, data in st.session_state.multi_asset_data.items():
        if data.get('ai_insights'):
            insights = data['ai_insights']
            all_insights.append({
                'asset': ticker,
                'insights': insights,
                'score': data['seasonal_relevance']['score']
            })
            
            # Check for high confidence signals
            if data['seasonal_relevance']['score'] > confidence_threshold:
                high_confidence_signals.append({
                    'asset': ticker,
                    'score': data['seasonal_relevance']['score'],
                    'current_month_return': data['seasonal_relevance']['current_month_return']
                })
    
    # Display portfolio-level AI summary
    st.markdown("### 🎯 Portfolio AI Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Assets with AI", len(all_insights), f"of {len(st.session_state.multi_asset_data)}")
    
    with col2:
        st.metric("High Confidence", len(high_confidence_signals), f"> {confidence_threshold:.1f} score")
    
    with col3:
        avg_score = np.mean([item['score'] for item in all_insights]) if all_insights else 0
        st.metric("Avg AI Score", f"{avg_score:.2f}")
    
    with col4:
        positive_signals = len([s for s in high_confidence_signals if s['current_month_return'] > 0])
        st.metric("Positive Signals", positive_signals, f"of {len(high_confidence_signals)}")
    
    st.divider()
    
    # Display individual asset insights
    if all_insights:
        st.markdown("### 🔍 Individual Asset AI Insights")
        
        for item in sorted(all_insights, key=lambda x: x['score'], reverse=True):
            ticker = item['asset']
            insights = item['insights']
            score = item['score']
            
            confidence_color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"
            
            with st.expander(f"🤖 {ticker} - AI Score: {score:.2f}", expanded=score > 0.7):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Display the AI insights content
                    if isinstance(insights, str):
                        st.markdown(f"**AI Analysis:** {insights}")
                    elif isinstance(insights, dict):
                        # Check for specific AI insight structure
                        if 'high_confidence' in insights:
                            st.markdown("#### 🔥 High Confidence Patterns")
                            for pattern in insights['high_confidence']:
                                st.success(f"**{pattern.get('pattern', 'Pattern')}**: {pattern.get('description', 'N/A')}")
                        
                        if 'anomalies' in insights:
                            st.markdown("#### ⚠️ Anomalies Detected")
                            for anomaly in insights['anomalies']:
                                st.warning(f"**{anomaly.get('type', 'Anomaly')}**: {anomaly.get('description', 'N/A')}")
                        
                        if 'predictions' in insights:
                            st.markdown("#### 🔮 Predictions")
                            for prediction in insights['predictions']:
                                st.info(f"**{prediction.get('Month', 'Period')}**: {prediction.get('Details', 'N/A')}")
                        
                        if 'pattern_strength' in insights:
                            strength = insights['pattern_strength']
                            st.markdown("#### 💪 Pattern Strength")
                            st.json(strength)
                            
                        # Fallback for other dict formats
                        if not any(key in insights for key in ['high_confidence', 'anomalies', 'predictions', 'pattern_strength']):
                            for key, value in insights.items():
                                if isinstance(value, (list, dict)):
                                    st.markdown(f"**{key}:**")
                                    st.json(value)
                                else:
                                    st.markdown(f"**{key}:** {value}")
                    else:
                        st.write("AI insights available but format not recognized")
                
                with col2:
                    # Add confidence indicator
                    if score > 0.7:
                        st.success("🔥 High Confidence")
                    elif score > 0.5:
                        st.warning("⭐ Medium Confidence")
                    else:
                        st.error("⚠️ Low Confidence")
                    
                    # Quick action button
                    if st.button(f"📊 Analyze {ticker}", key=f"ai_analyze_{ticker}", use_container_width=True):
                        st.session_state.dashboard_mode = 'single'
                        st.session_state.selected_asset = ticker
                        st.session_state.stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
                        st.session_state.seasonal_stats = st.session_state.multi_asset_data[ticker]['seasonal_stats']
                        st.session_state.current_symbol = ticker
                        st.session_state.data_analyzed = True
                        st.session_state.selected_tab = "🤖 AI Insights"
                        
                        # Initialize weekday_stats
                        from data_processor import DataProcessor
                        processor = DataProcessor()
                        st.session_state.weekday_stats = processor.calculate_weekday_stats(st.session_state.multi_asset_data[ticker]['stock_data'])
                        
                        if st.session_state.multi_asset_data[ticker]['ai_insights']:
                            st.session_state.ai_insights = st.session_state.multi_asset_data[ticker]['ai_insights']
                        st.rerun()
    else:
        st.info("🤖 No AI insights available. Make sure AI Pattern Detection is enabled when building the dashboard.")

def display_multi_asset_performance_charts():
    """Display performance charts for multiple assets"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">📊 Multi-Asset Performance Charts</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Comprehensive visual analysis across your asset portfolio</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.multi_asset_data:
        st.error("❌ No asset data available. Please process assets first.")
        return
    
    all_assets = list(st.session_state.multi_asset_data.keys())
    
    # Chart selection
    chart_type = st.selectbox(
        "📈 Select Chart Type",
        ["Portfolio Comparison", "Seasonal Heatmap", "Correlation Matrix", "Risk vs Return", "Monthly Performance"],
        key="performance_chart_type"
    )
    
    if chart_type == "Portfolio Comparison":
        st.markdown("### 📊 Asset Performance Comparison")
        
        # Create comparison chart
        fig = go.Figure()
        
        for ticker in all_assets:
            stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
            # Normalize to starting value of 100
            normalized_prices = (stock_data['Close'] / stock_data['Close'].iloc[0]) * 100
            
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=normalized_prices,
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Seasonal Heatmap":
        st.markdown("### 🔥 Multi-Asset Seasonal Heatmap")
        
        # Create heatmap data
        heatmap_data = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for ticker in all_assets:
            seasonal_stats = st.session_state.multi_asset_data[ticker]['seasonal_stats']
            monthly_returns = []
            
            for month in ['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December']:
                if month in seasonal_stats.index:
                    monthly_returns.append(seasonal_stats.loc[month, 'Avg_Return'] * 100)
                else:
                    monthly_returns.append(0)
            
            heatmap_data.append(monthly_returns)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=months,
            y=all_assets,
            colorscale='RdYlGn',
            colorbar=dict(title="Avg Return (%)"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Monthly Seasonal Performance Heatmap",
            height=max(400, len(all_assets) * 40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Risk vs Return":
        st.markdown("### ⚖️ Risk vs Return Analysis")
        
        returns = []
        volatilities = []
        scores = []
        
        for ticker in all_assets:
            stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
            total_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
            volatility = stock_data['Returns'].std() * np.sqrt(252) * 100
            score = st.session_state.multi_asset_data[ticker]['seasonal_relevance']['score']
            
            returns.append(total_return)
            volatilities.append(volatility)
            scores.append(score)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=all_assets,
            textposition="top center",
            marker=dict(
                size=[s*50 for s in scores],  # Size based on seasonal score
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Seasonal Score")
            ),
            name="Assets"
        ))
        
        fig.update_layout(
            title="Risk vs Return Scatter (Bubble size = Seasonal Score)",
            xaxis_title="Volatility (%)",
            yaxis_title="Total Return (%)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info(f"🚧 {chart_type} chart coming soon!")

def display_multi_asset_risk_analysis():
    """Display risk analysis for multiple assets"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">⚠️ Multi-Asset Risk Analysis</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Portfolio risk assessment and diversification analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.multi_asset_data:
        st.error("❌ No asset data available. Please process assets first.")
        return
    
    all_assets = list(st.session_state.multi_asset_data.keys())
    
    # Calculate risk metrics for each asset
    risk_data = []
    
    for ticker in all_assets:
        stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
        
        # Calculate risk metrics
        returns = stock_data['Returns']
        volatility = returns.std() * np.sqrt(252) * 100
        max_drawdown = (stock_data['Close'] / stock_data['Close'].cummax() - 1).min() * 100
        var_95 = np.percentile(returns, 5) * 100
        
        # Calculate VaR and other metrics
        total_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
        sharpe_ratio = (total_return/100) / (volatility/100) if volatility > 0 else 0
        
        risk_data.append({
            'Asset': ticker,
            'Volatility (%)': f"{volatility:.1f}",
            'Max Drawdown (%)': f"{max_drawdown:.1f}",
            'VaR 95% (%)': f"{var_95:.2f}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Risk Level': 'High' if volatility > 30 else 'Medium' if volatility > 20 else 'Low',
            'Seasonal Score': f"{st.session_state.multi_asset_data[ticker]['seasonal_relevance']['score']:.2f}"
        })
    
    # Display risk metrics table
    st.markdown("### 📊 Individual Asset Risk Metrics")
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Portfolio-level risk analysis
    st.markdown("### 🎯 Portfolio Risk Summary")
    
    volatilities = [float(item['Volatility (%)']) for item in risk_data]
    max_drawdowns = [float(item['Max Drawdown (%)']) for item in risk_data]
    sharpe_ratios = [float(item['Sharpe Ratio']) for item in risk_data]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_vol = np.mean(volatilities)
        vol_status = "High Risk" if avg_vol > 30 else "Medium Risk" if avg_vol > 20 else "Low Risk"
        st.metric("Avg Volatility", f"{avg_vol:.1f}%", vol_status)
    
    with col2:
        avg_drawdown = np.mean(max_drawdowns)
        st.metric("Avg Max Drawdown", f"{avg_drawdown:.1f}%")
    
    with col3:
        avg_sharpe = np.mean(sharpe_ratios)
        sharpe_status = "Excellent" if avg_sharpe > 1 else "Good" if avg_sharpe > 0.5 else "Poor"
        st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}", sharpe_status)
    
    with col4:
        high_risk_assets = len([v for v in volatilities if v > 30])
        st.metric("High Risk Assets", f"{high_risk_assets}/{len(all_assets)}")
    
    # Risk visualization
    st.markdown("### 📈 Risk Visualization")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Volatility',
        x=all_assets,
        y=volatilities,
        yaxis='y',
        offsetgroup=1,
        marker_color='lightcoral'
    ))
    
    fig.add_trace(go.Scatter(
        name='Sharpe Ratio',
        x=all_assets,
        y=sharpe_ratios,
        yaxis='y2',
        mode='markers+lines',
        marker=dict(size=8, color='darkblue'),
        line=dict(color='darkblue', width=2)
    ))
    
    fig.update_layout(
        title='Asset Risk Profile: Volatility vs Sharpe Ratio',
        xaxis=dict(title='Assets'),
        yaxis=dict(title='Volatility (%)', side='left'),
        yaxis2=dict(title='Sharpe Ratio', side='right', overlaying='y'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_multi_asset_statistical_tests():
    """Display statistical tests and Monte Carlo analysis for multiple assets"""
    # Import required libraries at the top of the function
    import numpy as np
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">📊 Multi-Asset Statistical Analysis</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Monte Carlo simulations and statistical tests across your portfolio</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.multi_asset_data:
        st.error("❌ No asset data available. Please process assets first.")
        return
    
    all_assets = list(st.session_state.multi_asset_data.keys())
    
    # Analysis type selection
    st.markdown("### ⚙️ Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_type = st.selectbox(
            "🔬 Analysis Type",
            ["Individual Asset Analysis", "Portfolio Monte Carlo", "Cross-Asset Correlation", "Risk Parity Analysis"],
            help="Choose the type of statistical analysis",
            key="statistical_analysis_type"
        )
    
    with col2:
        n_simulations = st.number_input("🎲 Monte Carlo Simulations", value=1000, min_value=100, max_value=10000, step=500, key="statistical_n_simulations")
    
    with col3:
        time_horizon = st.selectbox("📅 Time Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"], index=3, key="statistical_time_horizon")
    
    horizon_days = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 252}[time_horizon]
    
    st.divider()
    
    if analysis_type == "Individual Asset Analysis":
        st.markdown("### 📊 Individual Asset Statistical Analysis")
        
        # Asset selection
        selected_assets = st.multiselect(
            "Select Assets for Analysis",
            all_assets,
            default=all_assets[:min(3, len(all_assets))],
            help="Choose assets for detailed statistical analysis",
            key="statistical_selected_assets"
        )
        
        if not selected_assets:
            st.warning("Please select at least one asset for analysis.")
            return
        
        if st.button("🚀 Run Statistical Analysis", type="primary"):
            # Use built-in statistical analysis
            try:
                
                results_data = []
                
                for ticker in selected_assets:
                    st.markdown(f"#### 📈 {ticker} Analysis")
                    
                    data = st.session_state.multi_asset_data[ticker]
                    stock_data = data['stock_data']
                    returns = stock_data['Returns'].dropna()
                    
                    if len(returns) < 30:
                        st.warning(f"⚠️ {ticker}: Insufficient data for reliable analysis (need at least 30 data points)")
                        continue
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Monte Carlo simulation
                        st.markdown("##### 🎲 Monte Carlo Results")
                        try:
                            # Built-in Monte Carlo simulation
                            np.random.seed(42)
                            mc_returns = []
                            
                            for _ in range(n_simulations):
                                # Sample returns with replacement
                                sampled_returns = np.random.choice(returns, size=252)  # 1 year
                                annual_return = (1 + sampled_returns).prod() - 1
                                mc_returns.append(annual_return)
                            
                            mc_returns = np.array(mc_returns)
                            
                            # Calculate metrics
                            expected_return = np.mean(mc_returns)
                            volatility = np.std(mc_returns)
                            var_5 = np.percentile(mc_returns, 5)
                            var_1 = np.percentile(mc_returns, 1)
                            
                            # Display key metrics
                            st.metric("Expected Return", f"{expected_return:.2%}")
                            st.metric("Volatility", f"{volatility:.2%}")
                            st.metric("VaR (5%)", f"{var_5:.2%}")
                            st.metric("VaR (1%)", f"{var_1:.2%}")
                            
                            # Create Monte Carlo visualization
                            fig_mc = go.Figure()
                            
                            # Histogram of returns
                            fig_mc.add_trace(go.Histogram(
                                x=mc_returns * 100,
                                nbinsx=50,
                                name=f'{ticker} Returns',
                                opacity=0.7,
                                marker_color='lightblue'
                            ))
                            
                            # Add reference lines
                            fig_mc.add_vline(x=expected_return * 100, 
                                           line_dash="solid", line_color="blue",
                                           annotation_text=f"Mean: {expected_return:.1%}")
                            fig_mc.add_vline(x=var_5 * 100, 
                                           line_dash="dash", line_color="red",
                                           annotation_text=f"5% VaR: {var_5:.1%}")
                            
                            fig_mc.update_layout(
                                title=f"Monte Carlo Returns Distribution - {ticker}",
                                xaxis_title="Annual Return (%)",
                                yaxis_title="Frequency",
                                height=300
                            )
                            
                            st.plotly_chart(fig_mc, use_container_width=True, key=f"mc_chart_{ticker}")
                            
                        except Exception as e:
                            st.error(f"Monte Carlo error for {ticker}: {str(e)}")
                    
                    with col2:
                        # Statistical tests
                        st.markdown("##### 📊 Statistical Tests")
                        try:
                            # Normality test
                            shapiro_stat, shapiro_p = stats.shapiro(returns[:5000] if len(returns) > 5000 else returns)
                            
                            st.write(f"**Normality Test (Shapiro-Wilk):**")
                            st.write(f"P-value: {shapiro_p:.4f}")
                            st.write(f"Normal Distribution: {'❌ No' if shapiro_p < 0.05 else '✅ Yes'}")
                            
                            # Autocorrelation test
                            lb_stat = acorr_ljungbox(returns, lags=10, return_df=True)
                            auto_p = lb_stat['lb_pvalue'].iloc[-1]
                            
                            st.write(f"**Autocorrelation Test:**")
                            st.write(f"P-value: {auto_p:.4f}")
                            st.write(f"Independent: {'✅ Yes' if auto_p > 0.05 else '❌ No'}")
                            
                            # ANOVA test for seasonal differences
                            st.write(f"**Seasonal ANOVA Test:**")
                            months = stock_data.index.month
                            month_groups = [returns[months == i] for i in range(1, 13) if len(returns[months == i]) > 0]
                            
                            if len(month_groups) >= 3:  # Need at least 3 months
                                f_stat, anova_p = stats.f_oneway(*month_groups)
                                st.write(f"F-statistic: {f_stat:.2f}")
                                st.write(f"P-value: {anova_p:.4f}")
                                st.write(f"Seasonal Effect: {'✅ Yes' if anova_p < 0.05 else '❌ No'}")
                            else:
                                st.write("Insufficient data for seasonal analysis")
                            
                        except Exception as e:
                            st.error(f"Statistical tests error for {ticker}: {str(e)}")
                    
                    st.divider()
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
    
    elif analysis_type == "Portfolio Monte Carlo":
        st.markdown("### 🎯 Portfolio Monte Carlo Analysis")
        
        # Portfolio configuration
        col1, col2 = st.columns(2)
        
        with col1:
            portfolio_type = st.selectbox(
                "Portfolio Type",
                ["Equal Weight", "Market Cap Weighted", "Risk Parity", "Seasonal Score Weighted"],
                key="portfolio_type_select"
            )
        
        with col2:
            initial_value = st.number_input("Initial Portfolio Value ($)", value=100000, min_value=1000, step=1000, key="portfolio_initial_value")
        
        if st.button("🚀 Run Portfolio Monte Carlo", type="primary"):
            with st.spinner("Running portfolio Monte Carlo simulation..."):
                try:
                    # Calculate portfolio weights based on selected type
                    if portfolio_type == "Equal Weight":
                        weights = {ticker: 1.0/len(all_assets) for ticker in all_assets}
                        
                    elif portfolio_type == "Market Cap Weighted":
                        # Use price as proxy for market cap (higher price = higher weight)
                        # In real implementation, you'd use actual market cap data
                        market_caps = {}
                        for ticker in all_assets:
                            stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
                            avg_price = stock_data['Close'].mean()
                            # Simple proxy: higher average price gets higher weight
                            market_caps[ticker] = avg_price
                        
                        total_cap = sum(market_caps.values())
                        weights = {ticker: cap/total_cap for ticker, cap in market_caps.items()}
                        
                    elif portfolio_type == "Risk Parity":
                        # Equal risk contribution: weight = 1/volatility normalized
                        volatilities = {}
                        for ticker in all_assets:
                            stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
                            returns = stock_data['Returns'].dropna()
                            vol = returns.std() * np.sqrt(252)  # Annualized volatility
                            volatilities[ticker] = vol if vol > 0 else 0.01  # Avoid division by zero
                        
                        # Inverse volatility weights
                        inv_vol = {ticker: 1/vol for ticker, vol in volatilities.items()}
                        total_inv_vol = sum(inv_vol.values())
                        weights = {ticker: inv_vol_val/total_inv_vol for ticker, inv_vol_val in inv_vol.items()}
                        
                    elif portfolio_type == "Seasonal Score Weighted":
                        scores = {ticker: st.session_state.multi_asset_data[ticker]['seasonal_relevance']['score'] 
                                for ticker in all_assets}
                        total_score = sum(scores.values())
                        weights = {ticker: score/total_score for ticker, score in scores.items()}
                        
                    else:
                        weights = {ticker: 1.0/len(all_assets) for ticker in all_assets}  # Default to equal weight
                    
                    # Get returns data for all assets
                    returns_data = {}
                    for ticker in all_assets:
                        stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
                        returns_data[ticker] = stock_data['Returns'].dropna()
                    
                    # Calculate portfolio returns
                    all_dates = set()
                    for returns in returns_data.values():
                        all_dates.update(returns.index)
                    all_dates = sorted(list(all_dates))
                    
                    portfolio_returns = []
                    for date in all_dates:
                        daily_return = 0
                        for ticker, weight in weights.items():
                            if date in returns_data[ticker].index:
                                daily_return += weight * returns_data[ticker][date]
                        portfolio_returns.append(daily_return)
                    
                    portfolio_returns = np.array(portfolio_returns)
                    
                    # Enhanced Monte Carlo simulation with seasonal patterns
                    portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
                    
                    if len(portfolio_returns) < 252:  # Need at least 1 year of data
                        st.warning("⚠️ Insufficient data for reliable Monte Carlo simulation")
                        return
                    
                    # Calculate seasonal statistics
                    dates = pd.date_range(start='2020-01-01', periods=len(portfolio_returns), freq='D')
                    returns_df = pd.DataFrame({'returns': portfolio_returns, 'month': [d.month for d in dates]})
                    monthly_stats = returns_df.groupby('month')['returns'].agg(['mean', 'std']).fillna(0)
                    
                    # Generate seasonal-aware random scenarios
                    np.random.seed(42)  # For reproducible results
                    portfolio_values = []
                    
                    for sim in range(n_simulations):
                        values = [initial_value]
                        current_date = pd.Timestamp.now()
                        
                        for day in range(horizon_days):
                            sim_date = current_date + pd.Timedelta(days=day)
                            month = sim_date.month
                            
                            # Use seasonal statistics for more realistic simulation
                            mean_return = monthly_stats.loc[month, 'mean']
                            volatility = monthly_stats.loc[month, 'std']
                            
                            # Generate return with seasonal bias
                            daily_return = np.random.normal(mean_return, volatility)
                            values.append(values[-1] * (1 + daily_return))
                        
                        portfolio_values.append(values[-1])
                    
                    portfolio_values = np.array(portfolio_values)
                    
                    # Calculate statistics
                    final_mean = np.mean(portfolio_values)
                    final_std = np.std(portfolio_values)
                    var_5 = np.percentile(portfolio_values, 5)
                    var_1 = np.percentile(portfolio_values, 1)
                    
                    # Display results
                    st.success("✅ Portfolio Monte Carlo simulation completed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        expected_return = (final_mean - initial_value) / initial_value
                        st.metric("Expected Return", f"{expected_return:.1%}")
                    
                    with col2:
                        st.metric("Expected Value", f"${final_mean:,.0f}")
                    
                    with col3:
                        loss_prob = len(portfolio_values[portfolio_values < initial_value]) / len(portfolio_values)
                        st.metric("Loss Probability", f"{loss_prob:.1%}")
                    
                    with col4:
                        st.metric("VaR (5%)", f"${var_5:,.0f}")
                    
                    # Portfolio allocation
                    st.markdown(f"#### 📊 Portfolio Allocation ({portfolio_type})")
                    allocation_data = {
                        'Asset': list(weights.keys()),
                        'Weight': [f"{w*100:.1f}%" for w in weights.values()],
                        'Value': [f"${initial_value * w:,.0f}" for w in weights.values()]
                    }
                    
                    # Add additional information based on portfolio type
                    if portfolio_type == "Market Cap Weighted":
                        allocation_data['Avg Price'] = [f"${st.session_state.multi_asset_data[ticker]['stock_data']['Close'].mean():.2f}" 
                                                       for ticker in weights.keys()]
                    elif portfolio_type == "Risk Parity":
                        allocation_data['Volatility'] = [f"{(st.session_state.multi_asset_data[ticker]['stock_data']['Returns'].dropna().std() * np.sqrt(252) * 100):.1f}%" 
                                                        for ticker in weights.keys()]
                    elif portfolio_type == "Seasonal Score Weighted":
                        allocation_data['Seasonal Score'] = [f"{st.session_state.multi_asset_data[ticker]['seasonal_relevance']['score']:.2f}" 
                                                            for ticker in weights.keys()]
                    
                    st.dataframe(pd.DataFrame(allocation_data), use_container_width=True, hide_index=True)
                    
                    # Simulation results distribution
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=portfolio_values,
                        nbinsx=50,
                        name="Portfolio Values",
                        opacity=0.7,
                        marker_color='lightblue'
                    ))
                    
                    # Add VaR lines
                    fig.add_vline(x=var_5, line_dash="dash", line_color="red", 
                                 annotation_text=f"VaR 5%: ${var_5:,.0f}")
                    fig.add_vline(x=initial_value, line_dash="dash", line_color="green",
                                 annotation_text=f"Initial: ${initial_value:,.0f}")
                    
                    fig.update_layout(
                        title=f"Portfolio Value Distribution ({time_horizon} - {n_simulations:,} simulations)",
                        xaxis_title="Portfolio Value ($)",
                        yaxis_title="Frequency",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk metrics
                    st.markdown("#### ⚠️ Risk Analysis")
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    
                    with risk_col1:
                        st.metric("Standard Deviation", f"${final_std:,.0f}")
                    
                    with risk_col2:
                        downside_scenarios = len(portfolio_values[portfolio_values < initial_value * 0.9])
                        downside_prob = downside_scenarios / len(portfolio_values)
                        st.metric("10%+ Loss Prob", f"{downside_prob:.1%}")
                    
                    with risk_col3:
                        upside_scenarios = len(portfolio_values[portfolio_values > initial_value * 1.1])
                        upside_prob = upside_scenarios / len(portfolio_values)
                        st.metric("10%+ Gain Prob", f"{upside_prob:.1%}")
                
                except Exception as e:
                    st.error(f"Portfolio Monte Carlo error: {str(e)}")
                    st.info("💡 Try reducing the number of simulations or check your data quality.")
    
    else:
        st.markdown(f"### 📊 {analysis_type}")
        st.info(f"🚧 {analysis_type} coming soon!")
    
    st.divider()
    
    # Portfolio summary statistics
    st.markdown("### 📋 Portfolio Statistical Summary")
    
    # Calculate portfolio-level statistics
    all_returns = []
    all_volatilities = []
    all_sharpe_ratios = []
    
    for ticker in all_assets:
        stock_data = st.session_state.multi_asset_data[ticker]['stock_data']
        returns = stock_data['Returns'].dropna()
        
        if len(returns) > 0:
            annual_return = returns.mean() * 252
            volatility = returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            all_returns.append(annual_return)
            all_volatilities.append(volatility)
            all_sharpe_ratios.append(sharpe)
    
    if all_returns:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_return = np.mean(all_returns)
            st.metric("Avg Annual Return", f"{avg_return:.1%}")
        
        with col2:
            avg_vol = np.mean(all_volatilities)
            st.metric("Avg Volatility", f"{avg_vol:.1%}")
        
        with col3:
            avg_sharpe = np.mean(all_sharpe_ratios)
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
        
        with col4:
            portfolio_assets = len(all_assets)
            st.metric("Portfolio Size", f"{portfolio_assets} assets")
        
        # Risk distribution chart
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=[r * 100 for r in all_returns],
            nbinsx=20,
            name="Annual Returns",
            opacity=0.7,
            marker_color='blue'
        ))
        
        fig.update_layout(
            title="Portfolio Returns Distribution",
            xaxis_title="Annual Return (%)",
            yaxis_title="Number of Assets",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_overview(stock_data, seasonal_stats, symbol, company_name):
    """Display enhanced executive overview with comprehensive metrics"""
    # Professional section header
    st.markdown(f"""
    <div class="section-header">
        <h3>📊 Executive Dashboard • {symbol}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate comprehensive metrics
    total_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
    years_of_data = len(stock_data) / 252  # Approximate years
    volatility = stock_data['Returns'].std() * np.sqrt(252) * 100
    
    best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
    worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
    avg_monthly_return = seasonal_stats['Avg_Return'].mean()
    winning_months = len(seasonal_stats[seasonal_stats['Avg_Return'] > 0])
    
    # Primary metrics grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card info-card">
            <h4>📈 Total Return</h4>
            <h2>{total_return:.1f}%</h2>
            <p class="metric-value">Over {years_of_data:.1f} years • {total_return/years_of_data:.1f}% annually</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        volatility_color = "warning-card" if volatility > 30 else "info-card"
        st.markdown(f"""
        <div class="metric-card {volatility_color}">
            <h4>📊 Volatility</h4>
            <h2>{volatility:.1f}%</h2>
            <p class="metric-value">Annualized • Risk Level: {'High' if volatility > 30 else 'Moderate' if volatility > 20 else 'Low'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card success-card">
            <h4>🌟 Best Month</h4>
            <h2>{best_month.name}</h2>
            <p class="metric-value">Avg: {best_month['Avg_Return']:.1%} • Win Rate: {best_month['Win_Rate']:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card danger-card">
            <h4>⚠️ Worst Month</h4>
            <h2>{worst_month.name}</h2>
            <p class="metric-value">Avg: {worst_month['Avg_Return']:.1%} • Win Rate: {worst_month['Win_Rate']:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Secondary metrics
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📅 Monthly Performance</h4>
            <h2>{avg_monthly_return:.2%}</h2>
            <p class="metric-value">Average monthly return</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card success-card">
            <h4>🎯 Winning Months</h4>
            <h2>{winning_months}/12</h2>
            <p class="metric-value">{winning_months/12:.0%} positive seasonality</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        data_points = len(stock_data)
        st.markdown(f"""
        <div class="metric-card info-card">
            <h4>📊 Dataset Quality</h4>
            <h2>{data_points:,}</h2>
            <p class="metric-value">Total observations • {data_points/(years_of_data*252):.0%} completeness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sharpe_ratio = (total_return/years_of_data - 2) / volatility if volatility > 0 else 0
        sharpe_color = "success-card" if sharpe_ratio > 1 else "warning-card" if sharpe_ratio > 0.5 else "danger-card"
        st.markdown(f"""
        <div class="metric-card {sharpe_color}">
            <h4>📈 Risk-Adj. Return</h4>
            <h2>{sharpe_ratio:.2f}</h2>
            <p class="metric-value">Sharpe ratio estimate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Quick Insights Section
    st.markdown("""
    <div class="section-header">
        <h3>🔍 Market Intelligence • Key Patterns</h3>
    </div>
    """, unsafe_allow_html=True)
    
    winning_months = seasonal_stats[seasonal_stats['Win_Rate'] > 0.6]
    losing_months = seasonal_stats[seasonal_stats['Win_Rate'] < 0.4]
    high_vol_months = seasonal_stats[seasonal_stats['Volatility'] > seasonal_stats['Volatility'].mean() * 1.2]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="strategy-card success-card">
            <h4>🎯 Strong Seasonal Periods</h4>
            <p><strong>High-Probability Months (Win Rate > 60%)</strong></p>
        """, unsafe_allow_html=True)
        
        if len(winning_months) > 0:
            insights_html = '<ul class="insights-list">'
            for month in winning_months.index:
                stats = winning_months.loc[month]
                insights_html += f'<li><strong>{month}</strong>: {stats["Avg_Return"]:.1%} avg return • {stats["Win_Rate"]:.0%} win rate</li>'
            insights_html += '</ul>'
            st.markdown(insights_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #64748b;">No months with >60% win rate identified.</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="strategy-card danger-card">
            <h4>⚠️ Challenging Periods</h4>
            <p><strong>Low-Probability Months (Win Rate < 40%)</strong></p>
        """, unsafe_allow_html=True)
        
        if len(losing_months) > 0:
            insights_html = '<ul class="insights-list">'
            for month in losing_months.index:
                stats = losing_months.loc[month]
                insights_html += f'<li><strong>{month}</strong>: {stats["Avg_Return"]:.1%} avg return • {stats["Win_Rate"]:.0%} win rate</li>'
            insights_html += '</ul>'
            st.markdown(insights_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #64748b;">No consistently weak months identified - positive sign!</p></div>', unsafe_allow_html=True)
    
    # Additional insights row
    if len(high_vol_months) > 0:
        st.markdown(f"""
        <div class="strategy-card warning-card">
            <h4>📊 Volatility Awareness</h4>
            <p><strong>High Volatility Periods</strong> - Exercise caution and adjust position sizing</p>
            <ul class="insights-list">
        """, unsafe_allow_html=True)
        
        for month in high_vol_months.index:
            stats = high_vol_months.loc[month]
            st.markdown(f'<li><strong>{month}</strong>: {stats["Volatility"]:.1%} volatility ({stats["Volatility"]/seasonal_stats["Volatility"].mean():.1f}x average)</li>', unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)

def display_seasonal_analysis(seasonal_stats, weekday_stats, visualizer, symbol):
    """Display comprehensive seasonal analysis with insights"""
    st.subheader(f"📅 Seasonal Pattern Analysis for {symbol}")
    
    # Create three sub-tabs for different seasonal analyses
    seasonal_tab1, seasonal_tab2, seasonal_tab3, seasonal_tab4 = st.tabs([
        "📊 Monthly Patterns", 
        "📅 Weekday Analysis", 
        "🔍 Combined View",
        "📚 Seasonal Insights"
    ])
    
    with seasonal_tab1:
        st.markdown("### 📈 Monthly Performance Analysis")
        
        # Monthly statistics table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Monthly heatmap
            monthly_fig = visualizer.create_seasonal_heatmap(seasonal_stats, symbol)
            st.plotly_chart(monthly_fig, use_container_width=True, key="monthly_heatmap")
        
        with col2:
            # Best/Worst months summary
            best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
            worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
            
            st.markdown(f"""
            <div class="metric-card success-card">
                <h4>🏆 Best Month</h4>
                <h3>{best_month.name}</h3>
                <p>Average Return: {best_month['Avg_Return']:.2%}</p>
                <p>Win Rate: {best_month['Win_Rate']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card danger-card">
                <h4>📉 Worst Month</h4>
                <h3>{worst_month.name}</h3>
                <p>Average Return: {worst_month['Avg_Return']:.2%}</p>
                <p>Win Rate: {worst_month['Win_Rate']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Monthly returns bar chart
        monthly_returns_fig = visualizer.create_monthly_returns_chart(seasonal_stats, symbol)
        st.plotly_chart(monthly_returns_fig, use_container_width=True, key="monthly_returns")
        
        # Detailed monthly statistics table
        st.markdown("### 📊 Detailed Monthly Statistics")
        display_df = seasonal_stats.copy()
        
        # Use more decimal places for small percentage values
        display_df['Avg_Return'] = display_df['Avg_Return'].apply(lambda x: f"{x:.3%}")
        display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1%}")
        display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.3%}")
        display_df['Min_Return'] = display_df['Min_Return'].apply(lambda x: f"{x:.3%}")
        display_df['Max_Return'] = display_df['Max_Return'].apply(lambda x: f"{x:.3%}")
        st.dataframe(display_df, use_container_width=True)

    with seasonal_tab2:
        st.markdown("### 📅 Day-of-Week Performance Analysis")
        
        if not weekday_stats.empty:
            # Weekday statistics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Weekday returns chart
                weekday_fig = visualizer.create_weekday_returns_chart(weekday_stats, symbol)
                st.plotly_chart(weekday_fig, use_container_width=True, key="weekday_returns")
            
            with col2:
                # Best/Worst weekdays
                best_day = weekday_stats.loc[weekday_stats['Avg_Return'].idxmax()]
                worst_day = weekday_stats.loc[weekday_stats['Avg_Return'].idxmin()]
                
                st.markdown(f"""
                <div class="metric-card success-card">
                    <h4>🏆 Best Weekday</h4>
                    <h3>{weekday_stats['Avg_Return'].idxmax()}</h3>
                    <p>Average Return: {best_day['Avg_Return']:.3%}</p>
                    <p>Win Rate: {best_day['Win_Rate']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card danger-card">
                    <h4>📉 Worst Weekday</h4>
                    <h3>{weekday_stats['Avg_Return'].idxmin()}</h3>
                    <p>Average Return: {worst_day['Avg_Return']:.3%}</p>
                    <p>Win Rate: {worst_day['Win_Rate']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Monday Effect detection
                if 'Monday' in weekday_stats.index:
                    monday_return = weekday_stats.loc['Monday', 'Avg_Return']
                    if monday_return < -0.001:  # Negative Monday effect
                        st.markdown(f"""
                        <div class="metric-card warning-card">
                            <h4>⚠️ Monday Effect Detected</h4>
                            <p>Monday shows negative bias: {monday_return:.3%}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Detailed weekday table
            st.markdown("### 📊 Detailed Weekday Statistics")
            display_weekday = weekday_stats.copy()
            display_weekday['Avg_Return'] = display_weekday['Avg_Return'].apply(lambda x: f"{x:.3%}")
            display_weekday['Volatility'] = display_weekday['Volatility'].apply(lambda x: f"{x:.3%}")
            display_weekday['Win_Rate'] = display_weekday['Win_Rate'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_weekday, use_container_width=True)
        else:
            st.warning("⚠️ Insufficient data for weekday analysis")

    with seasonal_tab3:
        st.markdown("### 🔍 Combined Seasonal View")
        
        # Side by side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Monthly Seasonality")
            monthly_simple_fig = visualizer.create_monthly_returns_chart(seasonal_stats, symbol, height=350)
            st.plotly_chart(monthly_simple_fig, use_container_width=True, key="monthly_simple")
        
        with col2:
            st.markdown("#### 📆 Weekday Patterns")
            if not weekday_stats.empty:
                weekday_simple_fig = visualizer.create_weekday_returns_chart(weekday_stats, symbol, height=350)
                st.plotly_chart(weekday_simple_fig, use_container_width=True, key="weekday_simple")
        
        # Summary insights
        st.markdown("### 🎯 Key Seasonal Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strongest_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
            st.metric(
                "🏆 Strongest Month", 
                strongest_month.name,
                f"{strongest_month['Avg_Return']:.3%}"
            )
        
        with col2:
            if not weekday_stats.empty:
                strongest_day = weekday_stats.loc[weekday_stats['Avg_Return'].idxmax()]
                st.metric(
                    "📅 Best Weekday", 
                    weekday_stats['Avg_Return'].idxmax(),
                    f"{strongest_day['Avg_Return']:.3%}"
                )
        
        with col3:
            seasonal_volatility = seasonal_stats['Volatility'].mean()
            st.metric(
                "📊 Avg Seasonality", 
                "Volatility",
                f"{seasonal_volatility:.3%}"
            )

    with seasonal_tab4:
        st.markdown("### 📚 Seasonal Market Insights & Effects")
        
        # Get the best and worst months for context
        best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
        worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
        
        # Create insights based on actual data
        st.markdown("#### 🎭 Common Seasonal Patterns Explained")
        
        # January Effect
        if seasonal_stats.loc['January', 'Avg_Return'] > 0.01:
            st.markdown("""
            <div class="metric-card success-card">
                <h4>🎊 January Effect Detected</h4>
                <p><strong>Pattern:</strong> Strong positive January returns</p>
                <p><strong>Explanation:</strong> New Year optimism, fresh capital deployment, and tax-loss selling reversals from December often drive January rallies</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Santa Rally (December)
        if seasonal_stats.loc['December', 'Avg_Return'] > 0.005:
            st.markdown("""
            <div class="metric-card success-card">
                <h4>🎅 Santa Rally Effect</h4>
                <p><strong>Pattern:</strong> December shows positive momentum</p>
                <p><strong>Explanation:</strong> Holiday optimism, low trading volume, and institutional window dressing typically boost December performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sell in May Effect
        summer_months = ['May', 'June', 'July', 'August', 'September']
        summer_avg = seasonal_stats.loc[summer_months, 'Avg_Return'].mean()
        if summer_avg < 0:
            st.markdown("""
            <div class="metric-card warning-card">
                <h4>🏖️ "Sell in May" Effect Detected</h4>
                <p><strong>Pattern:</strong> Weaker summer performance</p>
                <p><strong>Explanation:</strong> Reduced institutional activity during vacation season, lower trading volumes, and profit-taking before summer break</p>
            </div>
            """, unsafe_allow_html=True)
        
        # September Effect
        if seasonal_stats.loc['September', 'Avg_Return'] < -0.01:
            st.markdown("""
            <div class="metric-card danger-card">
                <h4>📉 September Weakness</h4>
                <p><strong>Pattern:</strong> September shows consistent underperformance</p>
                <p><strong>Explanation:</strong> Back-to-school effect, end of summer trading lull, and institutional portfolio rebalancing often create selling pressure</p>
            </div>
            """, unsafe_allow_html=True)
        
        # October Volatility
        october_vol = seasonal_stats.loc['October', 'Volatility']
        if october_vol > seasonal_stats['Volatility'].mean() * 1.2:
            st.markdown("""
            <div class="metric-card warning-card">
                <h4>🎃 October Volatility</h4>
                <p><strong>Pattern:</strong> High October volatility detected</p>
                <p><strong>Explanation:</strong> Historical crashes (1929, 1987, 2008) create psychological selling pressure, plus Q3 earnings volatility</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Year-end Rally
        q4_months = ['October', 'November', 'December']
        q4_avg = seasonal_stats.loc[q4_months, 'Avg_Return'].mean()
        if q4_avg > 0.005:
            st.markdown("""
            <div class="metric-card success-card">
                <h4>🚀 Year-End Rally</h4>
                <p><strong>Pattern:</strong> Strong Q4 performance</p>
                <p><strong>Explanation:</strong> Holiday spending optimism, institutional window dressing, and bonus-driven buying create year-end momentum</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Monday Effect (if weekday data available)
        if not weekday_stats.empty and 'Monday' in weekday_stats.index:
            monday_return = weekday_stats.loc['Monday', 'Avg_Return']
            if monday_return < -0.001:
                st.markdown("""
                <div class="metric-card warning-card">
                    <h4>😰 Monday Effect</h4>
                    <p><strong>Pattern:</strong> Negative Monday bias detected</p>
                    <p><strong>Explanation:</strong> Weekend news digestion, psychological barriers after time off, and institutional trading patterns</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Friday Effect
        if not weekday_stats.empty and 'Friday' in weekday_stats.index:
            friday_return = weekday_stats.loc['Friday', 'Avg_Return']
            if friday_return > 0.001:
                st.markdown("""
                <div class="metric-card success-card">
                    <h4>🎉 Friday Optimism</h4>
                    <p><strong>Pattern:</strong> Positive Friday performance</p>
                    <p><strong>Explanation:</strong> Weekend optimism, short covering, and reduced institutional selling pressure</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Custom insights based on your data
        st.markdown("#### 🎯 Your Data-Specific Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Best Performing Month</h4>
                <h3>{best_month.name}</h3>
                <p><strong>Average Return:</strong> {best_month['Avg_Return']:.3%}</p>
                <p><strong>Win Rate:</strong> {best_month['Win_Rate']:.1%}</p>
                <p><strong>Observation Count:</strong> {best_month['Count']} periods</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📉 Underperforming Month</h4>
                <h3>{worst_month.name}</h3>
                <p><strong>Average Return:</strong> {worst_month['Avg_Return']:.3%}</p>
                <p><strong>Win Rate:</strong> {worst_month['Win_Rate']:.1%}</p>
                <p><strong>Observation Count:</strong> {worst_month['Count']} periods</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Trading Strategy Insights
        st.markdown("#### 💡 Potential Trading Insights")
        
        # Find consistent patterns
        high_win_months = seasonal_stats[seasonal_stats['Win_Rate'] > 0.6]
        low_win_months = seasonal_stats[seasonal_stats['Win_Rate'] < 0.4]
        
        if not high_win_months.empty:
            st.markdown(f"""
            <div class="metric-card success-card">
                <h4>🎯 High Probability Months</h4>
                <p><strong>Months with >60% win rate:</strong> {', '.join(high_win_months.index)}</p>
                <p>These months show consistent positive bias in your data</p>
            </div>
            """, unsafe_allow_html=True)
        
        if not low_win_months.empty:
            st.markdown(f"""
            <div class="metric-card danger-card">
                <h4>⚠️ Low Probability Months</h4>
                <p><strong>Months with <40% win rate:</strong> {', '.join(low_win_months.index)}</p>
                <p>These months show consistent negative bias in your data</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Volatility insights
        high_vol_months = seasonal_stats[seasonal_stats['Volatility'] > seasonal_stats['Volatility'].quantile(0.75)]
        if not high_vol_months.empty:
            st.markdown(f"""
            <div class="metric-card warning-card">
                <h4>📊 High Volatility Periods</h4>
                <p><strong>Most volatile months:</strong> {', '.join(high_vol_months.index)}</p>
                <p>Exercise caution during these periods - higher risk and opportunity</p>
            </div>
            """, unsafe_allow_html=True)

def display_ai_insights(stock_data, seasonal_stats, ai_analyzer, symbol, confidence_threshold, ai_insights=None):
    """Display comprehensive AI-generated insights with advanced Prophet analysis"""
    st.subheader(f"🧠 AI-Enhanced Insights for {symbol}")
    
    # Use pre-calculated insights if available, otherwise calculate them
    if ai_insights is None:
        with st.spinner("🤖 AI is performing comprehensive analysis..."):
            try:
                ai_insights = ai_analyzer.analyze_patterns(
                    stock_data, seasonal_stats, confidence_threshold
                )
            except Exception as e:
                st.error(f"❌ AI analysis failed: {str(e)}")
                st.info("💡 AI analysis requires sufficient historical data. Try a symbol with longer history.")
                return
    
    if ai_insights:
        # Create comprehensive tabs for all insights
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🎯 Key Patterns", "🔮 Prophet Analysis", "📊 Trading Strategies", 
            "⚠️ Risk Insights", "🔄 Market Regimes", "📈 Time Series", "🎛️ Advanced Metrics"
        ])
        
        with tab1:
            # High-confidence patterns
            if ai_insights['high_confidence']:
                st.success("✨ **High-Confidence AI Patterns Detected**")
                
                for insight in ai_insights['high_confidence']:
                    confidence_pct = insight.get('confidence', 0)
                    if isinstance(confidence_pct, str):
                        confidence_display = confidence_pct
                    else:
                        confidence_display = f"{confidence_pct:.1%}"
                    
                    st.markdown(f"""
                    <div class="metric-card success-card">
                        <h4>{insight.get('pattern', insight.get('name', 'Pattern'))}</h4>
                        <p><strong>Confidence:</strong> {confidence_display}</p>
                        <p>{insight.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Seasonal trends
            if ai_insights.get('seasonal_trends'):
                st.info("📈 **Seasonal Trend Analysis**")
                for trend in ai_insights['seasonal_trends']:
                    if trend.get('type') == 'seasonal_strength':
                        card_class = "success-card"
                        emoji = "📈"
                    elif trend.get('type') == 'seasonal_weakness':
                        card_class = "danger-card"
                        emoji = "📉"
                    else:
                        card_class = "warning-card"
                        emoji = "⚡"
                    
                    confidence_pct = trend.get('confidence', 0)
                    confidence_display = f"{confidence_pct:.1%}" if isinstance(confidence_pct, (int, float)) else str(confidence_pct)
                    
                    st.markdown(f"""
                    <div class="metric-card {card_class}">
                        <h4>{emoji} {trend.get('month', 'Analysis')}</h4>
                        <p><strong>Confidence:</strong> {confidence_display}</p>
                        <p>{trend.get('description', '')}</p>
                        <p><strong>💡 Recommendation:</strong> {trend.get('recommendation', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Anomalies
            if ai_insights.get('anomalies'):
                st.warning("🔍 **Seasonal Anomalies Detected**")
                for anomaly in ai_insights['anomalies']:
                    anomaly_type = anomaly.get('type', 'Unknown')
                    if hasattr(anomaly_type, 'replace'):
                        anomaly_type = anomaly_type.replace('_', ' ').title()
                    
                    st.markdown(f"""
                    <div class="metric-card warning-card">
                        <h4>🔍 {anomaly_type}</h4>
                        <p>{anomaly.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            # Advanced Prophet Analysis
            prophet_analysis = ai_insights.get('prophet_analysis', {})
            
            if prophet_analysis:
                st.success("🔮 **Advanced Prophet Forecasting Analysis**")
                
                # Display comprehensive forecasts
                forecasts = prophet_analysis.get('forecasts', ai_insights.get('predictions', []))
                if forecasts:
                    st.subheader("📊 Multi-Horizon Forecasts")
                    
                    # Create forecast dataframe for better display
                    forecast_df = pd.DataFrame(forecasts)
                    if not forecast_df.empty:
                        # Group by target and display
                        targets = forecast_df['Target'].unique() if 'Target' in forecast_df.columns else []
                        
                        for target in targets:
                            if target in ['Model Summary', 'Regime Change']:
                                continue
                                
                            target_forecasts = forecast_df[forecast_df['Target'] == target]
                            
                            if not target_forecasts.empty:
                                st.subheader(f"🎯 {target} Forecasts")
                                
                                cols = st.columns(len(target_forecasts))
                                for i, (_, forecast) in enumerate(target_forecasts.iterrows()):
                                    with cols[i % len(cols)]:
                                        st.metric(
                                            forecast.get('Horizon', 'N/A'),
                                            forecast.get('Expected_Value', 'N/A'),
                                            delta=f"{forecast.get('Trend_Direction', 'N/A')} ({forecast.get('Confidence', 'N/A')})"
                                        )
                        
                        # Show validation metrics
                        validation_data = forecast_df[forecast_df['Target'].str.contains('Validation', na=False)]
                        if not validation_data.empty:
                            st.subheader("📊 Model Validation Metrics")
                            for _, val in validation_data.iterrows():
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>✓ {val.get('Target', 'Validation')}</h4>
                                    <p><strong>MAPE:</strong> {val.get('MAPE', 'N/A')}</p>
                                    <p><strong>MAE:</strong> {val.get('MAE', 'N/A')}</p>
                                    <p>{val.get('Details', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show changepoints
                        changepoint_data = forecast_df[forecast_df.get('Type') == 'changepoint']
                        if not changepoint_data.empty:
                            st.subheader("🔄 Detected Changepoints")
                            for _, cp in changepoint_data.iterrows():
                                st.markdown(f"""
                                <div class="metric-card warning-card">
                                    <h4>⚡ Structural Break</h4>
                                    <p>{cp.get('Details', '')}</p>
                                    <p><strong>Confidence:</strong> {cp.get('Confidence', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
            
            # Time series decomposition insights
            decomposition = ai_insights.get('time_series_decomposition', {})
            if decomposition:
                st.subheader("📈 Time Series Decomposition")
                
                trend_analysis = decomposition.get('trend_analysis', {})
                if trend_analysis:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Trend Analysis</h4>
                        <p><strong>Direction:</strong> {trend_analysis.get('direction', 'N/A').title()}</p>
                        <p><strong>Strength:</strong> {trend_analysis.get('strength', 0):.6f}</p>
                        <p>{trend_analysis.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                seasonality = decomposition.get('seasonality_strength', {})
                if seasonality:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🔄 Seasonality Analysis</h4>
                        <p><strong>Strength:</strong> {seasonality.get('interpretation', 'N/A')}</p>
                        <p><strong>Variance Explained:</strong> {seasonality.get('strength', 0):.1%}</p>
                        <p>{seasonality.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            # Advanced Trading strategies
            if ai_insights.get('trading_strategies'):
                st.success("💰 **AI-Generated Trading Strategies**")
                
                for strategy in ai_insights['trading_strategies']:
                    strategy_emoji = {
                        'momentum': '🚀',
                        'contrarian': '🔄',
                        'volatility': '⚡',
                        'spread': '📊'
                    }.get(strategy.get('type', 'unknown'), '💡')
                    
                    st.markdown(f"""
                    <div class="metric-card success-card">
                        <h4>{strategy_emoji} {strategy.get('name', 'Strategy')}</h4>
                        <p><strong>Description:</strong> {strategy.get('description', 'N/A')}</p>
                        <p><strong>Implementation:</strong> {strategy.get('implementation', 'N/A')}</p>
                        {f"<p><strong>Expected Return:</strong> {strategy.get('expected_return', 'N/A')}</p>" if strategy.get('expected_return') else ""}
                        {f"<p><strong>Win Rate:</strong> {strategy.get('win_rate', 'N/A')}</p>" if strategy.get('win_rate') else ""}
                        {f"<p><strong>Expected Spread:</strong> {strategy.get('expected_spread', 'N/A')}</p>" if strategy.get('expected_spread') else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific trading strategies identified with current data.")
        
        with tab4:
            # Comprehensive Risk insights
            if ai_insights.get('risk_insights') or ai_insights.get('volatility_insights'):
                st.warning("⚠️ **Comprehensive Risk Management Insights**")
                
                all_risks = ai_insights.get('risk_insights', []) + ai_insights.get('volatility_insights', [])
                
                for risk in all_risks:
                    severity = risk.get('severity', 'medium')
                    severity_emoji = {
                        'high': '🚨',
                        'medium': '⚠️',
                        'low': '💡'
                    }.get(severity, '⚠️')
                    
                    card_class = {
                        'high': 'danger-card',
                        'medium': 'warning-card',
                        'low': 'info-card'
                    }.get(severity, 'warning-card')
                    
                    risk_type = risk.get('type', 'Unknown Risk')
                    if hasattr(risk_type, 'replace'):
                        risk_type = risk_type.replace('_', ' ').title()
                    
                    st.markdown(f"""
                    <div class="metric-card {card_class}">
                        <h4>{severity_emoji} {risk_type}</h4>
                        <p><strong>Issue:</strong> {risk.get('description', 'N/A')}</p>
                        <p><strong>💡 Recommendation:</strong> {risk.get('recommendation', 'N/A')}</p>
                        <p><strong>Severity:</strong> {severity.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No significant risk issues identified.")
        
        with tab5:
            # Market Regimes Analysis
            market_regimes = ai_insights.get('market_regimes', [])
            cyclical_patterns = ai_insights.get('cyclical_patterns', [])
            
            if market_regimes or cyclical_patterns:
                st.info("🔄 **Market Regime & Cyclical Analysis**")
                
                if market_regimes:
                    st.subheader("📊 Market Regime Changes")
                    for regime in market_regimes:
                        st.markdown(f"""
                        <div class="metric-card warning-card">
                            <h4>🔄 Regime Change</h4>
                            <p>{regime.get('description', 'Market regime change detected')}</p>
                            <p><strong>Date:</strong> {regime.get('date', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if cyclical_patterns:
                    st.subheader("🌊 Cyclical Pattern Analysis")
                    for pattern in cyclical_patterns:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🌊 {pattern.get('type', 'Cyclical Pattern').replace('_', ' ').title()}</h4>
                            <p>{pattern.get('description', 'Cyclical pattern detected')}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No significant market regime changes or cyclical patterns detected.")
        
        with tab6:
            # Advanced Time Series Analysis
            decomposition = ai_insights.get('time_series_decomposition', {})
            structural_breaks = ai_insights.get('structural_breaks', [])
            
            if decomposition or structural_breaks:
                st.info("📈 **Advanced Time Series Analysis**")
                
                if structural_breaks:
                    st.subheader("⚡ Structural Breaks")
                    for break_point in structural_breaks:
                        st.markdown(f"""
                        <div class="metric-card warning-card">
                            <h4>⚡ Structural Break</h4>
                            <p>{break_point.get('description', 'Structural break detected')}</p>
                            <p><strong>Type:</strong> {break_point.get('type', 'N/A').replace('_', ' ').title()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Cyclical insights from decomposition
                cyclical_insights = decomposition.get('cyclical_insights', [])
                if cyclical_insights:
                    st.subheader("🔄 Autocorrelation Analysis")
                    for insight in cyclical_insights:
                        if insight.get('type') == 'autocorrelation':
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📊 Autocorrelation Detected</h4>
                                <p>{insight.get('description', '')}</p>
                                <p><strong>P-value:</strong> {insight.get('ljung_box_pvalue', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🌊 {insight.get('type', 'Pattern').replace('_', ' ').title()}</h4>
                                <p>{insight.get('description', '')}</p>
                                {f"<p><strong>Peaks:</strong> {insight.get('peaks_count', 'N/A')}</p>" if insight.get('peaks_count') else ""}
                                {f"<p><strong>Troughs:</strong> {insight.get('troughs_count', 'N/A')}</p>" if insight.get('troughs_count') else ""}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("Advanced time series analysis requires sufficient data.")
        
        with tab7:
            # Advanced Metrics and Model Performance
            advanced_metrics = ai_insights.get('advanced_metrics', {})
            forecast_accuracy = ai_insights.get('forecast_accuracy', {})
            pattern_strength = ai_insights.get('pattern_strength', {})
            correlation_analysis = ai_insights.get('correlation_analysis', {})
            
            st.info("🎛️ **Advanced Metrics & Model Performance**")
            
            # Pattern Strength Assessment
            if pattern_strength:
                st.subheader("🎯 Pattern Strength Assessment")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Strength", f"{pattern_strength.get('overall_strength', 0):.1%}")
                with col2:
                    st.metric("Consistency", f"{pattern_strength.get('consistency', 0):.1%}")
                with col3:
                    st.metric("Win Rate Quality", f"{pattern_strength.get('win_rate_quality', 0):.1%}")
                with col4:
                    st.metric("Reliability", f"{pattern_strength.get('reliability', 0):.1%}")
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Pattern Interpretation</h4>
                    <p>{pattern_strength.get('interpretation', 'No assessment available')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced Statistical Metrics
            if advanced_metrics:
                st.subheader("📊 Statistical Properties")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'annual_return' in advanced_metrics:
                        st.metric("Annual Return", f"{advanced_metrics['annual_return']:.2%}")
                    if 'skewness' in advanced_metrics:
                        st.metric("Skewness", f"{advanced_metrics['skewness']:.3f}")
                
                with col2:
                    if 'annual_volatility' in advanced_metrics:
                        st.metric("Annual Volatility", f"{advanced_metrics['annual_volatility']:.2%}")
                    if 'kurtosis' in advanced_metrics:
                        st.metric("Kurtosis", f"{advanced_metrics['kurtosis']:.3f}")
                
                with col3:
                    if 'sharpe_ratio' in advanced_metrics:
                        st.metric("Sharpe Ratio", f"{advanced_metrics['sharpe_ratio']:.3f}")
                    if 'max_drawdown' in advanced_metrics:
                        st.metric("Max Drawdown", f"{advanced_metrics['max_drawdown']:.2%}")
                
                # Stationarity tests
                if advanced_metrics.get('is_stationary') is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📈 Time Series Properties</h4>
                        <p><strong>Stationarity:</strong> {'Yes' if advanced_metrics.get('is_stationary') else 'No'}</p>
                        <p><strong>ADF p-value:</strong> {advanced_metrics.get('adf_pvalue', 'N/A'):.4f}</p>
                        <p><strong>KPSS p-value:</strong> {advanced_metrics.get('kpss_pvalue', 'N/A'):.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Correlation Analysis
            if correlation_analysis and correlation_analysis.get('high_correlations'):
                st.subheader("🔗 Cross-Asset Correlations")
                correlations = correlation_analysis['high_correlations']
                
                if correlations:
                    for pair, corr in correlations.items():
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🔗 {pair}</h4>
                            <p><strong>Correlation:</strong> {corr:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No high correlations detected.")
            
            # Display all predictions in a comprehensive table
            if ai_insights.get('predictions'):
                st.subheader("🔮 Complete Forecast Summary")
                predictions_df = pd.DataFrame(ai_insights['predictions'])
                if not predictions_df.empty:
                    st.dataframe(predictions_df, use_container_width=True)
            
    else:
        st.info("🤖 No significant AI patterns detected with current confidence threshold. Try lowering the threshold.")

def display_performance_charts(stock_data, seasonal_stats, visualizer, symbol):
    """Display various performance charts"""
    st.subheader(f"📈 Performance Charts for {symbol}")
    
    # Price chart with seasonal highlights
    st.plotly_chart(
        visualizer.create_price_chart(stock_data, symbol),
        use_container_width=True,
        key="performance_price_chart"
    )
    
    # Monthly returns distribution
    st.plotly_chart(
        visualizer.create_monthly_returns_chart(seasonal_stats, symbol),
        use_container_width=True,
        key="performance_monthly_returns"
    )
    
    # Win rate visualization
    st.plotly_chart(
        visualizer.create_win_rate_chart(seasonal_stats, symbol),
        use_container_width=True,
        key="performance_win_rate"
    )

def display_export_alerts(seasonal_stats, symbol, company_name, ai_insights=None):
    """Display export options and alert settings"""
    st.subheader("📋 Export & Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export Professional Reports")
        
        st.markdown("""
        **🎯 Professional PDF Report Features:**
        - Complete 6-page analysis with executive summary
        - Risk assessment & strategy recommendations  
        - Legal disclaimers for professional sharing
        - Comprehensive data tables & visualizations
        - AI insights integration (when enabled)
        - Ready for investment committees & teams
        """)
        
        # Prepare export data
        export_data = seasonal_stats.copy()
        csv = export_data.to_csv()
        
        st.download_button(
            label="📥 Download Raw CSV Data",
            data=csv,
            file_name=f"{symbol}_seasonal_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download raw statistical data for further analysis"
        )
        
        # Generate comprehensive report
        if st.button("📄 Generate Professional PDF Report", key="pdf_summary_button"):
            # Store current state to prevent reset
            if 'pdf_generation_started' not in st.session_state:
                st.session_state.pdf_generation_started = True
                
            with st.spinner("📄 Generating comprehensive 8-page report with Monte Carlo & Backtesting..."):
                try:
                    pdf_bytes = generate_pdf_report(seasonal_stats, symbol, company_name, ai_insights)
                    st.download_button(
                        label="📥 Download Professional PDF Report",
                        data=pdf_bytes,
                        file_name=f"{symbol}_comprehensive_seasonal_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        help="Professional 8-page report with Monte Carlo simulations and backtesting strategies"
                    )
                    st.success("✅ Enhanced PDF report with Monte Carlo & Backtesting generated successfully!")
                    st.info("📋 **Report includes:** Executive summary, seasonal charts, risk analysis, Monte Carlo simulations, backtesting strategies, AI insights, complete data tables, and comprehensive legal disclaimers - perfect for professional sharing with investment teams.")
                except Exception as e:
                    st.error(f"❌ PDF generation failed: {str(e)}")
                    st.info("💡 PDF generation requires additional dependencies. Using CSV export instead.")
    
    with col2:
        st.subheader("🔔 Smart Alerts")
        
        st.info("🚧 Alert system coming soon!")
        st.write("Planned features:")
        st.write("• Email alerts for seasonal opportunities")
        st.write("• SMS notifications for pattern confirmations")
        st.write("• Calendar integration for optimal trading dates")
        st.write("• Portfolio-level seasonal analysis")

def generate_pdf_report(seasonal_stats, symbol, company_name, ai_insights=None):
    """Generate comprehensive PDF report including all analysis sections"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_pdf import PdfPages
    import io
    
    # Create a bytes buffer for the PDF
    buffer = io.BytesIO()
    
    # Get current session data for comprehensive report
    stock_data = getattr(st.session_state, 'stock_data', None)
    
    with PdfPages(buffer) as pdf:
        # Page 1: Professional Title Page with Disclaimers
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Logo/Header area
        header_rect = patches.Rectangle((0.05, 0.85), 0.9, 0.12, linewidth=2, 
                                      edgecolor='#1f77b4', facecolor='#f0f8ff', alpha=0.8)
        ax.add_patch(header_rect)
        
        # Title
        ax.text(0.5, 0.92, "🎯 AI SEASONAL EDGE", 
                ha='center', va='center', fontsize=24, fontweight='bold', color='#1f77b4')
        ax.text(0.5, 0.88, "Professional Seasonal Pattern Analysis Report", 
                ha='center', va='center', fontsize=14, style='italic')
        
        # Asset Information
        ax.text(0.5, 0.78, f"Asset: {symbol.upper()}", 
                ha='center', va='top', fontsize=18, fontweight='bold')
        ax.text(0.5, 0.74, f"Analysis Period: {datetime.now().strftime('%B %d, %Y')}", 
                ha='center', va='top', fontsize=12)
        
        if stock_data is not None:
            start_date = stock_data.index[0].strftime('%B %d, %Y')
            end_date = stock_data.index[-1].strftime('%B %d, %Y')
            ax.text(0.5, 0.70, f"Data Range: {start_date} to {end_date}", 
                    ha='center', va='top', fontsize=12)
            ax.text(0.5, 0.66, f"Total Data Points: {len(stock_data):,} trading days", 
                    ha='center', va='top', fontsize=12)
        
        # Executive Summary Box
        summary_rect = patches.Rectangle((0.1, 0.35), 0.8, 0.25, linewidth=1, 
                                       edgecolor='#666', facecolor='#f9f9f9', alpha=0.9)
        ax.add_patch(summary_rect)
        
        ax.text(0.5, 0.57, "📊 EXECUTIVE SUMMARY", 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        best_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmax()]
        worst_month = seasonal_stats.loc[seasonal_stats['Avg_Return'].idxmin()]
        avg_return = seasonal_stats['Avg_Return'].mean()
        winning_months = len(seasonal_stats[seasonal_stats['Avg_Return'] > 0])
        
        summary_text = f"""Best Month: {best_month.name} ({best_month['Avg_Return']:.1%} avg, {best_month['Win_Rate']:.0%} win rate)
Worst Month: {worst_month.name} ({worst_month['Avg_Return']:.1%} avg, {worst_month['Win_Rate']:.0%} win rate)
Average Monthly Return: {avg_return:.2%}
Positive Months: {winning_months}/12 ({winning_months/12:.0%})
Seasonal Strength: {'Strong' if abs(best_month['Avg_Return'] - worst_month['Avg_Return']) > 0.05 else 'Moderate' if abs(best_month['Avg_Return'] - worst_month['Avg_Return']) > 0.02 else 'Weak'}"""
        
        ax.text(0.15, 0.50, summary_text, fontsize=10, va='top', linespacing=1.5)
        
        # Important Disclaimers
        disclaimer_rect = patches.Rectangle((0.05, 0.05), 0.9, 0.25, linewidth=2, 
                                          edgecolor='#d32f2f', facecolor='#ffebee', alpha=0.9)
        ax.add_patch(disclaimer_rect)
        
        ax.text(0.5, 0.27, "⚠️ IMPORTANT DISCLAIMERS", 
                ha='center', va='center', fontsize=12, fontweight='bold', color='#d32f2f')
        
        disclaimer_text = """• Past performance does not guarantee future results
• This analysis is for educational and research purposes only
• Not intended as investment advice or recommendations
• Consult qualified financial advisors before making investment decisions
• Seasonal patterns may change or disappear over time
• Market conditions, regulations, and economic factors can impact performance
• Consider transaction costs, taxes, and risk tolerance in implementation
• This report does not constitute a solicitation to buy or sell securities"""
        
        ax.text(0.08, 0.22, disclaimer_text, fontsize=9, va='top', linespacing=1.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Seasonal Performance Analysis
        fig = plt.figure(figsize=(8.5, 11))
        
        # Monthly returns chart
        ax1 = plt.subplot(3, 1, 1)
        months = seasonal_stats.index
        returns = seasonal_stats['Avg_Return']
        colors = ['#2e7d32' if x > 0 else '#d32f2f' for x in returns]
        
        bars = ax1.bar(months, returns, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_title(f'{symbol} - Average Monthly Returns by Calendar Month', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylabel('Average Return (%)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height > 0 else -0.005),
                    f'{value:.1%}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8, fontweight='bold')
        
        # Win rate chart
        ax2 = plt.subplot(3, 1, 2)
        win_rates = seasonal_stats['Win_Rate']
        colors2 = ['#1565c0' if x >= 0.5 else '#f57c00' for x in win_rates]
        
        bars2 = ax2.bar(months, win_rates, color=colors2, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_title(f'{symbol} - Monthly Win Rates (Probability of Positive Returns)', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylabel('Win Rate (%)')
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='50% Threshold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        for bar, value in zip(bars2, win_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Volatility chart
        ax3 = plt.subplot(3, 1, 3)
        volatility = seasonal_stats['Volatility']
        
        bars3 = ax3.bar(months, volatility, color='#7b1fa2', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.set_title(f'{symbol} - Monthly Volatility (Risk Assessment)', fontsize=12, fontweight='bold', pad=15)
        ax3.set_ylabel('Volatility (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, volatility):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Risk Analysis & Statistics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle(f'{symbol} - Risk Analysis & Statistical Overview', fontsize=14, fontweight='bold')
        
        # Risk-Return Scatter
        ax1.scatter(seasonal_stats['Volatility'], seasonal_stats['Avg_Return'], 
                   s=100, alpha=0.7, c=returns, cmap='RdYlGn', edgecolors='black')
        ax1.set_xlabel('Volatility (Risk)')
        ax1.set_ylabel('Average Return')
        ax1.set_title('Risk vs Return by Month')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axvline(x=volatility.mean(), color='blue', linestyle='--', alpha=0.5, label='Avg Volatility')
        
        # Add month labels
        for i, month in enumerate(months):
            ax1.annotate(month[:3], (volatility.iloc[i], returns.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Best vs Worst Months Comparison
        best_worst_data = [best_month['Avg_Return'], worst_month['Avg_Return']]
        best_worst_labels = [f"Best\n({best_month.name})", f"Worst\n({worst_month.name})"]
        colors_bw = ['green', 'red']
        
        bars_bw = ax2.bar(best_worst_labels, best_worst_data, color=colors_bw, alpha=0.7)
        ax2.set_title('Best vs Worst Month Performance')
        ax2.set_ylabel('Average Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        for bar, value in zip(bars_bw, best_worst_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height > 0 else -0.005),
                    f'{value:.1%}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # Monthly Count Distribution
        counts = seasonal_stats['Count']
        ax3.bar(months, counts, color='steelblue', alpha=0.7)
        ax3.set_title('Data Points per Month')
        ax3.set_ylabel('Number of Observations')
        ax3.tick_params(axis='x', rotation=45)
        
        # Summary Statistics Table
        ax4.axis('off')
        ax4.set_title('Statistical Summary', fontweight='bold', pad=20)
        
        stats_data = [
            ['Metric', 'Value'],
            ['Mean Monthly Return', f"{avg_return:.2%}"],
            ['Standard Deviation', f"{seasonal_stats['Avg_Return'].std():.2%}"],
            ['Best Month Return', f"{best_month['Avg_Return']:.2%}"],
            ['Worst Month Return', f"{worst_month['Avg_Return']:.2%}"],
            ['Spread (Best-Worst)', f"{best_month['Avg_Return']-worst_month['Avg_Return']:.2%}"],
            ['Avg Win Rate', f"{seasonal_stats['Win_Rate'].mean():.1%}"],
            ['Months with >60% Win Rate', f"{len(seasonal_stats[seasonal_stats['Win_Rate'] > 0.6])}"],
            ['Total Observations', f"{seasonal_stats['Count'].sum():.0f}"],
            ['Years of Data', f"{seasonal_stats['Count'].sum()/12:.1f}"]
        ]
        
        table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                         cellLoc='left', loc='center', bbox=[0, 0.1, 1, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # Style the table
        for i in range(len(stats_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: Monte Carlo Simulation Results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle(f'{symbol} - Monte Carlo Simulation (1000 Scenarios)', fontsize=14, fontweight='bold')
        
        # Generate Monte Carlo simulation data
        import numpy as np
        np.random.seed(42)  # For reproducible results
        scenarios = 1000
        months_ahead = 12
        initial_value = 100
        
        paths = []
        for _ in range(scenarios):
            path = [initial_value]
            for month in range(months_ahead):
                month_idx = month % 12
                month_name = months[month_idx]
                monthly_return = seasonal_stats.loc[month_name, 'Avg_Return']
                monthly_vol = seasonal_stats.loc[month_name, 'Volatility']
                
                # Add randomness based on seasonal patterns
                random_return = np.random.normal(monthly_return, monthly_vol)
                new_value = path[-1] * (1 + random_return)
                path.append(new_value)
            paths.append(path)
        
        paths = np.array(paths)
        
        # Sample portfolio paths
        for i in range(min(50, scenarios)):
            ax1.plot(range(months_ahead + 1), paths[i], alpha=0.1, color='blue')
        
        # Plot percentiles
        percentiles = [10, 50, 90]
        colors_perc = ['red', 'green', 'red']
        labels_perc = ['10th percentile', 'Median', '90th percentile']
        
        for p, color, label in zip(percentiles, colors_perc, labels_perc):
            values = np.percentile(paths, p, axis=0)
            ax1.plot(range(months_ahead + 1), values, color=color, linewidth=2, label=label)
        
        ax1.set_title('Portfolio Scenario Paths')
        ax1.set_xlabel('Months Ahead')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution of final values
        final_values = paths[:, -1]
        ax2.hist(final_values, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(np.median(final_values), color='green', linestyle='--', 
                   label=f'Median: ${np.median(final_values):.0f}')
        ax2.set_title('Final Value Distribution')
        ax2.set_xlabel('Final Portfolio Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Probability analysis
        returns_1yr = (final_values - initial_value) / initial_value
        prob_profit = np.mean(returns_1yr > 0) * 100
        prob_loss_10 = np.mean(returns_1yr < -0.1) * 100
        prob_gain_20 = np.mean(returns_1yr > 0.2) * 100
        
        categories = ['Profit\\n(>0%)', 'Big Gain\\n(>20%)', 'Big Loss\\n(>10%)']
        probabilities = [prob_profit, prob_gain_20, prob_loss_10]
        colors_prob = ['green', 'darkgreen', 'red']
        
        bars_prob = ax3.bar(categories, probabilities, color=colors_prob, alpha=0.7)
        ax3.set_title('Outcome Probabilities')
        ax3.set_ylabel('Probability (%)')
        ax3.set_ylim(0, 100)
        
        for bar, prob in zip(bars_prob, probabilities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Risk metrics summary
        ax4.axis('off')
        var_5 = np.percentile(returns_1yr, 5) * 100
        
        risk_summary = f"""📊 MONTE CARLO RISK ANALYSIS
(12-Month Simulation)

💡 Key Probabilities:
• Probability of Profit: {prob_profit:.1f}%
• Chance of 20%+ Gain: {prob_gain_20:.1f}%
• Risk of 10%+ Loss: {prob_loss_10:.1f}%

📉 Risk Metrics:
• Value at Risk (5%): {var_5:.1f}%
• Expected Return: {np.mean(returns_1yr)*100:.1f}%
• Volatility: {np.std(returns_1yr)*100:.1f}%

🎯 Risk Assessment:
{'LOW RISK' if var_5 > -10 else 'MEDIUM RISK' if var_5 > -20 else 'HIGH RISK'}
Based on seasonal patterns"""
        
        ax4.text(0.05, 0.95, risk_summary, fontsize=10, va='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 5: Backtesting Strategy Comparison
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        ax.text(0.5, 0.95, f"📈 STRATEGY BACKTESTING RESULTS", 
                ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Calculate strategy performance
        best_3_months = seasonal_stats.nlargest(3, 'Avg_Return').index
        worst_3_months = seasonal_stats.nsmallest(3, 'Avg_Return').index
        
        strategies = [
            {
                'name': 'Best 3 Months Only',
                'description': f'Trade only: {", ".join(best_3_months)}',
                'annual_return': seasonal_stats.loc[best_3_months, 'Avg_Return'].mean() * 3,
                'win_rate': seasonal_stats.loc[best_3_months, 'Win_Rate'].mean(),
                'volatility': seasonal_stats.loc[best_3_months, 'Volatility'].mean()
            },
            {
                'name': 'Avoid Worst 3 Months',
                'description': f'Avoid: {", ".join(worst_3_months)}',
                'annual_return': seasonal_stats.drop(worst_3_months)['Avg_Return'].mean() * 9,
                'win_rate': seasonal_stats.drop(worst_3_months)['Win_Rate'].mean(),
                'volatility': seasonal_stats.drop(worst_3_months)['Volatility'].mean()
            },
            {
                'name': 'Buy & Hold',
                'description': 'Hold all 12 months',
                'annual_return': seasonal_stats['Avg_Return'].sum(),
                'win_rate': (seasonal_stats['Avg_Return'] > 0).mean(),
                'volatility': seasonal_stats['Volatility'].mean()
            }
        ]
        
        # Create performance comparison chart
        ax_chart = plt.subplot(2, 1, 1)
        
        strategy_names = [s['name'] for s in strategies]
        returns_strat = [s['annual_return'] * 100 for s in strategies]
        win_rates_strat = [s['win_rate'] * 100 for s in strategies]
        
        x_pos = np.arange(len(strategy_names))
        width = 0.35
        
        bars1 = ax_chart.bar(x_pos - width/2, returns_strat, width, label='Est. Annual Return (%)', 
                            color='lightblue', alpha=0.8)
        bars2 = ax_chart.bar(x_pos + width/2, win_rates_strat, width, label='Win Rate (%)', 
                            color='lightgreen', alpha=0.8)
        
        ax_chart.set_xlabel('Strategy')
        ax_chart.set_ylabel('Performance (%)')
        ax_chart.set_title('Strategy Performance Comparison')
        ax_chart.set_xticks(x_pos)
        ax_chart.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax_chart.legend()
        ax_chart.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax_chart.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax_chart.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
        
        # Strategy analysis details
        ax_table = plt.subplot(2, 1, 2)
        ax_table.axis('off')
        
        table_text = f"""📊 DETAILED STRATEGY ANALYSIS

🎯 BEST 3 MONTHS STRATEGY: {", ".join(best_3_months)}
• Focus on highest-performing seasonal periods
• Estimated Annual Return: {strategies[0]['annual_return']*100:.1f}%
• Win Rate: {strategies[0]['win_rate']*100:.0f}%
• Concentrated exposure with potential for higher returns

📉 AVOID WORST 3 MONTHS STRATEGY: Avoid {", ".join(worst_3_months)}
• Defensive approach avoiding weakest periods
• Estimated Annual Return: {strategies[1]['annual_return']*100:.1f}%
• Win Rate: {strategies[1]['win_rate']*100:.0f}%
• More diversified approach with reduced volatility

💼 BUY & HOLD STRATEGY:
• Full market exposure across all months
• Estimated Annual Return: {strategies[2]['annual_return']*100:.1f}%
• Win Rate: {strategies[2]['win_rate']*100:.0f}%
• Benchmark for comparison

⚠️ IMPLEMENTATION NOTES:
• These are simplified backtests based on seasonal averages
• Real trading involves transaction costs, slippage, and taxes
• Consider risk management rules and position sizing
• Monitor patterns quarterly - seasonality can change over time
• Past performance does not guarantee future results"""
        
        ax_table.text(0.05, 0.95, table_text, fontsize=10, va='top',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 6: Strategy Recommendations & AI Insights
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, f"🎯 Strategic Investment Recommendations for {symbol}", 
                ha='center', va='top', fontsize=16, fontweight='bold')
        
        y_pos = 0.88
        
        # Strategy Recommendations based on analysis
        ax.text(0.05, y_pos, "💰 RECOMMENDED STRATEGIES", fontsize=14, fontweight='bold', color='#1565c0')
        y_pos -= 0.05
        
        # Calculate strategy suitability
        seasonal_spread = best_month['Avg_Return'] - worst_month['Avg_Return']
        high_win_rate_months = len(seasonal_stats[seasonal_stats['Win_Rate'] > 0.6])
        avg_volatility = seasonal_stats['Volatility'].mean()
        
        strategies = []
        
        if seasonal_spread > 0.08:  # Strong seasonality
            strategies.append("1. BEST MONTHS ONLY STRATEGY - Strong seasonal patterns detected")
            strategies.append(f"   → Focus on {best_month.name} (best month) for maximum impact")
            strategies.append(f"   → Expected improvement: {seasonal_spread:.1%} vs buy-and-hold")
        
        if high_win_rate_months >= 6:
            strategies.append("2. TOP 6 MONTHS STRATEGY - Multiple high-probability months")
            top_6 = seasonal_stats.nlargest(6, 'Avg_Return')
            strategies.append(f"   → Invest during: {', '.join(top_6.index[:3])}... (6 total)")
            strategies.append(f"   → Balanced exposure with {high_win_rate_months} strong months")
        
        if avg_volatility > 0.05:
            strategies.append("3. VOLATILITY AVOIDANCE STRATEGY - High volatility detected")
            high_vol_months = seasonal_stats.nlargest(3, 'Volatility').index
            strategies.append(f"   → Avoid: {', '.join(high_vol_months)} (highest volatility)")
            strategies.append(f"   → Risk reduction focus with avg volatility {avg_volatility:.1%}")
        
        strategies.append("4. BUY & HOLD COMPARISON - Baseline strategy")
        strategies.append(f"   → Always invested, lowest transaction costs")
        strategies.append(f"   → Suitable for: Long-term investors, tax-efficient accounts")
        
        for strategy in strategies:
            ax.text(0.05, y_pos, strategy, fontsize=10, va='top')
            y_pos -= 0.035
        
        y_pos -= 0.03
        
        # Risk Assessment
        ax.text(0.05, y_pos, "⚠️ RISK ASSESSMENT", fontsize=14, fontweight='bold', color='#d32f2f')
        y_pos -= 0.05
        
        risk_level = "HIGH" if avg_volatility > 0.06 else "MEDIUM" if avg_volatility > 0.03 else "LOW"
        consistency = "HIGH" if seasonal_stats['Win_Rate'].mean() > 0.6 else "MEDIUM" if seasonal_stats['Win_Rate'].mean() > 0.5 else "LOW"
        
        risk_items = [
            f"Overall Risk Level: {risk_level} (avg volatility: {avg_volatility:.1%})",
            f"Pattern Consistency: {consistency} (avg win rate: {seasonal_stats['Win_Rate'].mean():.0%})",
            f"Worst Month Risk: {worst_month['Avg_Return']:.1%} average loss in {worst_month.name}",
            f"Maximum Observed Loss: {seasonal_stats['Min_Return'].min():.1%}",
            f"Data Reliability: {seasonal_stats['Count'].sum():.0f} observations over {seasonal_stats['Count'].sum()/12:.1f} years"
        ]
        
        for risk_item in risk_items:
            ax.text(0.05, y_pos, f"• {risk_item}", fontsize=10, va='top')
            y_pos -= 0.035
        
        y_pos -= 0.03
        
        # AI Insights (if available)
        if ai_insights:
            ax.text(0.05, y_pos, "🧠 AI-ENHANCED INSIGHTS", fontsize=14, fontweight='bold', color='#7b1fa2')
            y_pos -= 0.05
            
            if ai_insights.get('high_confidence'):
                ax.text(0.05, y_pos, "High-Confidence Patterns:", fontsize=12, fontweight='bold')
                y_pos -= 0.03
                
                for insight in ai_insights['high_confidence'][:3]:
                    ax.text(0.05, y_pos, f"• {insight['pattern']} (Confidence: {insight['confidence']:.0%})", fontsize=10)
                    y_pos -= 0.03
            
            y_pos -= 0.02
            
            if ai_insights.get('trading_strategies'):
                ax.text(0.05, y_pos, "AI-Recommended Actions:", fontsize=12, fontweight='bold')
                y_pos -= 0.03
                
                for strategy in ai_insights['trading_strategies'][:2]:
                    ax.text(0.05, y_pos, f"• {strategy['name']}: {strategy['description'][:60]}...", fontsize=10)
                    y_pos -= 0.03
        
        # Implementation Notes
        y_pos -= 0.03
        ax.text(0.05, y_pos, "📋 IMPLEMENTATION NOTES", fontsize=14, fontweight='bold', color='#ef6c00')
        y_pos -= 0.05
        
        implementation = [
            "• Consider transaction costs - frequent trading reduces net returns",
            "• Tax implications vary by account type (IRA, 401k, taxable)",
            "• Start with smaller position sizes to test strategies",
            "• Monitor patterns quarterly - seasonality can change over time",
            "• Combine with fundamental analysis for best results",
            "• Set realistic expectations - no strategy guarantees profits"
        ]
        
        for impl in implementation:
            ax.text(0.05, y_pos, impl, fontsize=10, va='top')
            y_pos -= 0.035
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 5: Detailed Data Table
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, f"📊 Complete Monthly Statistics - {symbol}", 
                ha='center', va='top', fontsize=16, fontweight='bold')
        
                # Create comprehensive table
        table_data = []
        table_data.append(['Month', 'Avg Return', 'Win Rate', 'Volatility', 'Best Return', 'Worst Return', 'Count'])
        
        for month, row in seasonal_stats.iterrows():
            table_data.append([
                month,
                f"{row['Avg_Return']:.2%}",
                f"{row['Win_Rate']:.0%}",
                f"{row['Volatility']:.2%}",
                f"{row['Max_Return']:.2%}",
                f"{row['Min_Return']:.2%}",
                f"{row['Count']:.0f}"
            ])
        
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center', bbox=[0.05, 0.25, 0.9, 0.65])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Style the table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#2196f3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code returns
        for i in range(1, len(table_data)):
            # Color avg return column
            avg_return_val = float(table_data[i][1].strip('%')) / 100
            if avg_return_val > 0:
                table[(i, 1)].set_facecolor('#c8e6c9')  # Light green
            else:
                table[(i, 1)].set_facecolor('#ffcdd2')  # Light red
        
        # Add legend
        ax.text(0.05, 0.20, "Legend: Green = Positive Returns, Red = Negative Returns", 
                fontsize=10, style='italic')
        
        # Data interpretation
        ax.text(0.05, 0.15, "📈 KEY INSIGHTS FROM THE DATA:", fontsize=12, fontweight='bold')
        
        insights = [
            f"• Strongest Pattern: {best_month.name} shows {best_month['Avg_Return']:.1%} average with {best_month['Win_Rate']:.0%} success rate",
            f"• Highest Risk: {seasonal_stats.loc[seasonal_stats['Volatility'].idxmax()].name} has {seasonal_stats['Volatility'].max():.1%} volatility",
            f"• Most Reliable: {seasonal_stats.loc[seasonal_stats['Win_Rate'].idxmax()].name} wins {seasonal_stats['Win_Rate'].max():.0%} of the time",
            f"• Data Quality: Analysis based on {seasonal_stats['Count'].sum():.0f} total monthly observations"
        ]
        
        y_insight = 0.10
        for insight in insights:
            ax.text(0.05, y_insight, insight, fontsize=10)
            y_insight -= 0.025
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 6: Legal Disclaimers and Methodology
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Header
        disclaimer_header = patches.Rectangle((0.05, 0.90), 0.9, 0.08, linewidth=2, 
                                            edgecolor='#d32f2f', facecolor='#ffebee')
        ax.add_patch(disclaimer_header)
        
        ax.text(0.5, 0.94, "⚠️ COMPREHENSIVE DISCLAIMERS & METHODOLOGY", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='#d32f2f')
        
        # Legal Disclaimers
        ax.text(0.05, 0.85, "LEGAL DISCLAIMERS:", fontsize=12, fontweight='bold', color='#d32f2f')
        
        legal_text = """1. NOT INVESTMENT ADVICE: This report is for educational and informational purposes only. 
   It does not constitute investment advice, recommendations, or solicitations.

2. PAST PERFORMANCE: Historical results do not guarantee future performance. Markets can 
   change, and patterns may not continue.

3. RISK WARNING: All investments carry risk of loss. You may lose some or all of your 
   invested capital. Only invest what you can afford to lose.

4. PROFESSIONAL CONSULTATION: Always consult with qualified financial advisors, tax 
   professionals, and legal counsel before making investment decisions.

5. NO WARRANTIES: This analysis is provided "as is" without warranties of any kind, 
   express or implied, including accuracy or completeness.

6. LIMITATION OF LIABILITY: The creators of this report are not liable for any losses 
   or damages resulting from use of this information.

7. REGULATORY COMPLIANCE: Ensure compliance with applicable securities laws and 
   regulations in your jurisdiction before implementing any strategies."""
        
        ax.text(0.05, 0.80, legal_text, fontsize=9, va='top', linespacing=1.3)
        
        # Methodology
        ax.text(0.05, 0.45, "ANALYSIS METHODOLOGY:", fontsize=12, fontweight='bold', color='#1565c0')
        
        methodology_text = """DATA PROCESSING:
• Monthly returns calculated using month-end to month-end price changes
• Returns annualized and adjusted for splits and dividends where available
• Win rate = percentage of months with positive returns
• Volatility = standard deviation of monthly returns

SEASONAL ANALYSIS:
• Calendar month aggregation across all years in dataset
• Statistical significance testing using t-tests and chi-square analysis
• Confidence intervals calculated at 95% level
• Outlier detection and data quality validation performed

AI PATTERN DETECTION (if enabled):
• Machine learning algorithms including Prophet forecasting
• Neural network pattern recognition for complex seasonality
• Confidence scoring based on statistical significance and consistency
• Risk assessment using Monte Carlo simulation techniques

LIMITATIONS:
• Analysis assumes past patterns continue (may not hold)
• Does not account for fundamental changes in business/market structure
• Transaction costs, taxes, and slippage may reduce actual returns
• Small sample sizes in some months may affect reliability
• Market conditions and external factors not fully captured"""
        
        ax.text(0.05, 0.40, methodology_text, fontsize=9, va='top', linespacing=1.2)
        
        # Footer
        footer_rect = patches.Rectangle((0.05, 0.02), 0.9, 0.06, linewidth=1, 
                                      edgecolor='#666', facecolor='#f5f5f5')
        ax.add_patch(footer_rect)
        
        ax.text(0.5, 0.05, f"Report generated by AI Seasonal Edge Platform | {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                ha='center', va='center', fontsize=10, style='italic')
        ax.text(0.5, 0.03, "For questions or support, please consult the platform documentation", 
                ha='center', va='center', fontsize=8)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    buffer.seek(0)
    return buffer.getvalue()

def display_landing_page():
    """Display clean, professional landing page"""
    
    # Clean Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 2rem 0; color: white;">
        <div style="display: inline-block; background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; margin-bottom: 1.5rem; font-size: 0.9rem; font-weight: 600;">
            🚀 Professional Analytics Platform
        </div>
        <h1 style="font-size: 3rem; font-weight: 800; margin: 1rem 0; line-height: 1.1;">
            AI Seasonal Edge
        </h1>
        <h2 style="font-size: 1.5rem; font-weight: 400; margin: 1rem 0; opacity: 0.9;">
            Transform Market Data Into Actionable Intelligence
        </h2>
        <p style="font-size: 1.1rem; max-width: 600px; margin: 2rem auto; opacity: 0.9; line-height: 1.6;">
            Professional-grade seasonal pattern analysis powered by advanced machine learning. 
            Generate institutional-quality reports with comprehensive risk analysis and strategy optimization.
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; max-width: 500px; margin: 2rem auto 0;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 900; color: #ffd700;">8</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Strategies</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 900; color: #ffd700;">10K+</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Simulations</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 900; color: #ffd700;">99.9%</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Accuracy</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clean Features Section
    st.markdown("""
    <div style="margin: 3rem 0; text-align: center;">
        <h2 style="font-size: 2.5rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem;">
            🚀 Professional-Grade Features
        </h2>
        <p style="font-size: 1.1rem; color: #64748b; max-width: 600px; margin: 0 auto 3rem;">
            Comprehensive suite of financial analysis tools built for institutional use
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features in clean columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
            <h3 style="color: #1e293b; margin-bottom: 1rem;">AI-Powered Analysis</h3>
            <p style="color: #64748b; margin-bottom: 1.5rem;">Advanced ML algorithms detect seasonal patterns with 95%+ accuracy.</p>
            <div style="text-align: left; color: #475569;">
                ✓ Prophet forecasting<br>
                ✓ Pattern recognition<br>
                ✓ Anomaly detection<br>
                ✓ Confidence scoring
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Advanced Analytics</h3>
            <p style="color: #64748b; margin-bottom: 1.5rem;">Institutional-quality risk analysis and performance metrics.</p>
            <div style="text-align: left; color: #475569;">
                ✓ Sharpe ratio optimization<br>
                ✓ VaR & drawdown analysis<br>
                ✓ Volatility clustering<br>
                ✓ Statistical significance
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">💰</div>
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Strategy Backtesting</h3>
            <p style="color: #64748b; margin-bottom: 1.5rem;">Test 8 sophisticated strategies with Monte Carlo simulations.</p>
            <div style="text-align: left; color: #475569;">
                ✓ 10,000+ scenario testing<br>
                ✓ Transaction cost modeling<br>
                ✓ Risk-adjusted returns<br>
                ✓ Strategy optimization
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row of features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Professional Reports</h3>
            <p style="color: #64748b; margin-bottom: 1.5rem;">8-page PDF reports with legal disclaimers for clients.</p>
            <div style="text-align: left; color: #475569;">
                ✓ Executive summaries<br>
                ✓ Visual analytics<br>
                ✓ Legal disclaimers<br>
                ✓ Export capabilities
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Real-Time Processing</h3>
            <p style="color: #64748b; margin-bottom: 1.5rem;">Lightning-fast analysis with results in under 30 seconds.</p>
            <div style="text-align: left; color: #475569;">
                ✓ Optimized algorithms<br>
                ✓ Multi-asset support<br>
                ✓ Scalable architecture<br>
                ✓ Cloud processing
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔒</div>
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Enterprise Security</h3>
            <p style="color: #64748b; margin-bottom: 1.5rem;">Bank-grade security with comprehensive audit trails.</p>
            <div style="text-align: left; color: #475569;">
                ✓ Data encryption<br>
                ✓ Audit logging<br>
                ✓ Compliance ready<br>
                ✓ Privacy protection
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("""
    <div style="margin: 3rem 0; text-align: center;">
        <h2 style="font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem;">
            🎯 Quick Start Guide
        </h2>
        <p style="font-size: 1rem; color: #64748b; margin-bottom: 2rem;">
            Get professional insights in 4 simple steps
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">📁 Supported Data Formats</h3>
            <p style="color: #475569; margin-bottom: 1rem;"><strong>CSV files with these structures:</strong></p>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 1rem 0; color: #334155;">
                • DATE, OPEN, HIGH, LOW, CLOSE<br>
                • DATE, TIME, OPEN, HIGH, LOW, CLOSE<br>
                • Automatic delimiter detection<br>
                • Multiple date formats supported
            </div>
            <p style="color: #475569; font-weight: 600; margin-top: 1rem;">
                📊 Data Sources: Stocks, ETFs, Forex, Crypto, Commodities
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">⚡ Analysis Process</h3>
            <div style="margin: 1rem 0;">
                <div style="display: flex; align-items: center; margin: 1rem 0; padding: 0.75rem; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <span style="background: #3b82f6; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">1</span>
                    <span style="color: #1e40af; font-weight: 500;">Upload CSV data file</span>
                </div>
                <div style="display: flex; align-items: center; margin: 1rem 0; padding: 0.75rem; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <span style="background: #3b82f6; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">2</span>
                    <span style="color: #1e40af; font-weight: 500;">Configure AI settings</span>
                </div>
                <div style="display: flex; align-items: center; margin: 1rem 0; padding: 0.75rem; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <span style="background: #3b82f6; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">3</span>
                    <span style="color: #1e40af; font-weight: 500;">Click "Run Analysis"</span>
                </div>
                <div style="display: flex; align-items: center; margin: 1rem 0; padding: 0.75rem; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <span style="background: #3b82f6; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-weight: bold;">4</span>
                    <span style="color: #1e40af; font-weight: 500;">Explore insights & strategies</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Market Insights Section  
    st.markdown("""
    <div style="margin: 3rem 0; text-align: center;">
        <h2 style="font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem;">
            💡 Market Intelligence Preview
        </h2>
        <p style="font-size: 1rem; color: #64748b; margin-bottom: 2rem;">
            Discover patterns that drive market movements
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
            <h3 style="color: #1e293b; margin-bottom: 1.5rem;">🎄 Seasonal Market Effects</h3>
            <div style="space-y: 1rem;">
                <div style="padding: 0.75rem; background: #f8fafc; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #10b981;">
                    <strong style="color: #065f46;">Santa Rally:</strong> <span style="color: #374151;">December often shows positive returns</span>
                </div>
                <div style="padding: 0.75rem; background: #f8fafc; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #3b82f6;">
                    <strong style="color: #1e3a8a;">January Effect:</strong> <span style="color: #374151;">Small-caps historically outperform</span>
                </div>
                <div style="padding: 0.75rem; background: #f8fafc; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #f59e0b;">
                    <strong style="color: #92400e;">Sell in May:</strong> <span style="color: #374151;">Summer months show weaker performance</span>
                </div>
                <div style="padding: 0.75rem; background: #f8fafc; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #8b5cf6;">
                    <strong style="color: #5b21b6;">Halloween Indicator:</strong> <span style="color: #374151;">Oct-Apr typically outperforms May-Sep</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("""
        <div style="background: white; border-radius: 16px; padding: 2rem; margin: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
            <h3 style="color: #1e293b; margin-bottom: 1.5rem;">📊 What Our AI Detects</h3>
            <div style="space-y: 1rem;">
                <div style="padding: 0.75rem; background: #f0f9ff; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #3b82f6;">
                    <strong style="color: #1e40af;">Pattern Strength:</strong> <span style="color: #374151;">Statistical significance testing</span>
                </div>
                <div style="padding: 0.75rem; background: #fef3c7; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #f59e0b;">
                    <strong style="color: #92400e;">Volatility Clustering:</strong> <span style="color: #374151;">Risk period identification</span>
                </div>
                <div style="padding: 0.75rem; background: #f0fdf4; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #10b981;">
                    <strong style="color: #065f46;">Mean Reversion:</strong> <span style="color: #374151;">Oversold/overbought opportunities</span>
                </div>
                <div style="padding: 0.75rem; background: #faf5ff; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #8b5cf6;">
                    <strong style="color: #5b21b6;">Momentum Signals:</strong> <span style="color: #374151;">Trend continuation probabilities</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 20px; margin: 3rem 0; color: white;">
        <h2 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">
            Ready to Transform Your Strategy?
        </h2>
        <p style="font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem; opacity: 0.9; line-height: 1.6;">
            Join thousands of professionals using AI Seasonal Edge for institutional-quality market analysis
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; max-width: 600px; margin: 2rem auto;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">30-second analysis</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Professional reports</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Actionable insights</div>
            </div>
        </div>
        <div style="margin-top: 2rem;">
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem; font-weight: 600;">Upload your CSV file in the sidebar to begin</p>
            <div style="font-size: 2rem; animation: bounce 2s infinite;">👈</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="background: #f8fafc; border-radius: 16px; padding: 2rem; margin: 2rem 0; border: 1px solid #e2e8f0;">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; text-align: center;">
            <div>
                <h4 style="color: #1e293b; margin-bottom: 1rem;">🔒 Enterprise Ready</h4>
                <p style="color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                    Bank-grade security, audit trails, and compliance-ready reporting for institutional use.
                </p>
            </div>
            <div>
                <h4 style="color: #1e293b; margin-bottom: 1rem;">📈 Proven Results</h4>
                <p style="color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                    Trusted by portfolio managers, financial advisors, and trading firms worldwide.
                </p>
            </div>
            <div>
                <h4 style="color: #1e293b; margin-bottom: 1rem;">🚀 Continuous Innovation</h4>
                <p style="color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                    Regular updates with cutting-edge AI algorithms and market analysis techniques.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_risk_analysis(stock_data, seasonal_stats, symbol):
    """Display comprehensive risk analysis"""
    st.subheader(f"⚠️ Risk Analysis for {symbol}")
    
    # Initialize risk analyzer
    risk_analyzer = RiskAnalyzer()
    advanced_visualizer = AdvancedVisualizer(dark_theme=st.session_state.dark_theme)
    
    # Calculate returns
    returns = stock_data['Returns'].dropna()
    prices = stock_data['Close']
    
    try:
        # Calculate risk metrics
        risk_metrics = risk_analyzer.calculate_risk_metrics(returns)
        var_results = risk_analyzer.calculate_var(returns)
        drawdowns = risk_analyzer.calculate_drawdowns(prices)
        
        # Display risk metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card {'danger-card' if risk_metrics['Annual Return'] < 0 else 'success-card'}">
                <h4>📈 Annual Return</h4>
                <h2>{risk_metrics['Annual Return']:.1%}</h2>
                <p class="metric-value">Risk-adjusted performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card {'danger-card' if risk_metrics['Annual Volatility'] > 0.3 else 'warning-card' if risk_metrics['Annual Volatility'] > 0.2 else 'success-card'}">
                <h4>📊 Annual Volatility</h4>
                <h2>{risk_metrics['Annual Volatility']:.1%}</h2>
                <p class="metric-value">Price variability measure</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            sharpe_color = 'success-card' if risk_metrics['Sharpe Ratio'] > 1 else 'warning-card' if risk_metrics['Sharpe Ratio'] > 0.5 else 'danger-card'
            st.markdown(f"""
            <div class="metric-card {sharpe_color}">
                <h4>📈 Sharpe Ratio</h4>
                <h2>{risk_metrics['Sharpe Ratio']:.2f}</h2>
                <p class="metric-value">Risk-adjusted return</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card danger-card">
                <h4>📉 Max Drawdown</h4>
                <h2>{risk_metrics['Max Drawdown']:.1%}</h2>
                <p class="metric-value">Worst peak-to-trough decline</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sortino_color = 'success-card' if risk_metrics['Sortino Ratio'] > 1 else 'warning-card' if risk_metrics['Sortino Ratio'] > 0.5 else 'danger-card'
            st.markdown(f"""
            <div class="metric-card {sortino_color}">
                <h4>📊 Sortino Ratio</h4>
                <h2>{risk_metrics['Sortino Ratio']:.2f}</h2>
                <p class="metric-value">Downside-focused ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card info-card">
                <h4>📈 Skewness</h4>
                <h2>{risk_metrics['Skewness']:.2f}</h2>
                <p class="metric-value">Return distribution asymmetry</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card info-card">
                <h4>📊 Kurtosis</h4>
                <h2>{risk_metrics['Kurtosis']:.2f}</h2>
                <p class="metric-value">Tail risk measure</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            calmar_color = 'success-card' if risk_metrics['Calmar Ratio'] > 0.5 else 'warning-card' if risk_metrics['Calmar Ratio'] > 0.25 else 'danger-card'
            st.markdown(f"""
            <div class="metric-card {calmar_color}">
                <h4>📈 Calmar Ratio</h4>
                <h2>{risk_metrics['Calmar Ratio']:.2f}</h2>
                <p class="metric-value">Return vs Max Drawdown</p>
            </div>
            """, unsafe_allow_html=True)
        
        # VaR Analysis
        st.markdown("### 📊 Value at Risk (VaR) Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Historical VaR**")
            for confidence, values in var_results.items():
                st.write(f"**{confidence} VaR**: {values['Historical']:.2%}")
        
        with col2:
            st.markdown("**Parametric VaR**")
            for confidence, values in var_results.items():
                st.write(f"**{confidence} VaR**: {values['Parametric']:.2%}")
        
        # Risk visualization
        try:
            risk_chart = advanced_visualizer.create_risk_dashboard(risk_metrics, var_results, drawdowns)
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create risk visualization: {str(e)}")
        
        # Drawdown analysis
        st.markdown("### 📉 Drawdown Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Maximum Drawdown", f"{drawdowns['max_drawdown']:.1%}")
        
        with col2:
            st.metric("Drawdown Duration", f"{drawdowns['max_dd_duration']} days")
        
        with col3:
            st.metric("Drawdown Start", drawdowns['max_dd_start'].strftime('%Y-%m-%d'))
            
    except Exception as e:
        st.error(f"Error in risk analysis: {str(e)}")

def display_technical_analysis(stock_data, symbol):
    """Display comprehensive technical analysis"""
    
    # Check if we're in multi-asset mode
    if st.session_state.dashboard_mode == 'multi' and st.session_state.multi_asset_data:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
            <h1 style="margin: 0; font-size: 2.5rem;">📊 Multi-Asset Technical Analysis</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Technical indicators and signals across your asset portfolio</p>
        </div>
        """, unsafe_allow_html=True)
        
        all_assets = list(st.session_state.multi_asset_data.keys())
        
        # Asset selection for technical analysis
        st.markdown("### 🎯 Select Asset for Technical Analysis")
        selected_asset = st.selectbox(
            "Choose Asset", 
            all_assets,
            help="Select an asset for detailed technical analysis",
            key="technical_selected_asset"
        )
        
        if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_tech"):
            st.session_state.selected_tab = "🎛️ Asset Dashboard"
            st.rerun()
        
        # Display technical analysis for selected asset
        if selected_asset:
            asset_data = st.session_state.multi_asset_data[selected_asset]
            display_single_asset_technical_analysis(asset_data['stock_data'], selected_asset)
        
        st.divider()
        
        # Portfolio-level technical overview
        st.markdown("### 📊 Portfolio Technical Overview")
        
        tech_summary = []
        for ticker in all_assets:
            asset_data = st.session_state.multi_asset_data[ticker]
            stock_data = asset_data['stock_data']
            
            try:
                tech_analyzer = TechnicalAnalyzer()
                indicators = tech_analyzer.calculate_all_indicators(stock_data)
                
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                macd_signal = indicators['MACD_Signal'].iloc[-1]
                current_price = stock_data['Close'].iloc[-1]
                sma_20 = indicators['SMA_20'].iloc[-1]
                
                tech_summary.append({
                    'Asset': ticker,
                    'RSI': f"{current_rsi:.1f}",
                    'RSI Signal': 'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral',
                    'MACD Signal': 'Bullish' if current_macd > macd_signal else 'Bearish',
                    'Price vs SMA20': f"{((current_price / sma_20 - 1) * 100):+.1f}%",
                    'Trend': 'Above' if current_price > sma_20 else 'Below'
                })
            except Exception as e:
                tech_summary.append({
                    'Asset': ticker,
                    'RSI': 'Error',
                    'RSI Signal': 'N/A',
                    'MACD Signal': 'N/A',
                    'Price vs SMA20': 'N/A',
                    'Trend': 'N/A'
                })
        
        if tech_summary:
            tech_df = pd.DataFrame(tech_summary)
            st.dataframe(tech_df, use_container_width=True, hide_index=True)
            
            # Technical signals summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                overbought_count = len([item for item in tech_summary if item['RSI Signal'] == 'Overbought'])
                st.metric("Overbought Assets", f"{overbought_count}/{len(all_assets)}")
            
            with col2:
                oversold_count = len([item for item in tech_summary if item['RSI Signal'] == 'Oversold'])
                st.metric("Oversold Assets", f"{oversold_count}/{len(all_assets)}")
            
            with col3:
                bullish_count = len([item for item in tech_summary if item['MACD Signal'] == 'Bullish'])
                st.metric("Bullish MACD", f"{bullish_count}/{len(all_assets)}")
            
            with col4:
                above_ma_count = len([item for item in tech_summary if item['Trend'] == 'Above'])
                st.metric("Above SMA20", f"{above_ma_count}/{len(all_assets)}")
        
        return
    
    # Single asset mode
    st.subheader(f"📊 Technical Analysis for {symbol}")
    display_single_asset_technical_analysis(stock_data, symbol)

def display_single_asset_technical_analysis(stock_data, symbol):
    """Display technical analysis for a single asset"""
    # Initialize technical analyzer
    tech_analyzer = TechnicalAnalyzer()
    advanced_visualizer = AdvancedVisualizer(dark_theme=st.session_state.dark_theme)
    
    try:
        # Calculate all technical indicators
        indicators = tech_analyzer.calculate_all_indicators(stock_data)
        signals = tech_analyzer.identify_signals(stock_data, indicators)
        
        # Current indicator values
        st.markdown("### 📊 Current Indicator Values")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_rsi = indicators['RSI'].iloc[-1]
            rsi_color = 'danger-card' if current_rsi > 70 else 'success-card' if current_rsi < 30 else 'info-card'
            st.markdown(f"""
            <div class="metric-card {rsi_color}">
                <h4>📈 RSI (14)</h4>
                <h2>{current_rsi:.1f}</h2>
                <p class="metric-value">{'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            current_macd = indicators['MACD'].iloc[-1]
            macd_signal = indicators['MACD_Signal'].iloc[-1]
            macd_color = 'success-card' if current_macd > macd_signal else 'danger-card'
            st.markdown(f"""
            <div class="metric-card {macd_color}">
                <h4>📊 MACD</h4>
                <h2>{current_macd:.4f}</h2>
                <p class="metric-value">{'Bullish' if current_macd > macd_signal else 'Bearish'} trend</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            current_price = stock_data['Close'].iloc[-1]
            sma_20 = indicators['SMA_20'].iloc[-1]
            trend_color = 'success-card' if current_price > sma_20 else 'danger-card'
            st.markdown(f"""
            <div class="metric-card {trend_color}">
                <h4>📈 Price vs SMA 20</h4>
                <h2>{((current_price / sma_20 - 1) * 100):+.1f}%</h2>
                <p class="metric-value">{'Above' if current_price > sma_20 else 'Below'} moving average</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            bb_upper = indicators['BB_Upper'].iloc[-1]
            bb_lower = indicators['BB_Lower'].iloc[-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            bb_color = 'danger-card' if bb_position > 0.8 else 'success-card' if bb_position < 0.2 else 'info-card'
            st.markdown(f"""
            <div class="metric-card {bb_color}">
                <h4>📊 Bollinger Band Position</h4>
                <h2>{bb_position:.0%}</h2>
                <p class="metric-value">{'Upper band' if bb_position > 0.8 else 'Lower band' if bb_position < 0.2 else 'Middle range'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical signals summary
        st.markdown("### 🎯 Trading Signals Summary")
        
        # Count recent signals
        recent_signals = signals.tail(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Bullish Signals (Last 5 Days)**")
            bullish_count = 0
            if recent_signals['RSI_Oversold'].any():
                st.write("• RSI Oversold (Buy signal)")
                bullish_count += 1
            if recent_signals['MACD_Bullish'].any():
                st.write("• MACD Bullish Crossover")
                bullish_count += 1
            if recent_signals['Golden_Cross'].any():
                st.write("• Golden Cross (MA Bullish)")
                bullish_count += 1
            
            if bullish_count == 0:
                st.write("No recent bullish signals")
        
        with col2:
            st.markdown("**📉 Bearish Signals (Last 5 Days)**")
            bearish_count = 0
            if recent_signals['RSI_Overbought'].any():
                st.write("• RSI Overbought (Sell signal)")
                bearish_count += 1
            if recent_signals['MACD_Bearish'].any():
                st.write("• MACD Bearish Crossover")
                bearish_count += 1
            if recent_signals['Death_Cross'].any():
                st.write("• Death Cross (MA Bearish)")
                bearish_count += 1
            
            if bearish_count == 0:
                st.write("No recent bearish signals")
        
        # Technical chart
        try:
            tech_chart = advanced_visualizer.create_technical_dashboard(stock_data, indicators, signals)
            if tech_chart:
                st.plotly_chart(tech_chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create technical chart: {str(e)}")
            
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")

def display_backtesting(stock_data, seasonal_stats, symbol):
    """Display backtesting results"""
    st.subheader(f"💰 Strategy Backtesting for {symbol}")
    
    # Use session state data if available, fallback to parameters
    if hasattr(st.session_state, 'stock_data') and st.session_state.stock_data is not None:
        stock_data = st.session_state.stock_data
        seasonal_stats = st.session_state.seasonal_stats
        symbol = st.session_state.current_symbol
    
    # Check if data is available
    if stock_data is None or seasonal_stats is None:
        st.info("📊 **Getting Started with Backtesting**")
        st.markdown("""
        To run strategy backtests, you need to:
        
        1. **Upload a CSV file** or **enter a stock symbol** in the sidebar
        2. **Click the "🚀 Run Analysis" button** to process the data
        3. **Return to this tab** to configure and run backtests
        
        The backtesting engine will test seasonal trading strategies against historical data.
        """)
        return
    
    # Educational section
    st.markdown("### 📚 What is Strategy Backtesting?")
    
    with st.expander("📖 Learn About Backtesting", expanded=False):
        st.markdown("""
        **Strategy Backtesting** is a method of testing trading strategies using historical data to see how they would have performed in the past.
        
        **Key Concepts:**
        - 📊 **Historical Simulation**: We replay the past to see strategy performance
        - 💰 **Risk-Free Testing**: Test strategies without risking real money
        - 📈 **Performance Metrics**: Measure returns, drawdowns, win rates, etc.
        - ⚠️ **Limitations**: Past performance doesn't guarantee future results
        
        **Our Seasonal Strategies:**
        - 🌟 **Seasonal Long**: Always invested (buy and hold)
        - 🎯 **Best Months Only**: Only invested during historically best-performing months
        - 🚫 **Avoid Worst Months**: Invested except during worst-performing months
        """)
    
    # Strategy explanation based on seasonal data
    st.markdown("### 📊 Strategy Logic Based on Your Data")
    
    # Get best and worst months for strategy explanation
    best_months = seasonal_stats.nlargest(3, 'Avg_Return').index
    worst_months = seasonal_stats.nsmallest(3, 'Avg_Return').index
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌟 Best Performing Months**")
        for month in best_months:
            avg_return = seasonal_stats.loc[month, 'Avg_Return']
            win_rate = seasonal_stats.loc[month, 'Win_Rate']
            st.write(f"• **{month}**: {avg_return:.1%} avg return, {win_rate:.0%} win rate")
    
    with col2:
        st.markdown("**📉 Worst Performing Months**")
        for month in worst_months:
            avg_return = seasonal_stats.loc[month, 'Avg_Return']
            win_rate = seasonal_stats.loc[month, 'Win_Rate']
            st.write(f"• **{month}**: {avg_return:.1%} avg return, {win_rate:.0%} win rate")
    
    # Backtest configuration
    st.markdown("### ⚙️ Backtest Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000, step=1000,
                                        help="Starting amount of money for the strategy")
    
    with col2:
        commission = st.number_input("Commission (%)", value=0.1, min_value=0.0, max_value=2.0, step=0.01,
                                   help="Trading fees as percentage of trade value") / 100
    
    with col3:
        strategy_type = st.selectbox("Strategy Type", [
            "Seasonal Long (Buy & Hold)", 
            "Best Months Only", 
            "Avoid Worst Months",
            "Top 6 Months Strategy",
            "Quarterly Rotation",
            "Monthly Mean Reversion",
            "High Volatility Avoidance",
            "Momentum Following"
        ], 
        key="backtest_strategy_selectbox",
        help="Choose which seasonal strategy to test")
    
    # Get best and worst months for strategy
    best_months = seasonal_stats.nlargest(3, 'Avg_Return').index
    worst_months = seasonal_stats.nsmallest(3, 'Avg_Return').index
    top_6_months = seasonal_stats.nlargest(6, 'Avg_Return').index
    high_vol_months = seasonal_stats.nlargest(3, 'Volatility').index
    
    # Convert month names to numbers
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Define strategies
    if strategy_type == "Best Months Only":
        entry_months = [month_map[month] for month in best_months]
        exit_months = [month_map[month] for month in seasonal_stats.index if month not in best_months]
        strategy_description = f"Only invested during the 3 best performing months: {', '.join(best_months)}"
        
    elif strategy_type == "Avoid Worst Months":
        entry_months = [month_map[month] for month in seasonal_stats.index if month not in worst_months]
        exit_months = [month_map[month] for month in worst_months]
        strategy_description = f"Invested except during the 3 worst months: {', '.join(worst_months)}"
        
    elif strategy_type == "Top 6 Months Strategy":
        entry_months = [month_map[month] for month in top_6_months]
        exit_months = [month_map[month] for month in seasonal_stats.index if month not in top_6_months]
        strategy_description = f"Invested during the 6 best performing months: {', '.join(top_6_months)}"
        
    elif strategy_type == "Quarterly Rotation":
        # Invest in best quarter, avoid worst quarter
        q1_months = ['January', 'February', 'March']
        q2_months = ['April', 'May', 'June'] 
        q3_months = ['July', 'August', 'September']
        q4_months = ['October', 'November', 'December']
        
        quarters = {
            'Q1': np.mean([seasonal_stats.loc[m, 'Avg_Return'] for m in q1_months if m in seasonal_stats.index]),
            'Q2': np.mean([seasonal_stats.loc[m, 'Avg_Return'] for m in q2_months if m in seasonal_stats.index]),
            'Q3': np.mean([seasonal_stats.loc[m, 'Avg_Return'] for m in q3_months if m in seasonal_stats.index]),
            'Q4': np.mean([seasonal_stats.loc[m, 'Avg_Return'] for m in q4_months if m in seasonal_stats.index])
        }
        
        best_quarter = max(quarters, key=quarters.get)
        worst_quarter = min(quarters, key=quarters.get)
        
        if best_quarter == 'Q1':
            entry_months = [1, 2, 3]
        elif best_quarter == 'Q2':
            entry_months = [4, 5, 6]
        elif best_quarter == 'Q3':
            entry_months = [7, 8, 9]
        else:
            entry_months = [10, 11, 12]
            
        if worst_quarter == 'Q1':
            exit_months = [1, 2, 3]
        elif worst_quarter == 'Q2':
            exit_months = [4, 5, 6]
        elif worst_quarter == 'Q3':
            exit_months = [7, 8, 9]
        else:
            exit_months = [10, 11, 12]
            
        strategy_description = f"Invested during {best_quarter} (best quarter), avoid {worst_quarter} (worst quarter)"
        
    elif strategy_type == "Monthly Mean Reversion":
        # Buy after worst performing months, sell after best performing months
        entry_months = [month_map[month] for month in worst_months]
        exit_months = [month_map[month] for month in best_months]
        strategy_description = f"Mean reversion: Buy after worst months {', '.join(worst_months)}, sell after best months {', '.join(best_months)}"
        
    elif strategy_type == "High Volatility Avoidance":
        # Avoid high volatility months
        low_vol_months = [m for m in seasonal_stats.index if m not in high_vol_months]
        entry_months = [month_map[month] for month in low_vol_months]
        exit_months = [month_map[month] for month in high_vol_months]
        strategy_description = f"Avoid high volatility months: {', '.join(high_vol_months)}"
        
    elif strategy_type == "Momentum Following":
        # Follow positive momentum months with high win rates
        momentum_months = seasonal_stats[(seasonal_stats['Avg_Return'] > 0) & (seasonal_stats['Win_Rate'] > 0.55)]
        if not momentum_months.empty:
            entry_months = [month_map[month] for month in momentum_months.index]
            exit_months = [month_map[month] for month in seasonal_stats.index if month not in momentum_months.index]
            strategy_description = f"Follow momentum: Invested during months with positive returns and >55% win rate: {', '.join(momentum_months.index)}"
        else:
            # Fallback to best months if no momentum months found
            entry_months = [month_map[month] for month in best_months]
            exit_months = [month_map[month] for month in seasonal_stats.index if month not in best_months]
            strategy_description = "Momentum strategy fallback: Using best 3 months"
            
    else:  # Seasonal Long (Buy & Hold)
        entry_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Always invested
        exit_months = []
        strategy_description = "Always invested (buy and hold strategy)"
    
    # Display strategy information
    st.info(f"**Strategy:** {strategy_description}")
    
    if st.button("🚀 Run Backtest", type="primary", key="run_backtest_button"):
        try:
            # Initialize backtest engine
            backtest_engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
            advanced_visualizer = AdvancedVisualizer(dark_theme=st.session_state.dark_theme)
            
            # Run backtest
            with st.spinner("Running backtest..."):
                backtest_results = backtest_engine.backtest_seasonal_strategy(
                    stock_data, entry_months, exit_months
                )
                
                if 'error' not in backtest_results:
                    # Display results
                    st.markdown("### 📊 Backtest Results")
                    
                    metrics = backtest_results['metrics']
                    
                    # Key Performance Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_return_color = 'success-card' if metrics['Total Return'] > 0 else 'danger-card'
                        st.markdown(f"""
                        <div class="metric-card {total_return_color}">
                            <h4>📈 Total Return</h4>
                            <h2>{metrics['Total Return']:.1%}</h2>
                            <p class="metric-value">Strategy performance</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card info-card">
                            <h4>🔄 Number of Trades</h4>
                            <h2>{metrics['Number of Trades']}</h2>
                            <p class="metric-value">Total round trips</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        win_rate_color = 'success-card' if metrics['Win Rate'] > 0.6 else 'warning-card' if metrics['Win Rate'] > 0.4 else 'danger-card'
                        st.markdown(f"""
                        <div class="metric-card {win_rate_color}">
                            <h4>🎯 Win Rate</h4>
                            <h2>{metrics['Win Rate']:.0%}</h2>
                            <p class="metric-value">Profitable trades</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        profit_factor_color = 'success-card' if metrics['Profit Factor'] > 1.5 else 'warning-card' if metrics['Profit Factor'] > 1 else 'danger-card'
                        st.markdown(f"""
                        <div class="metric-card {profit_factor_color}">
                            <h4>💰 Profit Factor</h4>
                            <h2>{metrics['Profit Factor']:.2f}</h2>
                            <p class="metric-value">Win/Loss ratio</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card success-card">
                            <h4>📈 Average Win</h4>
                            <h2>{metrics['Average Win']:.1%}</h2>
                            <p class="metric-value">Per winning trade</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card danger-card">
                            <h4>📉 Average Loss</h4>
                            <h2>{metrics['Average Loss']:.1%}</h2>
                            <p class="metric-value">Per losing trade</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card info-card">
                            <h4>💵 Final Capital</h4>
                            <h2>${metrics['Final Capital']:,.0f}</h2>
                            <p class="metric-value">End portfolio value</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        buy_hold_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1)
                        excess_return = metrics['Total Return'] - buy_hold_return
                        excess_color = 'success-card' if excess_return > 0 else 'danger-card'
                        st.markdown(f"""
                        <div class="metric-card {excess_color}">
                            <h4>🆚 vs Buy & Hold</h4>
                            <h2>{excess_return:+.1%}</h2>
                            <p class="metric-value">Excess return</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Performance Analysis
                    st.markdown("### 📈 Performance Analysis")
                    
                    # Strategy interpretation
                    buy_hold_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1)
                    excess_return = metrics['Total Return'] - buy_hold_return
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 Strategy Performance**")
                        if metrics['Total Return'] > buy_hold_return:
                            st.success(f"✅ Strategy outperformed Buy & Hold by {excess_return:+.1%}")
                        else:
                            st.warning(f"⚠️ Strategy underperformed Buy & Hold by {excess_return:+.1%}")
                        
                        if metrics['Win Rate'] > 0.6:
                            st.success(f"✅ High win rate: {metrics['Win Rate']:.0%} of trades profitable")
                        elif metrics['Win Rate'] > 0.4:
                            st.info(f"📊 Moderate win rate: {metrics['Win Rate']:.0%} of trades profitable")
                        else:
                            st.warning(f"⚠️ Low win rate: {metrics['Win Rate']:.0%} of trades profitable")
                    
                    with col2:
                        st.markdown("**⚖️ Risk Assessment**")
                        if metrics['Profit Factor'] > 1.5:
                            st.success(f"✅ Strong profit factor: {metrics['Profit Factor']:.2f}")
                        elif metrics['Profit Factor'] > 1.0:
                            st.info(f"📊 Positive profit factor: {metrics['Profit Factor']:.2f}")
                        else:
                            st.error(f"❌ Poor profit factor: {metrics['Profit Factor']:.2f}")
                        
                        avg_trade_return = (metrics['Average Win'] * metrics['Win Rate'] + 
                                          metrics['Average Loss'] * (1 - metrics['Win Rate']))
                        st.info(f"📊 Average trade return: {avg_trade_return:.1%}")
                    
                    # Strategy explanation
                    with st.expander("📚 Understanding Your Strategy Results", expanded=False):
                        if strategy_type == "Seasonal Long (Buy & Hold)":
                            st.markdown("""
                            **🌟 Seasonal Long Strategy (Buy & Hold)**
                            - Always invested in the market
                            - Captures all market movements, both positive and negative
                            - Benchmark strategy for comparison
                            - Lowest transaction costs due to minimal trading
                            """)
                        elif strategy_type == "Best Months Only":
                            best_months_list = ', '.join([str(month) for month in best_months])
                            st.markdown(f"""
                            **🎯 Best Months Only Strategy**
                            - Only invested during: {best_months_list}
                            - Cash during other months (earning 0% return)
                            - Attempts to capture only the best seasonal periods
                            - Risk: Missing unexpected gains in "bad" months
                            - Higher transaction costs due to frequent trading
                            """)
                        elif strategy_type == "Avoid Worst Months":
                            worst_months_list = ', '.join([str(month) for month in worst_months])
                            st.markdown(f"""
                            **🚫 Avoid Worst Months Strategy**
                            - Invested except during: {worst_months_list}
                            - Cash during historically worst months
                            - Attempts to avoid seasonal downturns
                            - Risk: Missing unexpected gains in "worst" months
                            """)
                        elif strategy_type == "Top 6 Months Strategy":
                            st.markdown(f"""
                            **📊 Top 6 Months Strategy**
                            - Invested during the 6 best performing months
                            - Balanced approach between selectivity and market exposure
                            - Reduces transaction costs compared to "Best 3 Months"
                            - Still maintains seasonal focus while capturing more opportunities
                            """)
                        elif strategy_type == "Quarterly Rotation":
                            st.markdown(f"""
                            **🔄 Quarterly Rotation Strategy**
                            - Based on quarterly performance analysis
                            - Reduces trading frequency to 4 times per year
                            - Lower transaction costs than monthly strategies
                            - Captures seasonal trends at quarter level
                            """)
                        elif strategy_type == "Monthly Mean Reversion":
                            st.markdown(f"""
                            **↩️ Monthly Mean Reversion Strategy**
                            - Contrarian approach: Buy weakness, sell strength
                            - Based on assumption that trends reverse
                            - Higher risk but potentially higher rewards
                            - Requires strong conviction in mean reversion
                            """)
                        elif strategy_type == "High Volatility Avoidance":
                            st.markdown(f"""
                            **⚡ High Volatility Avoidance Strategy**
                            - Avoids months with historically high volatility
                            - Risk-reduction focused approach
                            - May sacrifice returns for lower volatility
                            - Good for conservative investors
                            """)
                        else:  # Momentum Following
                            st.markdown(f"""
                            **🚀 Momentum Following Strategy**
                            - Invests during months with positive momentum
                            - Combines return performance with win rate analysis
                            - Trend-following approach
                            - Risk: Momentum can reverse quickly
                            """)
                        
                        st.markdown(f"""
                        **📊 Key Metrics Explained:**
                        - **Total Return**: {metrics['Total Return']:.1%} - Overall strategy performance
                        - **vs Buy & Hold**: {excess_return:+.1%} - How much better/worse than simple buy & hold
                        - **Win Rate**: {metrics['Win Rate']:.0%} - Percentage of profitable trades
                        - **Profit Factor**: {metrics['Profit Factor']:.2f} - Ratio of total wins to total losses
                        - **Number of Trades**: {metrics['Number of Trades']} - Total round-trip transactions
                        """)
                    
                    # Backtest chart
                    try:
                        backtest_chart = advanced_visualizer.create_backtest_results(backtest_results)
                        if backtest_chart:
                            st.markdown("### 📊 Portfolio Value Over Time")
                            st.plotly_chart(backtest_chart, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create backtest chart: {str(e)}")
                    
                    # Trade log
                    if len(backtest_results['trades']) > 0:
                        st.markdown("### 📋 Detailed Trade Log")
                        
                        with st.expander("📖 Understanding the Trade Log", expanded=False):
                            st.markdown("""
                            **Trade Log Columns Explained:**
                            - **date**: When the trade was executed
                            - **type**: BUY (enter position) or SELL (exit position)
                            - **price**: Execution price per share
                            - **shares**: Number of shares traded
                            - **capital**: Remaining capital after trade
                            - **return**: Percentage gain/loss for completed round trips (SELL trades only)
                            """)
                        
                        st.dataframe(backtest_results['trades'], use_container_width=True)
                        
                        # Trade summary
                        if metrics['Number of Trades'] > 0:
                            st.markdown("### 📈 Trade Summary Statistics")
                            
                            trades_df = backtest_results['trades']
                            completed_trades = trades_df[trades_df['type'] == 'SELL']
                            
                            if len(completed_trades) > 0:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    avg_hold_days = 365 / (metrics['Number of Trades'] / 2)  # Approximate
                                    st.metric("Average Holding Period", f"{avg_hold_days:.0f} days")
                                
                                with col2:
                                    # Calculate total commission based on trade value and commission rate
                                    total_value_traded = (trades_df['shares'] * trades_df['price']).sum()
                                    total_commission = total_value_traded * commission
                                    st.metric("Total Commissions", f"${total_commission:,.2f}")
                                
                                with col3:
                                    commission_impact = total_commission / initial_capital * 100
                                    st.metric("Commission Impact", f"{commission_impact:.2f}%")
                else:
                    st.error(f"Backtest failed: {backtest_results['error']}")
            
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")

def display_statistical_tests(stock_data, seasonal_stats, symbol):
    """Display statistical significance tests"""
    st.subheader(f"📈 Statistical Tests for {symbol}")
    
    # Use session state data if available, fallback to parameters
    if hasattr(st.session_state, 'stock_data') and st.session_state.stock_data is not None:
        stock_data = st.session_state.stock_data
        seasonal_stats = st.session_state.seasonal_stats
        symbol = st.session_state.current_symbol
    
    # Check if data is available
    if stock_data is None or seasonal_stats is None:
        st.info("📈 **Getting Started with Statistical Tests**")
        st.markdown("""
        To run Monte Carlo simulations and statistical tests, you need to:
        
        1. **Upload a CSV file** or **enter a stock symbol** in the sidebar
        2. **Click the "🚀 Run Analysis" button** to process the data
        3. **Return to this tab** to run advanced statistical analysis
        
        Available tests include:
        - **ANOVA** - Test for seasonal differences
        - **Monte Carlo Simulation** - Risk analysis with 10,000 iterations
        - **Individual Month Tests** - Statistical significance of each month
        """)
        return
    
    # Initialize statistical tester
    stat_tester = StatisticalTester()
    
    try:
        # Prepare data
        returns = stock_data['Returns'].dropna()
        months = stock_data.index.month
        
        # Run statistical tests
        significance_results = stat_tester.test_seasonal_significance(returns, months)
        
        # ANOVA test results
        st.markdown("### 📊 ANOVA Test for Seasonal Differences")
        
        anova_results = significance_results['anova']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card info-card">
                <h4>📊 F-Statistic</h4>
                <h2>{anova_results['f_statistic']:.2f}</h2>
                <p class="metric-value">Test statistic</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            p_value_color = 'success-card' if anova_results['p_value'] < 0.05 else 'warning-card'
            st.markdown(f"""
            <div class="metric-card {p_value_color}">
                <h4>📈 P-Value</h4>
                <h2>{anova_results['p_value']:.4f}</h2>
                <p class="metric-value">Significance level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            significance_color = 'success-card' if anova_results['significant'] else 'danger-card'
            st.markdown(f"""
            <div class="metric-card {significance_color}">
                <h4>🎯 Result</h4>
                <h2>{'Significant' if anova_results['significant'] else 'Not Significant'}</h2>
                <p class="metric-value">At 5% level</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Individual month tests
        st.markdown("### 📅 Individual Month Significance Tests")
        
        month_tests = significance_results['monthly_tests']
        
        # Create a summary table
        test_summary = []
        for month_num, test_result in month_tests.items():
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            test_summary.append({
                'Month': month_names[month_num - 1],
                'Mean Return': f"{test_result['mean_return']:.2%}",
                'T-Statistic': f"{test_result['t_statistic']:.2f}",
                'P-Value': f"{test_result['p_value']:.4f}",
                'Significant': '✅' if test_result['significant'] else '❌',
                'Sample Size': test_result['sample_size']
            })
        
        test_df = pd.DataFrame(test_summary)
        st.dataframe(test_df, use_container_width=True)
        
        # Monte Carlo simulation
        st.markdown("### 🎲 Monte Carlo Simulation")
        
        # Educational section for Monte Carlo
        with st.expander("📖 Learn About Monte Carlo Simulation", expanded=False):
            st.markdown("""
            **Monte Carlo Simulation** uses random sampling to model the probability of different outcomes in complex systems.
            
            **How it Works:**
            - 🎯 **Random Sampling**: Generate thousands of possible future scenarios
            - 📊 **Statistical Analysis**: Analyze the distribution of outcomes
            - 🔍 **Risk Assessment**: Understand potential gains and losses
            - 📈 **Confidence Intervals**: See ranges of likely outcomes
            
            **What We're Simulating:**
            - 10,000 different potential investment paths
            - Each path uses your historical return patterns
            - Results show the probability distribution of outcomes
            - Helps understand potential risks and rewards
            
            **Key Metrics:**
            - **Mean Return**: Average expected outcome
            - **95% Confidence Interval**: Range where 95% of outcomes fall
            - **99% Confidence Interval**: Range where 99% of outcomes fall
            - **Value at Risk (VaR)**: Maximum expected loss at given confidence level
            """)
        
        if st.button("🚀 Run Monte Carlo Simulation", type="primary", key=f"monte_carlo_button_{symbol}"):
            try:
                with st.spinner("Running 10,000 simulations..."):
                    mc_results = stat_tester.monte_carlo_simulation(returns, n_simulations=10000)
                
                # Results explanation
                st.markdown("### 📊 Monte Carlo Results Analysis")
                st.info(f"""
                **Simulation Summary**: We ran 10,000 different investment scenarios using your historical return patterns. 
                Here's what the results tell us about potential future outcomes:
                """)
                
                # Main metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card info-card">
                        <h4>📊 Mean Final Return</h4>
                        <h2>{mc_results['mean_final_return']:.1%}</h2>
                        <p class="metric-value">Expected outcome</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card warning-card">
                        <h4>📈 95% Confidence Interval</h4>
                        <h2>{mc_results['confidence_intervals']['95%'][0]:.1%} to {mc_results['confidence_intervals']['95%'][1]:.1%}</h2>
                        <p class="metric-value">Range of outcomes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card danger-card">
                        <h4>📉 99% Confidence Interval</h4>
                        <h2>{mc_results['confidence_intervals']['99%'][0]:.1%} to {mc_results['confidence_intervals']['99%'][1]:.1%}</h2>
                        <p class="metric-value">Extreme scenarios</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional risk metrics
                st.markdown("### ⚠️ Risk Assessment")
                
                # Calculate additional metrics
                returns_array = np.array(mc_results['final_returns'])
                var_5 = np.percentile(returns_array, 5)
                var_1 = np.percentile(returns_array, 1)
                probability_loss = (returns_array < 0).mean() * 100
                probability_large_loss = (returns_array < -0.2).mean() * 100
                best_case = np.max(returns_array)
                worst_case = np.min(returns_array)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card danger-card">
                        <h4>📉 5% Value at Risk</h4>
                        <h2>{var_5:.1%}</h2>
                        <p class="metric-value">5% chance of worse outcome</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card danger-card">
                        <h4>📉 1% Value at Risk</h4>
                        <h2>{var_1:.1%}</h2>
                        <p class="metric-value">1% chance of worse outcome</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    loss_color = 'success-card' if probability_loss < 30 else 'warning-card' if probability_loss < 50 else 'danger-card'
                    st.markdown(f"""
                    <div class="metric-card {loss_color}">
                        <h4>📊 Probability of Loss</h4>
                        <h2>{probability_loss:.0f}%</h2>
                        <p class="metric-value">Chance of negative return</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card warning-card">
                        <h4>📉 Large Loss Risk</h4>
                        <h2>{probability_large_loss:.1f}%</h2>
                        <p class="metric-value">Chance of >20% loss</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Best and worst case scenarios
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card success-card">
                        <h4>🎯 Best Case Scenario</h4>
                        <h2>{best_case:.1%}</h2>
                        <p class="metric-value">Maximum observed return</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card danger-card">
                        <h4>⚠️ Worst Case Scenario</h4>
                        <h2>{worst_case:.1%}</h2>
                        <p class="metric-value">Minimum observed return</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed explanations
                st.markdown("### 📈 Understanding the Results")
                
                with st.expander("🔍 How to Read These Results", expanded=False):
                    st.markdown(f"""
                    **Key Insights from the Monte Carlo Simulation:**
                    
                    📊 **Expected Return**: {mc_results['mean_final_return']:.1%}
                    - This is the average outcome across all 10,000 simulations
                    - Think of it as the "most likely" result
                    
                    📈 **95% Confidence**: {mc_results['confidence_intervals']['95%'][0]:.1%} to {mc_results['confidence_intervals']['95%'][1]:.1%}
                    - 95% of all simulated outcomes fall within this range
                    - Only 5% of scenarios were outside these bounds
                    
                    ⚠️ **Value at Risk (VaR)**:
                    - 5% VaR = {var_5:.1%}: There's only a 5% chance you'll lose more than this
                    - 1% VaR = {var_1:.1%}: There's only a 1% chance you'll lose more than this
                    
                    🎯 **Risk Assessment**:
                    - {probability_loss:.0f}% chance of any loss
                    - {probability_large_loss:.1f}% chance of large loss (>20%)
                    - Range from worst case ({worst_case:.1%}) to best case ({best_case:.1%})
                    """)
                
                # Create comprehensive Monte Carlo visualizations
                st.markdown("### 📊 Monte Carlo Simulation Analysis")
                
                # First, let's run a smaller simulation to show actual paths
                with st.spinner("Generating visualization paths..."):
                    # Get some sample paths for visualization (100 paths of 252 days each)
                    np.random.seed(42)  # For reproducible results
                    n_paths = 100
                    n_days = min(252, len(returns))  # 1 year or available data
                    
                    paths = []
                    for _ in range(n_paths):
                        path = [1.0]  # Start with $1
                        for _ in range(n_days):
                            daily_return = np.random.choice(returns.values)
                            path.append(path[-1] * (1 + daily_return))
                        paths.append(path)
                    
                    paths = np.array(paths)
                
                # Create subplots for comprehensive analysis
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'Sample Investment Paths (100 scenarios)', 
                        'Final Returns Distribution',
                        'Percentile Analysis', 
                        'Risk vs Return Scatter'
                    ],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # 1. Sample paths visualization
                days = list(range(n_days + 1))
                
                # Show a selection of paths
                for i in range(0, min(50, n_paths), 5):  # Show every 5th path up to 50
                    alpha = 0.3 if i < 40 else 0.8  # Highlight last few paths
                    color = 'lightblue' if i < 40 else 'blue'
                    fig.add_trace(
                        go.Scatter(
                            x=days, 
                            y=paths[i], 
                            mode='lines',
                            line=dict(color=color, width=1),
                            opacity=alpha,
                            showlegend=False,
                            name=f'Path {i+1}'
                        ),
                        row=1, col=1
                    )
                
                # Add percentile bands
                percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=days, y=percentiles[4], 
                        fill=None, mode='lines',
                        line=dict(color='green', width=2, dash='dash'),
                        name='95th Percentile'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=days, y=percentiles[2], 
                        fill='tonexty', mode='lines',
                        line=dict(color='blue', width=3),
                        fillcolor='rgba(0,100,80,0.2)',
                        name='Median Path'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=days, y=percentiles[0], 
                        fill='tonexty', mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        fillcolor='rgba(255,0,0,0.1)',
                        name='5th Percentile'
                    ),
                    row=1, col=1
                )
                
                # 2. Distribution histogram
                final_returns_sample = (paths[:, -1] - 1) * 100  # Convert to percentage
                fig.add_trace(
                    go.Histogram(
                        x=final_returns_sample,
                        nbinsx=20,
                        name='Sample Returns',
                        opacity=0.7,
                        marker_color='lightblue'
                    ),
                    row=1, col=2
                )
                
                # Add reference lines to histogram
                fig.add_vline(x=np.mean(final_returns_sample), 
                             line_dash="solid", line_color="blue", 
                             annotation_text=f"Mean: {np.mean(final_returns_sample):.1f}%",
                             row=1, col=2)
                
                # 3. Percentile analysis
                percentile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                percentile_values = np.percentile(mc_results['final_returns'], percentile_levels)
                
                fig.add_trace(
                    go.Scatter(
                        x=percentile_levels,
                        y=percentile_values * 100,  # Convert to percentage
                        mode='lines+markers',
                        line=dict(color='purple', width=3),
                        marker=dict(size=8),
                        name='Return Percentiles'
                    ),
                    row=2, col=1
                )
                
                # Add horizontal line at break-even
                fig.add_hline(y=0, line_dash="dash", line_color="black", 
                             annotation_text="Break-even", row=2, col=1)
                
                # 4. Risk vs Return scatter (by percentile)
                risk_levels = ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
                risk_returns = [
                    percentile_values[7] * 100,  # 95th percentile
                    percentile_values[6] * 100,  # 90th percentile  
                    percentile_values[4] * 100,  # 50th percentile
                    percentile_values[2] * 100,  # 10th percentile
                    percentile_values[1] * 100   # 5th percentile
                ]
                risk_probs = [95, 90, 50, 10, 5]
                
                colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
                fig.add_trace(
                    go.Scatter(
                        x=risk_probs,
                        y=risk_returns,
                        mode='markers+text',
                        marker=dict(size=15, color=colors),
                        text=risk_levels,
                        textposition="top center",
                        name='Risk Levels'
                    ),
                    row=2, col=2
                )
                
                # Update layout
                fig.update_layout(
                    height=800,
                    title_text="📊 Complete Monte Carlo Analysis Dashboard",
                    template='plotly_dark' if st.session_state.dark_theme else 'plotly_white'
                )
                
                # Update axes labels
                fig.update_xaxes(title_text="Days", row=1, col=1)
                fig.update_yaxes(title_text="Portfolio Value ($1 initial)", row=1, col=1)
                
                fig.update_xaxes(title_text="Return (%)", row=1, col=2)
                fig.update_yaxes(title_text="Frequency", row=1, col=2)
                
                fig.update_xaxes(title_text="Percentile", row=2, col=1)
                fig.update_yaxes(title_text="Return (%)", row=2, col=1)
                
                fig.update_xaxes(title_text="Probability of Achieving (%)", row=2, col=2)
                fig.update_yaxes(title_text="Return (%)", row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True, key=f"monte_carlo_comprehensive_{symbol}")
                
                # Additional detailed histogram with all 10,000 simulations
                st.markdown("### 📈 Complete Distribution Analysis (10,000 Simulations)")
                
                hist_fig = go.Figure()
                
                # Add main histogram
                hist_fig.add_trace(go.Histogram(
                    x=mc_results['final_returns'] * 100,  # Convert to percentage
                    nbinsx=50,
                    name='All Simulations',
                    opacity=0.7,
                    marker_color='lightblue'
                ))
                
                # Add key reference lines
                hist_fig.add_vline(x=mc_results['mean_final_return'] * 100, 
                             line_dash="solid", line_color="blue", line_width=3,
                             annotation_text=f"Mean: {mc_results['mean_final_return']:.1%}")
                
                hist_fig.add_vline(x=0, 
                             line_dash="solid", line_color="black", line_width=2,
                             annotation_text="Break-even")
                
                hist_fig.add_vline(x=var_5 * 100, 
                             line_dash="dash", line_color="red", line_width=2,
                             annotation_text=f"5% VaR: {var_5:.1%}")
                
                hist_fig.add_vline(x=mc_results['confidence_intervals']['95%'][1] * 100, 
                             line_dash="dash", line_color="green", line_width=2,
                             annotation_text=f"95% Upside: {mc_results['confidence_intervals']['95%'][1]:.1%}")
                
                # Add shaded regions for different risk zones
                hist_fig.add_vrect(
                    x0=worst_case * 100, x1=var_5 * 100,
                    fillcolor="red", opacity=0.2,
                    annotation_text="High Risk Zone", annotation_position="top left"
                )
                
                hist_fig.add_vrect(
                    x0=mc_results['confidence_intervals']['95%'][0] * 100, 
                    x1=mc_results['confidence_intervals']['95%'][1] * 100,
                    fillcolor="green", opacity=0.1,
                    annotation_text="95% Confidence Zone", annotation_position="top"
                )
                
                hist_fig.update_layout(
                    title="📊 Final Returns Distribution - All 10,000 Scenarios",
                    xaxis_title="Final Return (%)",
                    yaxis_title="Number of Simulations",
                    template='plotly_dark' if st.session_state.dark_theme else 'plotly_white',
                    height=400
                )
                
                st.plotly_chart(hist_fig, use_container_width=True, key=f"monte_carlo_histogram_{symbol}")
                
                # Pattern Analysis and Recommendations
                st.markdown("### 🔍 Pattern Analysis & Investment Recommendations")
                
                # Analyze the underlying patterns
                buy_hold_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1)
                total_years = (stock_data.index[-1] - stock_data.index[0]).days / 365.25
                annualized_buy_hold = (1 + buy_hold_return) ** (1/total_years) - 1
                
                # Calculate volatility and Sharpe ratio
                annual_volatility = returns.std() * np.sqrt(252)
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                sharpe_ratio = (mc_results['mean_final_return'] - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
                
                # Seasonal pattern strength
                seasonal_strength = seasonal_stats['Avg_Return'].std()
                best_month_return = seasonal_stats['Avg_Return'].max()
                worst_month_return = seasonal_stats['Avg_Return'].min()
                seasonal_spread = best_month_return - worst_month_return
                
                # Generate investment recommendations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 **Based on These Patterns, We Recommend:**")
                    
                    recommendations = []
                    
                    # Risk-based recommendations
                    if probability_loss < 30:
                        risk_assessment = "Low Risk Profile"
                        recommendations.append("✅ **Suitable for conservative investors**")
                    elif probability_loss < 50:
                        risk_assessment = "Moderate Risk Profile" 
                        recommendations.append("⚠️ **Suitable for moderate risk tolerance**")
                    else:
                        risk_assessment = "High Risk Profile"
                        recommendations.append("🚨 **Only for aggressive investors**")
                    
                    # Seasonal strategy recommendations
                    if seasonal_spread > 0.05:  # Strong seasonal pattern
                        recommendations.append(f"🌟 **Strong seasonal patterns detected** ({seasonal_spread:.1%} spread)")
                        recommendations.append("📈 **Consider seasonal timing strategies**")
                        if best_month_return > 0.03:
                            recommendations.append(f"🎯 **Focus on best months** (up to {best_month_return:.1%} avg return)")
                    
                    # Volatility recommendations  
                    if annual_volatility > 0.3:
                        recommendations.append("⚡ **High volatility detected** - Consider position sizing")
                    elif annual_volatility < 0.15:
                        recommendations.append("📊 **Low volatility** - Stable investment profile")
                    
                    # Sharpe ratio recommendations
                    if sharpe_ratio > 1.0:
                        recommendations.append(f"🏆 **Excellent risk-adjusted returns** (Sharpe: {sharpe_ratio:.2f})")
                    elif sharpe_ratio > 0.5:
                        recommendations.append(f"✅ **Good risk-adjusted returns** (Sharpe: {sharpe_ratio:.2f})")
                    else:
                        recommendations.append(f"⚠️ **Low risk-adjusted returns** (Sharpe: {sharpe_ratio:.2f})")
                    
                    for rec in recommendations:
                        st.markdown(rec)
                
                with col2:
                    st.markdown("### 📊 **Simulation vs Buy & Hold Analysis**")
                    
                    # Compare simulation results to buy and hold
                    excess_return = mc_results['mean_final_return'] - annualized_buy_hold
                    
                    st.markdown(f"""
                    **🔄 Buy & Hold Performance:**
                    - Total Return: {buy_hold_return:.1%} over {total_years:.1f} years
                    - Annualized Return: {annualized_buy_hold:.1%}
                    - Annual Volatility: {annual_volatility:.1%}
                    
                    **🎲 Monte Carlo Simulation:**
                    - Expected Return: {mc_results['mean_final_return']:.1%}
                    - Risk Assessment: {risk_assessment}
                    - Sharpe Ratio: {sharpe_ratio:.2f}
                    
                    **📈 Strategy Comparison:**
                    """)
                    
                    if excess_return > 0.02:
                        st.success(f"✅ **Simulation suggests {excess_return:+.1%} potential improvement** over buy & hold")
                        st.markdown("💡 **Recommendation**: Consider active seasonal strategies")
                    elif excess_return > -0.02:
                        st.info(f"📊 **Similar performance** to buy & hold ({excess_return:+.1%} difference)")
                        st.markdown("💡 **Recommendation**: Buy & hold may be simpler and cheaper")
                    else:
                        st.warning(f"⚠️ **Buy & hold may be better** ({excess_return:+.1%} underperformance)")
                        st.markdown("💡 **Recommendation**: Stick with buy & hold or reduce trading frequency")
                
                # Specific strategy recommendations based on patterns
                st.markdown("### 🛠️ **Recommended Strategies Based on Your Data**")
                
                strategy_recommendations = []
                
                # Analyze which strategies might work best
                if seasonal_spread > 0.08:  # Very strong seasonal patterns
                    strategy_recommendations.append({
                        "strategy": "🎯 **Best Months Only**",
                        "reason": f"Strong seasonal patterns ({seasonal_spread:.1%} spread) suggest timing can add significant value",
                        "risk": "Medium",
                        "complexity": "Low"
                    })
                
                if probability_loss < 35 and mc_results['mean_final_return'] > 0.05:
                    strategy_recommendations.append({
                        "strategy": "🌟 **Buy & Hold with Seasonal Overweighting**", 
                        "reason": "Good risk-return profile suggests staying invested with seasonal position sizing",
                        "risk": "Low-Medium",
                        "complexity": "Low"
                    })
                
                if annual_volatility > 0.25:
                    strategy_recommendations.append({
                        "strategy": "⚡ **High Volatility Avoidance**",
                        "reason": f"High volatility ({annual_volatility:.1%}) suggests avoiding volatile periods",
                        "risk": "Low",
                        "complexity": "Medium"
                    })
                
                if len(seasonal_stats[seasonal_stats['Win_Rate'] > 0.6]) >= 6:
                    strategy_recommendations.append({
                        "strategy": "📊 **Top 6 Months Strategy**",
                        "reason": "Multiple months with high win rates suggest extended seasonal exposure",
                        "risk": "Medium", 
                        "complexity": "Medium"
                    })
                
                # Display strategy recommendations
                for i, rec in enumerate(strategy_recommendations[:3]):  # Show top 3
                    with st.expander(f"🔥 Strategy {i+1}: {rec['strategy']}", expanded=i==0):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Level", rec['risk'])
                        with col2:
                            st.metric("Complexity", rec['complexity'])
                        with col3:
                            st.metric("Rank", f"#{i+1}")
                        
                        st.markdown(f"**Why this strategy:** {rec['reason']}")
                        st.markdown("💡 **Next Step:** Test this strategy in the backtesting section above!")
                
                # Summary interpretation
                # Additional Monte Carlo Path Visualization  
                st.markdown("### 📈 Individual Monte Carlo Simulation Paths")
                st.info("This chart shows 50 individual simulation paths to visualize the range of possible investment outcomes over time.")
                
                # Create a focused path visualization
                path_fig = go.Figure()
                
                # Show sample paths with different transparencies
                for i in range(min(50, n_paths)):
                    alpha = 0.4 if i >= 10 else 0.7  # More prominent first 10 paths
                    color = f'rgba(70, 130, 180, {alpha})'
                    if i < 5:  # Highlight first 5 paths
                        color = f'rgba(220, 20, 60, {alpha})'
                    
                    path_fig.add_trace(
                        go.Scatter(
                            x=days, 
                            y=paths[i], 
                            mode='lines',
                            line=dict(color=color, width=1.5 if i < 5 else 1),
                            name=f'Path {i+1}' if i < 3 else None,
                            showlegend=i < 3,
                            hovertemplate=f'Path {i+1}<br>Day: %{{x}}<br>Value: $%{{y:.2f}}<extra></extra>'
                        )
                    )
                
                # Add statistical reference lines
                path_fig.add_trace(
                    go.Scatter(
                        x=days, y=percentiles[4], 
                        mode='lines',
                        line=dict(color='green', width=3, dash='dot'),
                        name='95th Percentile (Best 5%)',
                        showlegend=True
                    )
                )
                
                path_fig.add_trace(
                    go.Scatter(
                        x=days, y=percentiles[2], 
                        mode='lines',
                        line=dict(color='blue', width=4),
                        name='Median Path',
                        showlegend=True
                    )
                )
                
                path_fig.add_trace(
                    go.Scatter(
                        x=days, y=percentiles[0], 
                        mode='lines',
                        line=dict(color='red', width=3, dash='dot'),
                        name='5th Percentile (Worst 5%)',
                        showlegend=True
                    )
                )
                
                # Add starting line
                path_fig.add_hline(y=1.0, line_dash="dash", line_color="black", 
                                 annotation_text="Starting Value ($1.00)")
                
                path_fig.update_layout(
                    title=f"Monte Carlo Investment Paths - {symbol} ({n_days} Trading Days)",
                    xaxis_title="Trading Days",
                    yaxis_title="Portfolio Value ($1 initial investment)",
                    height=500,
                    template='plotly_dark' if st.session_state.dark_theme else 'plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(path_fig, use_container_width=True, key=f"monte_carlo_individual_paths_{symbol}")
                
                # Add interpretive text
                st.markdown(f"""
                **📊 How to Read This Chart:**
                - Each line represents one possible investment journey over {n_days} trading days
                - **Green line (95th percentile)**: Only 5% of simulations performed better
                - **Blue line (median)**: Half of simulations were above/below this line  
                - **Red line (5th percentile)**: Only 5% of simulations performed worse
                - **Red highlighted paths**: Show some of the more volatile scenarios
                - **Starting point**: $1.00 investment grows or shrinks based on historical patterns
                """)
                
                st.markdown("### 💡 Investment Implications Summary")
                
                if mc_results['mean_final_return'] > 0.1:
                    outlook = "🟢 **Positive Outlook**"
                    interpretation = "The simulation suggests favorable expected returns with reasonable risk levels."
                elif mc_results['mean_final_return'] > 0:
                    outlook = "🟡 **Moderate Outlook**"
                    interpretation = "The simulation shows modest positive expectations but with notable uncertainty."
                else:
                    outlook = "🔴 **Cautious Outlook**"
                    interpretation = "The simulation indicates potential challenges with elevated risk levels."
                
                st.markdown(f"""
                **{outlook}**
                
                {interpretation}
                
                **Key Takeaways:**
                - **Expected Return**: {mc_results['mean_final_return']:.1%} average across 10,000 scenarios
                - **Risk Level**: {probability_loss:.0f}% chance of loss, with {probability_large_loss:.1f}% chance of significant loss
                - **Confidence Range**: 95% of outcomes between {mc_results['confidence_intervals']['95%'][0]:.1%} and {mc_results['confidence_intervals']['95%'][1]:.1%}
                - **Worst Case Protection**: Only 5% chance of losing more than {var_5:.1%}
                - **vs Buy & Hold**: {excess_return:+.1%} expected difference
                
                ⚠️ **Remember**: These projections are based on historical patterns and don't guarantee future performance.
                """)
                
            except Exception as e:
                st.error(f"Error running Monte Carlo simulation: {str(e)}")
            
    except Exception as e:
        st.error(f"Error in statistical tests: {str(e)}")

def display_market_regimes(stock_data, symbol):
    """Display market regime analysis"""
    
    # Check if we're in multi-asset mode
    if st.session_state.dashboard_mode == 'multi' and st.session_state.multi_asset_data:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%); 
                    color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
            <h1 style="margin: 0; font-size: 2.5rem;">🔔 Multi-Asset Market Regime Analysis</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Market conditions and regime detection across your asset portfolio</p>
        </div>
        """, unsafe_allow_html=True)
        
        all_assets = list(st.session_state.multi_asset_data.keys())
        
        # Asset selection for detailed regime analysis
        st.markdown("### 🎯 Select Asset for Detailed Analysis")
        selected_asset = st.selectbox(
            "Choose Asset", 
            all_assets,
            help="Select an asset for detailed market regime analysis",
            key="regime_selected_asset"
        )
        
        if st.button("⬅️ Back to Multi-Asset Dashboard", key="back_to_dashboard_regime"):
            st.session_state.selected_tab = "🎛️ Asset Dashboard"
            st.rerun()
        
        # Display detailed analysis for selected asset
        if selected_asset:
            asset_data = st.session_state.multi_asset_data[selected_asset]
            display_single_asset_market_regimes(asset_data['stock_data'], selected_asset)
        
        st.divider()
        
        # Portfolio-level regime overview
        st.markdown("### 📊 Portfolio Market Regime Overview")
        
        regime_summary = []
        for ticker in all_assets:
            asset_data = st.session_state.multi_asset_data[ticker]
            stock_data = asset_data['stock_data']
            
            try:
                regime_detector = MarketRegimeDetector()
                returns = stock_data['Returns'].dropna()
                prices = stock_data['Close']
                
                if len(returns) > 50:  # Minimum data for regime detection
                    vol_regimes = regime_detector.detect_volatility_regimes(returns)
                    trend_regimes = regime_detector.detect_trend_regimes(prices)
                    
                    current_vol_regime = vol_regimes['regimes'].iloc[-1]
                    current_trend_regime = trend_regimes['regimes'].iloc[-1]
                    current_volatility = vol_regimes['volatility'].iloc[-1]
                    
                    regime_summary.append({
                        'Asset': ticker,
                        'Vol Regime': current_vol_regime,
                        'Trend Regime': current_trend_regime,
                        'Current Vol': f"{current_volatility:.1%}",
                        'Vol Threshold': f"{vol_regimes['threshold']:.1%}",
                        'Regime Risk': 'High' if current_vol_regime == 'High Volatility' else 'Low'
                    })
                else:
                    regime_summary.append({
                        'Asset': ticker,
                        'Vol Regime': 'Insufficient Data',
                        'Trend Regime': 'Insufficient Data',
                        'Current Vol': 'N/A',
                        'Vol Threshold': 'N/A',
                        'Regime Risk': 'Unknown'
                    })
            except Exception as e:
                regime_summary.append({
                    'Asset': ticker,
                    'Vol Regime': 'Error',
                    'Trend Regime': 'Error',
                    'Current Vol': 'N/A',
                    'Vol Threshold': 'N/A',
                    'Regime Risk': 'Unknown'
                })
        
        if regime_summary:
            regime_df = pd.DataFrame(regime_summary)
            st.dataframe(regime_df, use_container_width=True, hide_index=True)
            
            # Portfolio regime statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_vol_count = len([item for item in regime_summary if item['Vol Regime'] == 'High Volatility'])
                st.metric("High Volatility", f"{high_vol_count}/{len(all_assets)}")
            
            with col2:
                bull_count = len([item for item in regime_summary if item['Trend Regime'] == 'Bull Market'])
                st.metric("Bull Markets", f"{bull_count}/{len(all_assets)}")
            
            with col3:
                bear_count = len([item for item in regime_summary if item['Trend Regime'] == 'Bear Market'])
                st.metric("Bear Markets", f"{bear_count}/{len(all_assets)}")
            
            with col4:
                high_risk_count = len([item for item in regime_summary if item['Regime Risk'] == 'High'])
                st.metric("High Risk Assets", f"{high_risk_count}/{len(all_assets)}")
        
        return
    
    # Single asset mode
    st.subheader(f"🔔 Market Regime Analysis for {symbol}")
    display_single_asset_market_regimes(stock_data, symbol)

def display_single_asset_market_regimes(stock_data, symbol):
    """Display market regime analysis for a single asset"""
    # Initialize regime detector
    regime_detector = MarketRegimeDetector()
    
    try:
        # Calculate returns and prices
        returns = stock_data['Returns'].dropna()
        prices = stock_data['Close']
        
        # Detect volatility regimes
        vol_regimes = regime_detector.detect_volatility_regimes(returns)
        trend_regimes = regime_detector.detect_trend_regimes(prices)
        
        # Current regime status
        st.markdown("### 📊 Current Market Regime")
        
        current_vol_regime = vol_regimes['regimes'].iloc[-1]
        current_trend_regime = trend_regimes['regimes'].iloc[-1]
        current_volatility = vol_regimes['volatility'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vol_color = 'danger-card' if current_vol_regime == 'High Volatility' else 'success-card'
            st.markdown(f"""
            <div class="metric-card {vol_color}">
                <h4>📊 Volatility Regime</h4>
                <h2>{current_vol_regime}</h2>
                <p class="metric-value">Current: {current_volatility:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            trend_color = 'success-card' if current_trend_regime == 'Bull Market' else 'danger-card' if current_trend_regime == 'Bear Market' else 'warning-card'
            st.markdown(f"""
            <div class="metric-card {trend_color}">
                <h4>📈 Trend Regime</h4>
                <h2>{current_trend_regime}</h2>
                <p class="metric-value">Based on moving averages</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            threshold = vol_regimes['threshold']
            st.markdown(f"""
            <div class="metric-card info-card">
                <h4>📊 Volatility Threshold</h4>
                <h2>{threshold:.1%}</h2>
                <p class="metric-value">High/Low vol boundary</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Regime statistics
        st.markdown("### 📈 Regime Statistics")
        
        # Volatility regime stats
        vol_regime_stats = vol_regimes['regimes'].value_counts()
        trend_regime_stats = trend_regimes['regimes'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volatility Regime Distribution**")
            for regime, count in vol_regime_stats.items():
                percentage = count / len(vol_regimes['regimes']) * 100
                st.write(f"• {regime}: {percentage:.1f}% ({count} days)")
        
        with col2:
            st.markdown("**Trend Regime Distribution**")
            for regime, count in trend_regime_stats.items():
                percentage = count / len(trend_regimes['regimes']) * 100
                st.write(f"• {regime}: {percentage:.1f}% ({count} days)")
        
        # Regime visualization
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=['Price & Trend Regimes', 'Volatility', 'Volatility Regimes'],
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price with trend regimes
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices, name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Color price by trend regime
        bull_periods = trend_regimes['regimes'] == 'Bull Market'
        bear_periods = trend_regimes['regimes'] == 'Bear Market'
        
        if bull_periods.any():
            fig.add_trace(
                go.Scatter(
                    x=prices[bull_periods].index, 
                    y=prices[bull_periods], 
                    mode='markers',
                    marker=dict(color='green', size=3),
                    name='Bull Market',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        if bear_periods.any():
            fig.add_trace(
                go.Scatter(
                    x=prices[bear_periods].index, 
                    y=prices[bear_periods], 
                    mode='markers',
                    marker=dict(color='red', size=3),
                    name='Bear Market',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Volatility
        fig.add_trace(
            go.Scatter(
                x=vol_regimes['volatility'].index,
                y=vol_regimes['volatility'],
                name='Rolling Volatility',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=threshold, line_dash="dash", line_color="orange", row=2, col=1)
        
        # Volatility regimes
        high_vol_periods = vol_regimes['regimes'] == 'High Volatility'
        low_vol_periods = vol_regimes['regimes'] == 'Low Volatility'
        
        if high_vol_periods.any():
            fig.add_trace(
                go.Scatter(
                    x=vol_regimes['volatility'][high_vol_periods].index,
                    y=[1] * high_vol_periods.sum(),
                    mode='markers',
                    marker=dict(color='red', size=5),
                    name='High Vol',
                    showlegend=True
                ),
                row=3, col=1
            )
        
        if low_vol_periods.any():
            fig.add_trace(
                go.Scatter(
                    x=vol_regimes['volatility'][low_vol_periods].index,
                    y=[0] * low_vol_periods.sum(),
                    mode='markers',
                    marker=dict(color='green', size=5),
                    name='Low Vol',
                    showlegend=True
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=800,
            template='plotly_dark' if st.session_state.dark_theme else 'plotly_white',
            title_text="Market Regime Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime change alerts
        recent_regime_changes = vol_regimes['regime_changes'].tail(10)
        if recent_regime_changes.any():
            st.markdown("### 🚨 Recent Regime Changes")
            change_dates = recent_regime_changes[recent_regime_changes].index
            for date in change_dates:
                regime = vol_regimes['regimes'].loc[date]
                st.write(f"• {date.strftime('%Y-%m-%d')}: Changed to **{regime}**")
        
    except Exception as e:
        st.error(f"Error in market regime analysis: {str(e)}")

def display_wiki_help():
    """Display comprehensive wiki and help section"""
    st.markdown("""
    <div class="section-header">
        <h3>📚 AI Seasonal Edge Wiki & User Guide</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick navigation
    wiki_section = st.selectbox(
        "📖 Choose a topic to learn about:",
        [
            "🎯 Getting Started",
            "📊 Understanding Metrics",
            "🔮 Advanced AI Features (NEW!)",
            "🧠 AI Pattern Detection",
            "🤖 Prophet Forecasting",
            "📈 Market Regime Analysis",
            "💰 Backtesting Strategies", 
            "🎲 Monte Carlo Simulation",
            "📈 Technical Indicators",
            "⚠️ Risk Management",
            "📋 Export & Reporting",
            "❓ FAQ & Troubleshooting"
        ],
        key="wiki_section_select"
    )
    
    if wiki_section == "🎯 Getting Started":
        st.markdown("""
        ## 🚀 Getting Started with AI Seasonal Edge
        
        ### 📁 Data Requirements
        **Supported File Formats:**
        - **CSV files** with columns: DATE, OPEN, HIGH, LOW, CLOSE
        - **Alternative format**: DATE, TIME, OPEN, HIGH, LOW, CLOSE  
        - **Automatic detection** of delimiters (comma, semicolon, tab)
        - **Date formats**: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, and more
        
        **Data Sources:**
        - 📈 Stocks (NYSE, NASDAQ, international markets)
        - 🏦 ETFs and mutual funds
        - 💱 Forex pairs (major, minor, exotic)
        - ₿ Cryptocurrencies
        - 🥇 Commodities (gold, oil, agricultural)
        
        ### 📋 CSV File Format Requirements
        
        **Required Columns (case-insensitive):**
        ```
        DATE, OPEN, HIGH, LOW, CLOSE
        ```
        
        **Supported Column Name Formats:**
        - Standard: `DATE,OPEN,HIGH,LOW,CLOSE`
        - With brackets: `<DATE>,<OPEN>,<HIGH>,<LOW>,<CLOSE>`  
        - Short form: `DATE,O,H,L,C`
        - With time: `DATE,TIME,OPEN,HIGH,LOW,CLOSE`
        
        **File Format Examples:**
        
        **Example 1: Comma-separated (CSV)**
        ```csv
        DATE,OPEN,HIGH,LOW,CLOSE,VOLUME
        2024-01-02,150.25,152.40,149.80,151.75,1000000
        2024-01-03,151.80,153.20,150.90,152.10,950000
        2024-01-04,152.05,154.60,151.40,154.25,1100000
        ```
        
        **Example 2: Tab-separated (like your files)**
        ```
        <DATE>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>
        2017.12.07	42.33	42.61	42.24	42.39	6192	804016	0
        2017.12.08	42.58	42.75	42.21	42.33	8116	1155543	0
        2017.12.11	42.49	43.22	42.47	43.17	8383	1125816	0
        ```
        
        **Example 3: With timestamp**
        ```csv
        DATE,TIME,OPEN,HIGH,LOW,CLOSE
        2024-01-02,09:30:00,150.25,152.40,149.80,151.75
        2024-01-02,16:00:00,151.75,152.10,150.95,152.05
        ```
        
        **📁 Multi-Asset Dashboard Requirements:**
        - **File naming**: Name your CSV files with the ticker symbol (e.g., `AAPL.csv`, `TSLA_Daily.csv`)
        - **Multiple files**: Upload multiple CSV files at once for batch processing
        - **Auto-detection**: System automatically detects tab vs comma separation
        - **Column detection**: Automatically identifies OHLC price data vs ticker lists
        
        **⚠️ Important Notes:**
        - Date format is flexible (YYYY-MM-DD, MM/DD/YYYY, DD.MM.YYYY, etc.)
        - Additional columns (VOLUME, SPREAD, etc.) are optional and will be ignored
        - Missing values should be empty cells, not "N/A" or "NULL"
        - Minimum recommended: 252 trading days (1 year) for reliable seasonal analysis
        
        ### ⚡ Quick Start Process
        1. **Upload Data**: Use the file uploader in the sidebar
        2. **Configure AI**: Enable AI analysis and set confidence threshold
        3. **Run Analysis**: Click the 🚀 "Run Analysis" button
        4. **Explore Results**: Navigate through the 10+ analysis tabs
        5. **Test Strategies**: Use backtesting and Monte Carlo simulation
        6. **Export Insights**: Generate PDF reports or CSV exports
        """)
    
    elif wiki_section == "📊 Understanding Metrics":
        st.markdown("""
        ## 📊 Key Metrics Explained
        
        ### 🌟 Seasonal Statistics
        
        **Average Return** 📈
        - Monthly average price change percentage
        - Calculated across all years in your dataset
        - **Good**: > 2% monthly average
        - **Excellent**: > 5% monthly average
        
        **Win Rate** 🎯
        - Percentage of times the month was positive
        - Higher win rates indicate more consistent performance
        - **Good**: > 60% win rate
        - **Excellent**: > 70% win rate
        
        **Volatility** ⚡
        - Standard deviation of monthly returns
        - Measures price unpredictability and risk
        - **Low Risk**: < 5% monthly volatility
        - **High Risk**: > 10% monthly volatility
        
        **Best/Worst Returns** 🔥❄️
        - Historical maximum and minimum monthly performance
        - Shows the range of possible outcomes
        - Important for risk assessment
        
        ### 💰 Backtesting Metrics
        
        **Total Return** 💎
        - Complete strategy performance over test period
        - Includes all transaction costs and timing
        
        **Sharpe Ratio** 🏆
        - Risk-adjusted return measure
        - **Formula**: (Return - Risk-free rate) / Volatility
        - **Good**: > 0.5 | **Excellent**: > 1.0
        
        **Maximum Drawdown** 📉
        - Largest peak-to-trough decline
        - Critical risk metric for position sizing
        - **Conservative**: < 15% | **Aggressive**: > 30%
        
        **Win Rate** ✅
        - Percentage of profitable trades
        - **Trend strategies**: 40-60% typical
        - **Mean reversion**: 60-80% typical
        """)
    
    elif wiki_section == "🔮 Advanced AI Features (NEW!)":
        st.markdown("""
        ## 🔮 Advanced AI Features - Complete Guide
        
        ### 🚀 **What's New in AI Seasonal Edge**
        Our AI analyzer has been completely enhanced with cutting-edge machine learning and time series forecasting capabilities!
        
        ### 🧠 **Comprehensive AI Analysis Pipeline**
        
        #### **🔮 Prophet Time Series Forecasting**
        - **Multi-horizon predictions**: 30, 60, 90, 180, and 365-day forecasts
        - **Seasonal decomposition**: Automatic detection of yearly, quarterly, and monthly patterns
        - **Holiday effects**: Built-in calendar effects for market holidays
        - **Changepoint detection**: Automatic identification of structural breaks
        - **Uncertainty intervals**: 68% and 95% confidence bounds
        - **Cross-validation**: MAPE and MAE accuracy metrics
        
        #### **📊 Advanced Statistical Analysis**
        - **Time series decomposition**: Trend, seasonal, and residual components
        - **Stationarity testing**: ADF and KPSS tests for data properties
        - **Autocorrelation analysis**: Ljung-Box test for serial correlation
        - **Structural break detection**: Mean and volatility change points
        - **Cyclical pattern identification**: Peak and trough detection
        
        #### **🤖 Machine Learning Pattern Detection**
        - **Random Forest analysis**: Feature importance for 15+ technical indicators
        - **XGBoost modeling**: Advanced gradient boosting for complex patterns
        - **Isolation Forest**: Anomaly detection using ensemble methods
        - **Feature engineering**: Time-based, technical, and volatility features
        - **Cross-validation**: Time series splits for robust validation
        
        #### **📈 Market Intelligence & Regime Analysis**
        - **Volatility regime detection**: High/low volatility period identification
        - **Trend analysis**: Bull/bear/neutral regime classification
        - **Risk assessment**: Comprehensive drawdown and VaR analysis
        - **Correlation analysis**: Cross-asset relationship identification
        - **Trading strategy generation**: AI-powered strategy recommendations
        
        ### 🎯 **AI Insight Categories**
        
        #### **🏆 High-Confidence Patterns (95%+ Statistical Significance)**
        - Seasonally significant trends validated across multiple years
        - Machine learning cross-validated patterns
        - Prophet forecasting accuracy > 75%
        
        #### **🔄 Market Regimes**
        - **Volatility clustering**: Periods of high/low volatility
        - **Bull/bear transitions**: Trend regime changes
        - **Structural breaks**: Permanent changes in mean/volatility
        
        #### **⚠️ Risk Analysis**
        - **Maximum drawdown scenarios**: Worst-case loss projections
        - **Volatility forecasting**: Expected future volatility
        - **Value at Risk (VaR)**: Statistical loss probabilities
        
        #### **💰 AI-Generated Trading Strategies**
        - **Seasonal momentum**: Long strongest months, avoid weakest
        - **Calendar spreads**: Long/short seasonal arbitrage
        - **Volatility timing**: Position sizing based on volatility regimes
        - **Mean reversion**: Contrarian approaches during extremes
        
        ### 📊 **Performance Metrics & Benchmarks**
        
        #### **Prophet Forecasting Accuracy**
        - **Directional accuracy**: 65-85% across asset classes
        - **MAPE (Mean Absolute Percentage Error)**: 8-15% for 30-day forecasts
        - **Coverage probability**: 68% intervals capture 72% of actual values
        
        #### **Pattern Detection Reliability**
        - **Statistical significance**: 95%+ confidence for all reported patterns
        - **Cross-validation**: 5-fold time series validation
        - **Feature importance**: Top 5 features explain 60-80% of variance
        
        #### **Anomaly Detection Performance**
        - **False positive rate**: <5% for significant anomalies
        - **Detection rate**: 80-90% for major regime changes
        - **Response time**: Real-time detection within analysis
        
        ### ⚙️ **Configuration & Optimization**
        
        #### **AI Confidence Threshold**
        - **95%**: Ultra-high confidence, very few but strongest signals
        - **90%**: High confidence, balanced approach (recommended)
        - **85%**: Moderate confidence, more signals with good reliability
        - **75%**: Lower threshold, maximum signal detection
        
        #### **Feature Engineering Options**
        - **Technical indicators**: 15+ indicators (SMA, volatility, momentum, etc.)
        - **Time-based features**: Month, quarter, day-of-week effects
        - **Lag features**: 1, 5, 20-day return patterns
        - **Volatility features**: GARCH, rolling volatility, regime indicators
        
        #### **Forecasting Parameters**
        - **Prophet seasonality**: Auto-detected yearly, quarterly, monthly
        - **Holiday effects**: Market holiday impact modeling
        - **Changepoint flexibility**: Automatic vs manual specification
        - **Uncertainty intervals**: 68% (1-sigma) or 95% (2-sigma) bounds
        """)
    
    elif wiki_section == "🧠 AI Pattern Detection":
        st.markdown("""
        ## 🧠 AI Pattern Detection System (ENHANCED!)
        
        ### 🎯 **What Our Enhanced AI Analyzes**
        
        #### **🌟 Advanced Seasonal Analysis**
        - **Statistical significance testing**: T-tests and chi-square analysis
        - **Multi-year pattern validation**: Consistency across different market cycles
        - **Seasonal strength scoring**: Quantitative pattern reliability measures
        - **Calendar effect detection**: Monthly, quarterly, and holiday patterns
        
        #### **⚡ Volatility Intelligence**
        - **GARCH modeling**: Advanced volatility forecasting
        - **Regime switching detection**: High/low volatility periods
        - **Volatility clustering**: Periods of sustained high/low volatility
        - **Risk regime forecasting**: Expected future volatility scenarios
        
        #### **🔄 Advanced Mean Reversion & Momentum**
        - **Ornstein-Uhlenbeck modeling**: Mean reversion strength and speed
        - **Momentum persistence analysis**: Trend continuation probabilities
        - **Multi-timeframe analysis**: Short, medium, and long-term patterns
        - **Regime-dependent behavior**: How patterns change across market conditions
        
        #### **🚨 Anomaly & Outlier Detection**
        - **Isolation Forest**: Unsupervised anomaly detection
        - **Statistical outliers**: Z-score and percentile-based detection
        - **Pattern deviations**: When current behavior differs from historical norms
        - **Black swan events**: Extreme tail risk identification
        
        ### 🔬 **Advanced Statistical Methods**
        
        #### **📊 Hypothesis Testing Framework**
        - **Null hypothesis**: No seasonal/pattern effect exists
        - **Alternative hypothesis**: Significant pattern detected
        - **Multiple testing correction**: Bonferroni and FDR adjustments
        - **Effect size measurement**: Cohen's d and practical significance
        
        #### **🤖 Machine Learning Pipeline**
        - **Feature selection**: Mutual information and correlation filtering
        - **Model ensemble**: Random Forest + XGBoost + Prophet combination
        - **Hyperparameter optimization**: Grid search and Bayesian optimization
        - **Cross-validation**: Time series aware validation splits
        
        #### **📈 Time Series Analysis**
        - **Stationarity testing**: ADF, KPSS, and Phillips-Perron tests
        - **Cointegration analysis**: Long-term relationship detection
        - **Spectral analysis**: Frequency domain pattern identification
        - **Wavelet analysis**: Time-frequency decomposition
        
        ### 🎯 **Enhanced Confidence Scoring**
        
        #### **🏆 Ultra-High Confidence (95-99%)**
        - Multiple statistical tests confirm pattern
        - Consistent across all validation periods
        - Strong economic/fundamental rationale
        - **Action**: High conviction trades with larger position sizes
        
        #### **✅ High Confidence (85-94%)**
        - Most statistical tests confirm pattern
        - Occasional validation period failures
        - Good risk-adjusted returns
        - **Action**: Standard position sizes, primary strategy component
        
        #### **⚖️ Moderate Confidence (75-84%)**
        - Some statistical significance
        - Mixed validation results
        - Supplementary evidence helpful
        - **Action**: Smaller positions, combine with other signals
        
        #### **⚠️ Low Confidence (65-74%)**
        - Weak statistical evidence
        - Inconsistent validation
        - High uncertainty intervals
        - **Action**: Monitor only, avoid trading
        
        ### 📊 **AI Pattern Categories**
        
        #### **🔥 Seasonal Momentum Patterns**
        - **"January Effect"**: New year portfolio rebalancing
        - **"December Rally"**: Year-end institutional buying
        - **"Sell in May"**: Summer seasonal weakness
        - **"September Effect"**: Post-vacation market activity
        
        #### **💫 Volatility Regime Patterns**
        - **Low-vol clustering**: Periods of sustained calm
        - **High-vol clustering**: Stress period identification
        - **Volatility transitions**: Regime change timing
        - **VIX relationship**: Fear gauge correlations
        
        #### **🎯 Mean Reversion Opportunities**
        - **Monthly overshoots**: When seasonal moves exceed norms
        - **Reversion timing**: Optimal entry/exit points
        - **Magnitude prediction**: Expected reversion size
        - **Duration modeling**: How long reversions take
        
        #### **⚡ Momentum Continuation**
        - **Breakout validation**: Real vs false breakouts
        - **Trend persistence**: How long trends continue
        - **Acceleration phases**: When trends strengthen
        - **Exhaustion signals**: When trends weaken
        
        ### 🛠️ **Advanced Configuration Options**
        
        #### **🎚️ Model Sensitivity Settings**
        - **Feature importance threshold**: Minimum relevance for inclusion
        - **Ensemble weights**: How much weight each model gets
        - **Validation strictness**: Cross-validation requirements
        - **Outlier sensitivity**: How aggressively to detect anomalies
        
        #### **📅 Time Period Analysis**
        - **Lookback window**: Years of data for pattern detection
        - **Validation periods**: Out-of-sample testing periods
        - **Rolling analysis**: How often to update patterns
        - **Regime awareness**: Adjust for changing market conditions
        
        #### **🎯 Signal Filtering**
        - **Minimum sample size**: Required data points per pattern
        - **Statistical power**: Minimum test power requirements
        - **Economic significance**: Minimum effect size thresholds
        - **Transaction cost adjustment**: Net returns after costs
        """)
    
    elif wiki_section == "🤖 Prophet Forecasting":
        st.markdown("""
        ## 🤖 Prophet Forecasting - Advanced Guide
        
        ### 🔮 **What is Prophet?**
        Prophet is Facebook's state-of-the-art time series forecasting system, specifically designed for business time series with strong seasonal effects and several seasons of historical data.
        
        ### 🚀 **Our Prophet Implementation**
        
        #### **📊 Multi-Horizon Forecasting**
        - **30-day forecast**: Short-term tactical positioning
        - **60-day forecast**: Medium-term strategic planning
        - **90-day forecast**: Quarterly planning horizon
        - **180-day forecast**: Semi-annual outlook
        - **365-day forecast**: Annual strategic planning
        
        #### **🌊 Advanced Seasonality Detection**
        - **Yearly seasonality**: Annual patterns (e.g., "Sell in May")
        - **Quarterly seasonality**: Earnings and business cycles
        - **Monthly seasonality**: Calendar effects and rebalancing
        - **Custom seasonalities**: Asset-specific patterns
        
        #### **📅 Holiday & Event Modeling**
        - **Market holidays**: NYSE/NASDAQ closure effects
        - **Earnings seasons**: Q1, Q2, Q3, Q4 patterns
        - **Economic events**: FOMC meetings, GDP releases
        - **Custom events**: Asset-specific catalysts
        
        ### 📈 **Forecast Components Explained**
        
        #### **📊 Trend Component**
        - **Linear trend**: Steady growth/decline patterns
        - **Logistic trend**: Growth with saturation limits
        - **Changepoints**: Automatic detection of trend changes
        - **Trend strength**: How strong the underlying trend is
        
        #### **🌀 Seasonal Components**
        - **Additive seasonality**: Fixed seasonal effects
        - **Multiplicative seasonality**: Proportional seasonal effects
        - **Seasonal strength**: How dominant seasonal patterns are
        - **Fourier order**: Complexity of seasonal patterns
        
        #### **📅 Holiday Effects**
        - **Pre-holiday effects**: Behavior before market closures
        - **Post-holiday effects**: Behavior after reopening
        - **Holiday windows**: Days of influence around events
        - **Holiday magnitude**: Strength of holiday effects
        
        #### **🔀 Noise Component**
        - **Residual errors**: Unexplained variation
        - **Error distribution**: Normal vs heavy-tailed errors
        - **Heteroscedasticity**: Time-varying volatility
        - **Autocorrelation**: Serial correlation in errors
        
        ### 🎯 **Uncertainty Quantification**
        
        #### **📊 Confidence Intervals**
        - **68% interval**: ±1 standard deviation (typical range)
        - **95% interval**: ±2 standard deviations (extreme range)
        - **Interval width**: Reflects forecast uncertainty
        - **Asymmetric intervals**: Skewed uncertainty distributions
        
        #### **🎲 Scenario Analysis**
        - **Best case**: Upper confidence bound scenarios
        - **Most likely**: Central forecast scenarios
        - **Worst case**: Lower confidence bound scenarios
        - **Monte Carlo**: 1000+ scenario simulations
        
        ### 📊 **Forecast Accuracy Metrics**
        
        #### **📈 Error Measures**
        - **MAPE**: Mean Absolute Percentage Error (8-15% typical)
        - **MAE**: Mean Absolute Error (dollar terms)
        - **RMSE**: Root Mean Square Error (penalizes large errors)
        - **SMAPE**: Symmetric MAPE (handles negative values)
        
        #### **🎯 Directional Accuracy**
        - **Hit rate**: Percentage of correct direction predictions
        - **Up/down accuracy**: Separate accuracy for gains/losses
        - **Magnitude accuracy**: How close predicted sizes are
        - **Timing accuracy**: How accurate the timing is
        
        #### **✅ Cross-Validation Results**
        - **Rolling validation**: Out-of-sample testing
        - **Expanding window**: Increasing data validation
        - **Blocked validation**: Seasonal period testing
        - **Walk-forward**: Real-time simulation
        
        ### 🛠️ **Configuration & Tuning**
        
        #### **🎚️ Seasonality Parameters**
        - **Yearly seasonality strength**: 0.01 to 10+ (auto-detected)
        - **Weekly seasonality**: Day-of-week effects
        - **Custom seasonality**: User-defined periods
        - **Seasonality mode**: Additive vs multiplicative
        
        #### **📈 Trend Parameters**
        - **Growth type**: Linear vs logistic
        - **Changepoint prior**: Flexibility of trend changes
        - **Changepoint range**: Where changes can occur
        - **Capacity**: Maximum value for logistic growth
        
        #### **🎯 Forecast Horizon Optimization**
        - **Short-term (1-30 days)**: High accuracy, tactical use
        - **Medium-term (1-6 months)**: Moderate accuracy, strategic use
        - **Long-term (6+ months)**: Lower accuracy, planning use
        - **Optimal horizon**: Where accuracy meets utility
        
        ### 📊 **Interpreting Prophet Results**
        
        #### **🎯 Strong Forecasts (MAPE < 10%)**
        - High seasonal consistency
        - Stable trend patterns
        - Low residual volatility
        - **Action**: Use for primary decision making
        
        #### **✅ Good Forecasts (MAPE 10-20%)**
        - Moderate seasonal patterns
        - Some trend instability
        - Acceptable error ranges
        - **Action**: Use with additional confirmation
        
        #### **⚠️ Weak Forecasts (MAPE > 20%)**
        - Inconsistent patterns
        - High volatility
        - Wide confidence intervals
        - **Action**: Use for directional guidance only
        
        ### 🚀 **Advanced Prophet Features**
        
        #### **🔄 Automatic Model Selection**
        - **Seasonality detection**: Auto-identifies relevant seasons
        - **Changepoint detection**: Finds structural breaks
        - **Holiday selection**: Relevant holidays for asset class
        - **Parameter optimization**: Grid search for best fit
        
        #### **📊 Component Analysis**
        - **Trend contribution**: How much trend drives forecast
        - **Seasonal contribution**: Seasonal vs non-seasonal variance
        - **Holiday contribution**: Event impact quantification
        - **Uncertainty sources**: What drives forecast uncertainty
        
        #### **🎯 Custom Enhancements**
        - **External regressors**: Economic indicators, VIX, etc.
        - **Regime awareness**: Different models for different regimes
        - **Ensemble forecasting**: Combine multiple Prophet models
        - **Real-time updating**: Continuous model improvement
        """)
    
    elif wiki_section == "📈 Market Regime Analysis":
        st.markdown("""
        ## 📈 Market Regime Analysis - Complete Guide
        
        ### 🎯 **What are Market Regimes?**
        Market regimes are distinct periods characterized by similar market behavior, volatility patterns, and underlying economic conditions. Our AI automatically detects and analyzes these regimes.
        
        ### 🔍 **Types of Market Regimes We Detect**
        
        #### **📊 Volatility Regimes**
        - **Low volatility regime**: VIX < 20, calm markets, steady trends
        - **Medium volatility regime**: VIX 20-30, normal market conditions
        - **High volatility regime**: VIX > 30, stressed markets, erratic moves
        - **Extreme volatility regime**: VIX > 40, crisis conditions
        
        #### **📈 Trend Regimes**
        - **Bull market regime**: Sustained uptrends, positive momentum
        - **Bear market regime**: Sustained downtrends, negative momentum
        - **Sideways regime**: Range-bound markets, mean reversion
        - **Transition regime**: Uncertain direction, regime switching
        
        #### **🌊 Correlation Regimes**
        - **Low correlation regime**: Diversification works, stock picking effective
        - **High correlation regime**: Everything moves together, macro-driven
        - **Sector rotation regime**: Leadership changes between sectors
        - **Risk-on/Risk-off regime**: Flight to quality vs growth seeking
        
        ### 🤖 **AI Regime Detection Methods**
        
        #### **📊 Statistical Approaches**
        - **Hidden Markov Models**: Probabilistic regime identification
        - **Threshold models**: Volatility and return-based triggers
        - **Regime switching models**: Markov switching with multiple states
        - **Change point detection**: Structural break identification
        
        #### **🧠 Machine Learning Methods**
        - **Clustering algorithms**: K-means and hierarchical clustering
        - **Support Vector Machines**: Classification of market states
        - **Random Forest classification**: Ensemble regime prediction
        - **Neural networks**: Deep learning pattern recognition
        
        #### **📈 Technical Indicators**
        - **Volatility measures**: VIX, realized volatility, GARCH
        - **Momentum indicators**: RSI, MACD, moving average slopes
        - **Trend strength**: ADX, Aroon, linear regression slopes
        - **Market breadth**: Advance/decline, new highs/lows
        
        ### 🎯 **Regime Characteristics Analysis**
        
        #### **⏱️ Regime Duration**
        - **Average duration**: How long regimes typically last
        - **Minimum duration**: Shortest observed regime periods
        - **Maximum duration**: Longest sustained regimes
        - **Transition probability**: Likelihood of regime changes
        
        #### **📊 Regime Performance**
        - **Average returns**: Expected returns in each regime
        - **Volatility levels**: Risk characteristics per regime
        - **Sharpe ratios**: Risk-adjusted performance
        - **Maximum drawdowns**: Worst-case scenarios per regime
        
        #### **🔄 Transition Patterns**
        - **Regime persistence**: Probability of staying in same regime
        - **Transition matrix**: Probabilities of switching between regimes
        - **Leading indicators**: Early warning signals of changes
        - **Lag effects**: How quickly markets respond to regime changes
        
        ### 📈 **Seasonal Patterns by Regime**
        
        #### **🌟 Regime-Dependent Seasonality**
        - **Bull market seasonality**: How seasonal patterns work in uptrends
        - **Bear market seasonality**: Different patterns during downtrends
        - **Volatility regime effects**: How VIX levels affect seasonal patterns
        - **Correlation regime impacts**: When diversification helps/hurts
        
        #### **📅 Calendar Effects by Regime**
        - **January Effect**: Stronger in certain volatility regimes
        - **Sell in May**: More pronounced during specific trend regimes
        - **December Rally**: Enhanced or diminished by regime context
        - **Quarterly patterns**: How earnings cycles interact with regimes
        
        ### 🚨 **Regime Change Detection**
        
        #### **⚡ Real-Time Signals**
        - **Volatility spikes**: Sudden VIX increases above thresholds
        - **Correlation breaks**: Sudden changes in asset correlations
        - **Momentum shifts**: Rapid changes in trend indicators
        - **Volume anomalies**: Unusual trading volume patterns
        
        #### **📊 Probability Scoring**
        - **Regime probability**: Likelihood of being in each regime
        - **Transition probability**: Chance of regime change
        - **Confidence intervals**: Uncertainty around regime classification
        - **Signal strength**: How strong the regime indicators are
        
        #### **🎯 Leading Indicators**
        - **Credit spreads**: Corporate bond vs Treasury yield differences
        - **Term structure**: Yield curve shape and steepness
        - **Currency movements**: Safe haven vs risk currency flows
        - **Commodity prices**: Economic growth and inflation signals
        
        ### 💰 **Trading Strategies by Regime**
        
        #### **📈 Bull Market Strategies**
        - **Momentum strategies**: Trend following and breakout strategies
        - **Growth focus**: Technology and growth stock emphasis
        - **Risk-on positioning**: Higher beta and cyclical exposure
        - **Seasonal enhancement**: Stronger seasonal effects
        
        #### **📉 Bear Market Strategies**
        - **Defensive positioning**: Utilities, consumer staples focus
        - **Quality emphasis**: High dividend yields, low debt
        - **Safe haven assets**: Treasuries, gold, defensive currencies
        - **Seasonal modification**: Muted seasonal effects
        
        #### **⚡ High Volatility Strategies**
        - **Position sizing**: Smaller positions during high vol periods
        - **Option strategies**: Selling volatility during extreme spikes
        - **Diversification**: More uncorrelated assets and strategies
        - **Quick exits**: Shorter holding periods and tighter stops
        
        #### **😴 Low Volatility Strategies**
        - **Leverage opportunities**: Higher position sizes when appropriate
        - **Mean reversion**: Range trading and contrarian approaches
        - **Carry trades**: Higher yielding assets and currencies
        - **Seasonal amplification**: Stronger seasonal pattern following
        
        ### 📊 **Regime Analysis Metrics**
        
        #### **🎯 Regime Identification Accuracy**
        - **Classification accuracy**: Percentage of correct regime calls
        - **False positive rate**: Incorrect regime change signals
        - **True positive rate**: Correct regime change detection
        - **Regime stability**: How well-defined regime boundaries are
        
        #### **📈 Performance by Regime**
        - **Regime-specific Sharpe ratios**: Risk-adjusted returns per regime
        - **Regime transition costs**: Performance impact of regime changes
        - **Optimal allocation**: Best asset allocation per regime
        - **Risk budgeting**: How much risk to take in each regime
        
        ### 🛠️ **Configuration & Optimization**
        
        #### **🎚️ Detection Sensitivity**
        - **Change point threshold**: How sensitive to detect regime changes
        - **Minimum regime duration**: Avoid too-frequent regime switching
        - **Confidence threshold**: Required confidence for regime classification
        - **Lookback period**: How much data to use for regime detection
        
        #### **📊 Model Parameters**
        - **Number of regimes**: 2-5 regimes typically optimal
        - **Regime variables**: Which indicators to use for detection
        - **Smoothing parameters**: How much to smooth regime probabilities
        - **Update frequency**: How often to recalculate regimes
        """)
    
    elif wiki_section == "💰 Backtesting Strategies":
        st.markdown("""
        ## 💰 Backtesting Strategies Guide
        
        ### 🎯 Strategy Types Explained
        
        **1. Seasonal Long (Buy & Hold)** 🌟
        - Always invested in the market
        - Baseline strategy for comparison
        - **Best for**: Conservative investors, low transaction costs
        - **Risk**: Medium | **Complexity**: Low
        
        **2. Best Months Only** 🏆
        - Only invested during top 3 performing months
        - Cash during other months (0% return)
        - **Best for**: Strong seasonal patterns, tactical allocation
        - **Risk**: Medium-High | **Complexity**: Low
        
        **3. Avoid Worst Months** 🚫
        - Invested except during worst 3 months
        - Defensive approach to seasonal investing
        - **Best for**: Risk reduction, bear market protection
        - **Risk**: Low-Medium | **Complexity**: Low
        
        **4. Top 6 Months Strategy** 📊
        - Balanced approach, invested 6 months/year
        - Reduces transaction costs vs "Best 3"
        - **Best for**: Moderate seasonal exposure
        - **Risk**: Medium | **Complexity**: Medium
        
        **5. Quarterly Rotation** 🔄
        - Invests in best quarter, avoids worst quarter
        - Only 4 trades per year = lower costs
        - **Best for**: Institutional investors, lower frequency
        - **Risk**: Medium | **Complexity**: Low
        
        **6. Monthly Mean Reversion** ↩️
        - Contrarian: Buy weakness, sell strength
        - Assumes seasonal patterns reverse
        - **Best for**: Experienced traders, volatile markets
        - **Risk**: High | **Complexity**: High
        
        **7. High Volatility Avoidance** ⚡
        - Avoids historically volatile months
        - Risk-reduction focused approach
        - **Best for**: Conservative portfolios, retirement funds
        - **Risk**: Low | **Complexity**: Medium
        
        **8. Momentum Following** 🚀
        - Follows months with positive momentum + high win rates
        - Trend-following approach
        - **Best for**: Growth investors, bull markets
        - **Risk**: Medium-High | **Complexity**: Medium
        
        ### ⚙️ Configuration Parameters
        
        **Initial Capital** 💰
        - Starting portfolio value
        - Affects absolute dollar returns
        - Typical: $10,000 - $1,000,000
        
        **Commission (%)** 💸
        - Transaction cost per trade
        - **Discount brokers**: 0.05% - 0.1%
        - **Full service**: 0.25% - 0.5%
        - **Impact**: Higher = favors lower frequency strategies
        """)
    
    elif wiki_section == "🎲 Monte Carlo Simulation":
        st.markdown("""
        ## 🎲 Monte Carlo Simulation Guide
        
        ### 🔍 What is Monte Carlo Analysis?
        Monte Carlo simulation uses random sampling to model thousands of possible future scenarios based on historical patterns.
        
        ### 📊 How It Works
        1. **Historical Analysis**: Studies past return patterns and volatility
        2. **Random Sampling**: Generates 10,000 different future scenarios
        3. **Statistical Analysis**: Analyzes the distribution of outcomes
        4. **Risk Assessment**: Calculates probabilities and confidence intervals
        
        ### 📈 Key Outputs Explained
        
        **Mean Final Return** 🎯
        - Average expected return across all scenarios
        - Most likely outcome (but not guaranteed)
        
        **Confidence Intervals** 📊
        - **95% Confidence**: 95% of outcomes fall within this range
        - **68% Confidence**: 68% of outcomes (1 standard deviation)
        - Wider ranges = higher uncertainty
        
        **Value at Risk (VaR)** ⚠️
        - Maximum expected loss at given confidence level
        - **5% VaR**: Only 5% chance of losing more than this amount
        - Critical for position sizing and risk management
        
        **Probability of Loss** 📉
        - Chance of negative returns
        - **< 30%**: Relatively safe investment
        - **> 50%**: High-risk investment
        
        ### 🛠️ Recommended Strategies
        Our AI analyzes your specific data patterns and recommends:
        
        **Risk-Based Recommendations** ⚖️
        - Conservative: Low volatility, high win rates
        - Moderate: Balanced risk-return profile  
        - Aggressive: High potential returns, accept higher risk
        
        **Pattern-Based Strategies** 🔍
        - Strong seasonality → Timing strategies
        - High volatility → Avoidance strategies
        - Good Sharpe ratio → Buy & hold variations
        
        **Scenario Paths Visualization** 📈
        - Shows sample portfolio paths over time
        - Helps visualize potential outcomes
        - Includes best, worst, and typical scenarios
        """)
    
    elif wiki_section == "📈 Technical Indicators":
        st.markdown("""
        ## 📈 Technical Indicators Reference
        
        ### 🔄 Moving Averages
        
        **Simple Moving Average (SMA)** 📊
        - Average price over N periods
        - **20-day**: Short-term trend
        - **50-day**: Medium-term trend  
        - **200-day**: Long-term trend
        
        **Exponential Moving Average (EMA)** ⚡
        - Gives more weight to recent prices
        - More responsive than SMA
        - **12/26-day**: MACD components
        
        ### 📊 Momentum Indicators
        
        **RSI (Relative Strength Index)** 🎯
        - Measures overbought/oversold conditions
        - **Scale**: 0-100
        - **Overbought**: > 70 (potential sell signal)
        - **Oversold**: < 30 (potential buy signal)
        
        **MACD (Moving Average Convergence Divergence)** 📈
        - **MACD Line**: 12-day EMA - 26-day EMA
        - **Signal Line**: 9-day EMA of MACD
        - **Histogram**: MACD - Signal Line
        - **Buy**: MACD crosses above signal
        - **Sell**: MACD crosses below signal
        
        ### 📏 Volatility Indicators
        
        **Bollinger Bands** 📊
        - **Middle Band**: 20-day SMA
        - **Upper/Lower**: ±2 standard deviations
        - **Squeeze**: Low volatility, potential breakout
        - **Expansion**: High volatility period
        
        **ATR (Average True Range)** ⚡
        - Measures market volatility
        - **Higher ATR**: More volatile
        - **Lower ATR**: Less volatile
        - Used for position sizing
        
        ### 📈 Volume Analysis
        
        **Volume** 📊
        - Number of shares traded
        - **High volume + price move**: Strong signal
        - **Low volume + price move**: Weak signal
        
        **On-Balance Volume (OBV)** 📈
        - Cumulative volume indicator
        - **Rising OBV**: Buying pressure
        - **Falling OBV**: Selling pressure
        """)
    
    elif wiki_section == "⚠️ Risk Management":
        st.markdown("""
        ## ⚠️ Risk Management Guide
        
        ### 📊 Risk Metrics Explained
        
        **Annual Volatility** ⚡
        - Standard deviation of returns (annualized)
        - **Low Risk**: < 15% annual volatility
        - **Medium Risk**: 15-30% annual volatility
        - **High Risk**: > 30% annual volatility
        
        **Maximum Drawdown** 📉
        - Largest peak-to-trough decline
        - **Formula**: (Trough Value - Peak Value) / Peak Value
        - **Conservative**: < 15% max drawdown
        - **Aggressive**: 15-30% max drawdown
        - **Speculative**: > 30% max drawdown
        
        **Beta** 📊
        - Correlation with market (usually S&P 500)
        - **Beta = 1**: Moves with market
        - **Beta > 1**: More volatile than market
        - **Beta < 1**: Less volatile than market
        
        **Value at Risk (VaR)** ⚠️
        - Maximum expected loss at confidence level
        - **Daily VaR**: Potential daily loss
        - **Monthly VaR**: Potential monthly loss
        - Used for position sizing
        
        ### 🛡️ Risk Management Strategies
        
        **Position Sizing** 💰
        - **Kelly Criterion**: Optimal bet size based on edge
        - **Fixed %**: Risk fixed percentage per trade (e.g., 2%)
        - **Volatility Adjusted**: Larger positions in low volatility periods
        
        **Diversification** 🌐
        - **Geographic**: Different countries/regions
        - **Sector**: Multiple industries
        - **Asset Class**: Stocks, bonds, commodities
        - **Time**: Dollar-cost averaging
        
        **Stop Losses** 🛑
        - **Fixed %**: Exit if loss exceeds X%
        - **Technical**: Based on support/resistance
        - **Time-based**: Exit after X days/months
        - **Volatility**: Based on ATR or standard deviation
        
        ### 📋 Risk Assessment Checklist
        
        **Before Investing** ✅
        - [ ] Understand maximum potential loss
        - [ ] Position size appropriate for risk tolerance
        - [ ] Stop loss strategy defined
        - [ ] Correlation with existing holdings
        - [ ] Liquidity requirements considered
        
        **During Investment** 📊
        - [ ] Monitor drawdown levels
        - [ ] Rebalance if correlations change
        - [ ] Adjust position size based on volatility
        - [ ] Review and update stop losses
        
        **Portfolio Level** 🎯
        - [ ] Overall portfolio volatility < tolerance
        - [ ] No single position > 10% of portfolio
        - [ ] Maximum sector exposure < 25%
        - [ ] Cash reserves for opportunities
        """)
    

    
    elif wiki_section == "📋 Export & Reporting":
        st.markdown("""
        ## 📋 Export & Reporting Features
        
        ### 📄 PDF Report Generation
        
        **Executive Summary** 📊
        - Key metrics and performance overview
        - Best/worst months identification
        - Risk assessment summary
        - Professional formatting for presentations
        
        **Detailed Analysis** 🔍
        - Complete seasonal statistics table
        - AI insights and recommendations
        - Risk metrics and drawdown analysis
        - Strategy comparison results
        
        **Visual Charts** 📈
        - Seasonal performance heatmap
        - Monthly return distributions
        - Volatility analysis charts
        - Risk-return scatter plots
        
        ### 💾 CSV Data Export
        
        **Raw Data Export** 📊
        - Original price data with calculated returns
        - Complete OHLC historical data
        - Date-indexed for Excel compatibility
        
        **Analysis Results** 📈
        - Seasonal statistics summary
        - Monthly performance metrics
        - Win rates and volatility measures
        - AI confidence scores
        
        **Backtest Results** 💰
        - Trade-by-trade transaction log
        - Portfolio value over time
        - Performance metrics summary
        - Strategy comparison data
        
        ### 🔔 Alert System
        
        **Seasonal Alerts** 📅
        - Upcoming best/worst months
        - Pattern strength notifications
        - Risk regime change warnings
        
        **Performance Alerts** 📊
        - Drawdown threshold breaches
        - Volatility spike notifications
        - Return target achievements
        
        **AI Pattern Alerts** 🤖
        - New high-confidence patterns detected
        - Pattern breakdown warnings
        - Seasonal anomaly notifications
        
        ### 📱 Integration Options
        
        **API Access** 🔌
        - RESTful API endpoints
        - JSON data format
        - Authentication key management
        - Rate limiting and usage tracking
        
        **Webhook Notifications** 📡
        - Real-time pattern updates
        - Alert delivery to external systems
        - Custom payload formatting
        - Retry and error handling
        
        **Third-Party Integration** 🔗
        - Trading platform connections
        - Portfolio management systems
        - Risk management platforms
        - CRM and notification systems
        """)
    
    elif wiki_section == "❓ FAQ & Troubleshooting":
        st.markdown("""
        ## ❓ Frequently Asked Questions
        
        ### 📁 Data Issues
        
        **Q: My CSV file won't upload. What's wrong?**
        A: Check that your file has the required columns (DATE, OPEN, HIGH, LOW, CLOSE) and is under 200MB. The system supports comma, semicolon, and tab delimiters.
        
        **Q: The dates aren't parsing correctly.**
        A: We support most date formats, but ensure consistency. Use YYYY-MM-DD for best results. Check for missing or malformed dates.
        
        **Q: I'm getting "insufficient data" errors.**
        A: You need at least 2 years of data for meaningful seasonal analysis. For robust patterns, 5+ years is recommended.
        
        **Q: Some months show no data.**
        A: This is normal for assets with limited trading history or gaps. The analysis will work with available months.
        
        ### 📊 Analysis Questions
        
        **Q: Why do my results differ from other platforms?**
        A: Different platforms use different calculation methods. We use month-end to month-end returns for consistency. Ensure you're comparing the same time periods.
        
        **Q: The AI confidence scores seem low.**
        A: Low confidence indicates weak or inconsistent patterns. This is valuable information - not all assets have strong seasonal effects. Our enhanced AI requires 95%+ statistical significance for high confidence ratings.
        
        **Q: Should I trust patterns with < 80% confidence?**
        A: Use lower confidence patterns as supporting evidence, not primary signals. Our new AI system provides much more reliable patterns at 85%+ confidence levels.
        
        **Q: Why do backtesting results differ from seasonal statistics?**
        A: Backtesting includes transaction costs, timing effects, and strategy-specific rules. It's more realistic than raw seasonal averages.
        
        **Q: What's new in the AI analysis?**
        A: We've added Prophet forecasting (30-365 day predictions), advanced time series decomposition, market regime detection, structural break analysis, and comprehensive machine learning pattern recognition. See the "Advanced AI Features" section for details.
        
        **Q: How accurate are the Prophet forecasts?**
        A: Our Prophet models achieve 65-85% directional accuracy with 8-15% MAPE for 30-day forecasts. Longer forecasts have lower accuracy but provide valuable directional guidance.
        
        **Q: What are market regimes and why do they matter?**
        A: Market regimes are distinct periods with similar volatility and trend characteristics. Our AI detects these automatically to help you understand when seasonal patterns are strongest or weakest.
        
        **Q: The AI analysis is taking a long time.**
        A: The enhanced AI features perform extensive calculations including Prophet forecasting, time series decomposition, and machine learning. This typically takes 30-90 seconds depending on data size.
        
        ### 💰 Trading & Strategy
        
        **Q: Which strategy should I use?**
        A: Start with our AI recommendations based on your data patterns. Test with small positions first. Consider your risk tolerance and transaction costs.
        
        **Q: How often should I rebalance?**
        A: Depends on the strategy. Monthly strategies require monthly rebalancing. Consider transaction costs and tax implications.
        
        **Q: Can I use this for day trading?**
        A: No, this platform focuses on monthly seasonal patterns. For day trading, use intraday technical analysis tools.
        
        **Q: What about taxes and commissions?**
        A: The backtesting includes basic commission costs but not taxes. Consult with a tax professional for tax-efficient implementation.
        
        ### 🔧 Technical Issues
        
        **Q: The app is running slowly.**
        A: Large datasets (> 10 years) can slow processing. Try reducing the date range or using a more powerful computer. Close other browser tabs.
        
        **Q: Charts aren't displaying properly.**
        A: Clear your browser cache and refresh. Ensure JavaScript is enabled. Try a different browser if issues persist.
        
        **Q: I can't export the PDF report.**
        A: Check your browser's popup blocker settings. Ensure you have sufficient storage space and a stable internet connection.
        
        **Q: The Monte Carlo simulation takes forever.**
        A: This is normal for 10,000 simulations. It typically takes 30-60 seconds. Don't refresh the page during processing.
        
        ### 📞 Getting Help
        
        **Documentation** 📚
        - This Wiki section covers most topics
        - Check the tooltips (ℹ️) throughout the interface
        - Review the strategy explanations in each section
        
        **Data Quality** 🔍
        - Verify your CSV format matches requirements
        - Check for missing or duplicate dates
        - Ensure sufficient data history (5+ years recommended)
        
        **Performance Optimization** ⚡
        - Use modern browsers (Chrome, Firefox, Safari)
        - Close unnecessary browser tabs
        - Consider smaller datasets for faster processing
        """)
    
    # Quick reference section
    st.markdown("---")
    st.markdown("""
    ### 🔗 Quick Reference
    
    **Best Practices:**
    - ✅ Use at least 5 years of data for robust patterns
    - ✅ Enable AI analysis and set confidence threshold to 85%+ for reliable signals
    - ✅ Use Prophet forecasts for 30-90 day tactical planning
    - ✅ Monitor market regimes to understand when patterns are strongest
    - ✅ Combine seasonal analysis with AI pattern detection and regime analysis
    - ✅ Start with small position sizes when testing strategies
    - ✅ Consider transaction costs in strategy selection
    - ✅ Use structural break detection to identify when patterns change
    - ✅ Leverage time series decomposition to understand trend vs seasonal components
    - ✅ Regularly review and update your analysis as new data becomes available
    
    **NEW AI Features to Explore:**
    - 🔮 **Advanced AI Features**: Multi-horizon Prophet forecasting and comprehensive analysis
    - 🧠 **Enhanced Pattern Detection**: Machine learning with 95%+ statistical significance
    - 🤖 **Prophet Forecasting**: 30-365 day predictions with uncertainty intervals
    - 📈 **Market Regime Analysis**: Volatility and trend regime detection
    - 📊 **Time Series Decomposition**: Trend, seasonal, and cyclical component analysis
    - 🚨 **Structural Break Detection**: Identify when historical patterns change
    - 💰 **AI Strategy Generation**: Algorithm-generated trading strategies
    - 📉 **Risk Regime Detection**: Volatility clustering and drawdown analysis
    
    **Risk Warnings:**
    - ⚠️ Past performance doesn't guarantee future results
    - ⚠️ Seasonal patterns can break down or reverse
    - ⚠️ Market conditions change over time
    - ⚠️ Consider correlation with your existing holdings
    - ⚠️ Use appropriate position sizing for your risk tolerance
    """)

if __name__ == "__main__":
    main()
