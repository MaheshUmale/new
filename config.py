"""
Configuration file for Multi-Timeframe, Multi-Strategy Scanner System
"""

# Timeframe configurations
HIGH_POTENTIAL_TFS = ['|15', '|30', '|60']  # Squeeze/Breakout scanners
HIGH_PROBABILITY_TFS = ['|60', '|240', '']   # Trending/Pullback scanners

# Market settings
DEFAULT_MARKET = 'india'
DEFAULT_EXCHANGE = 'NSE'
MIN_PRICE = 10.0
MAX_PRICE = 100000.0
MIN_VOLUME = 100000
MIN_VALUE_TRADED = 10000000
MIN_BETA = 1.2

# Strategy-specific settings
SQUEEZE_SETTINGS = {
    'bb_period': 20,
    'kc_period': 20,
    'atr_period': 14,
    'squeeze_lookback': 5,
    'min_volume_multiplier': 1.2,
    'sqz_width_factor': 0.75
}

TREND_SETTINGS = {
    'adx_min': 25,
    'adx_max': 75,
    'sma_period': 50,
    'ema_period': 20,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'min_trend_strength': 0.8
}

PULLBACK_SETTINGS = {
    'max_pullback_depth': 0.05,  # 5% max pullback
    'min_pullback_depth': 0.01,  # 1% min pullback
    'ema_support_period': 20,
    'volume_confirmation': 1.5,
    'atr_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
}

# --- Potency Score Weights and Risk Management Constants ---
WEIGHTS = {
    'W_RVOL': 25,
    'W_TF': 20,
    'W_SETUP': 10,
    'MAX_RISK_RS': 1000.0,
    'MAX_CAPITAL_RS': 100000.0,
    'RR_RATIO': 2.0,
    'ATR_SL_MULTIPLE': 2.0,
    'SQZ_WIDTH_FACTOR': 0.75
}


# Database settings
MONGODB_URI = 'mongodb://localhost:27017/'
DB_NAME = 'multi_strategy_scanner'
COLLECTION_NAME = 'scan_results'

# Flask server settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Scanning intervals (seconds)
SCAN_INTERVAL = 60
CLEANUP_INTERVAL = 3600  # 1 hour

# UI settings
MAX_RESULTS_PER_STRATEGY = 50
REFRESH_INTERVAL = 30000  # 30 seconds
HEATMAP_COLORS = {
    'high_potential': '#00ff00',  # Green
    'high_probability': '#ff6600',  # Orange
    'neutral': '#ffff00'  # Yellow
}
