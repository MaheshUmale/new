"""

Configuration file for Multi-Timeframe, Multi-Strategy Scanner System
"""

# Timeframe configurations
HIGH_POTENTIAL_TFS = ['|15', '|30', '|60']  # Squeeze/Breakout scanners
HIGH_PROBABILITY_TFS = ['|60', '|240', '']   # Trending/Pullback scanners

# Market settings
DEFAULT_MARKET = 'america'
DEFAULT_EXCHANGE = 'NASDAQ'
MIN_PRICE = 5.0
MAX_PRICE = 500.0
MIN_VOLUME = 100000
MIN_VALUE_TRADED = 1000000

# Strategy-specific settings
SQUEEZE_SETTINGS = {
    'bb_period': 20,
    'kc_period': 20,
    'atr_period': 14,
    'squeeze_lookback': 5,
    'min_volume_multiplier': 1.2
}

TREND_SETTINGS = {
    'adx_min': 25,
    'adx_max': 60,
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

# Database settings
MONGODB_URI = 'mongodb://localhost:27017/'
DB_NAME = 'multi_strategy_scanner'
COLLECTION_NAME = 'scan_results'

# Flask server settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Scanning intervals (seconds)
SCAN_INTERVAL = 300  # 5 minutes
CLEANUP_INTERVAL = 3600  # 1 hour

# Scoring weights
POTENCY_SCORE_WEIGHTS = {
    'trend_strength': 0.3,
    'pullback_quality': 0.25,
    'volume_confirmation': 0.2,
    'momentum_score': 0.15,
    'technical_alignment': 0.1
}

# UI settings
MAX_RESULTS_PER_STRATEGY = 50
REFRESH_INTERVAL = 30000  # 30 seconds
HEATMAP_COLORS = {
    'high_potential': '#00ff00',  # Green
    'high_probability': '#ff6600',  # Orange
    'neutral': '#ffff00'  # Yellow
}