
# Import necessary libraries
from tradingview_screener import Query, col, And, Or
import pandas as pd
from pymongo import MongoClient
import os

# --- Constants ---
# Define volume thresholds for different timeframes
VOLUME_THRESHOLDS = {
    '|1': 25000, '|5': 100000, '|15': 300000, '|30': 500000,
    '|60': 1000000, '|120': 2000000, '|240': 4000000, '': 5000000,
    '|1W': 25000000, '|1M': 100000000
}

# Define timeframes and their mappings for ordering, display, and suffixing
timeframes = ['|1', '|5', '|15', '|30', '|60', '|120', '|240', '', '|1W', '|1M']
tf_order_map = {'|1': 1, '|5': 2, '|15': 3, '|30': 4, '|60': 5, '|120': 6, '|240': 7, '': 8, '|1W': 9, '|1M': 10}
tf_display_map = {'|1': '1m', '|5': '5m', '|15': '15m', '|30': '30m', '|60': '1H', '|120': '2H', '|240': '4H', '': 'Daily', '|1W': 'Weekly', '|1M': 'Monthly'}
tf_suffix_map = {v: k for k, v in tf_display_map.items()}

# Define weights and scores for potency calculation
POTENCY_WEIGHTS = {'RVOL': 2.0, 'TF_Order': 1.5, 'Breakout_Type_Score': 1.0}
BREAKOUT_TYPE_SCORES = {'Both': 3, 'Squeeze': 2, 'Donchian': 1, 'None': 0}

# Timeframes for scanning
TIMEFRAMES = ['15m', '30m', '60m', '1d']
HIGH_POTENTIAL_TIMEFRAMES = ['15m', '30m']  # For squeeze/breakout
HIGH_PROBABILITY_TIMEFRAMES = ['60m', '1d']  # For trending/pullback

# Strategy types
STRATEGY_TYPES = {
    'HIGH_POTENTIAL': 'High Potential (Squeeze/Breakout)',
    'HIGH_PROBABILITY': 'High Probability (Trending/Pullback)'
}

# Define UI columns
UI_COLUMNS = ['name', 'ticker', 'logoid', 'relative_volume_10d_calc', 'Breakout_TFs', 'fired_timestamp', 'momentum', 'breakout_type', 'highest_tf', 'Potency_Score', 'Strategy_Type']

# Trending/Pullback specific columns
TRENDING_COLUMNS = ['name', 'ticker', 'logoid', 'relative_volume_10d_calc', 'trend_strength', 'pullback_depth', 'timeframe', 'momentum', 'Potency_Score', 'Strategy_Type']

# Construct select columns for all timeframes
select_cols = ['name', 'logoid', 'close', 'relative_volume_10d_calc', 'beta_1_year']
for tf in timeframes:
    select_cols.extend([
        f'KltChnl.lower{tf}', f'KltChnl.upper{tf}', f'BB.lower{tf}', f'BB.upper{tf}', f'DonchCh20.Upper{tf}', f'DonchCh20.Lower{tf}',
        f'KltChnl.lower[1]{tf}', f'KltChnl.upper[1]{tf}', f'BB.lower[1]{tf}', f'BB.upper[1]{tf}', f'DonchCh20.Upper[1]{tf}', f'DonchCh20.Lower[1]{tf}',
        f'ATRP{tf}', f'SMA20{tf}', f'volume{tf}', f'average_volume_10d_calc{tf}', f'close{tf}', f'Value.Traded{tf}', f'MACD.hist{tf}', f'MACD.hist[1]{tf}'
    ])

def run_intraday_scan(settings, cookies, db_name):
    """Run the intraday scan for breakouts."""
    if cookies is None:
        return {"fired": pd.DataFrame()}

    # Define base filters for the scan
    base_filters = [
        col('beta_1_year') > settings['beta_1_year'],
        col('is_primary') == True,
        col('typespecs').has(['', 'common', 'preferred','foreign-issuer']),
        col('type').isin(['dr', 'stock']),
        col('close').between(settings['min_price'], settings['max_price']),
        col('active_symbol') == True,
        col('Value.Traded|5') > settings['min_value_traded'],
    ]
    # Define trend, breakout, and volume spike filters
    trendFilter = [Or(And(col(f'EMA20{tf}') > col(f'EMA200{tf}'), col(f'close{tf}') > col(f'EMA20{tf}')), And(col(f'EMA20{tf}') < col(f'EMA200{tf}'), col(f'close{tf}') < col(f'EMA20{tf}'))) for tf in timeframes]
    donchian_break = [Or(col(f'DonchCh20.Upper{tf}') > col(f'DonchCh20.Upper[1]{tf}'), col(f'DonchCh20.Lower{tf}') < col(f'DonchCh20.Lower[1]{tf}')) for tf in timeframes]
    squeeze_breakout = [Or(And(col(f'BB.upper[1]{tf}') < col(f'KltChnl.upper[1]{tf}'), col(f'BB.upper{tf}') >= col(f'KltChnl.upper{tf}')), And(col(f'BB.lower[1]{tf}') > col(f'KltChnl.lower[1]{tf}'), col(f'BB.lower{tf}') <= col(f'KltChnl.lower[1]{tf}'))) for tf in timeframes]
    RVol_spike = [Or( Or(col(f'volume{tf}').above_pct(col(f'average_volume_10d_calc{tf}'), settings['RVOL_threshold']), Or(col(f'relative_volume_10d_calc{tf}') > settings['RVOL_threshold'], col('relative_volume_intraday|5') > settings['RVOL_threshold']))) for tf in timeframes]
    # vol_filters = [col(f'volume{tf}').above_pct(col(f'average_volume_10d_calc{tf}'), settings['RVOL_threshold']) for tf in timeframes]
    # Combine filters and create query
    # base_filters += vol_filters
    filters = [And(*base_filters),   Or(And(*RVol_spike ,*trendFilter), And(*RVol_spike,*squeeze_breakout), And(*RVol_spike,*donchian_break))]
    filtersEasy = [And(*base_filters), And(*RVol_spike), Or(*trendFilter,*squeeze_breakout,*donchian_break)]
    filtersSuperEasy  = [And(*base_filters),  And(*RVol_spike,Or(*trendFilter,*squeeze_breakout,*donchian_break))]
    filtersHard = [And(*base_filters), And(*RVol_spike), And(*trendFilter), And(*squeeze_breakout),And(*donchian_break)]
    query = Query().select(*select_cols).where2(And(*filtersEasy)).set_markets(settings['market']).limit(1000).set_property('symbols', {'query': {}})

    try:
        _, df = query.get_scanner_data(cookies=cookies)
        if df is not None:
            df = df.fillna(value=pd.NA).replace({pd.NA: None}).replace([float('inf'), float('-inf')], None).where(pd.notnull(df), None)
            if not df.empty:
                print(f"Scan found {len(df)} breakouts.")
                df = identify_combined_breakout_timeframes(df, timeframes)
                df = calculate_potency_score(df, tf_order_map, tf_suffix_map)
            else:
                print("No breakouts found in scan.")
                return {"fired": pd.DataFrame()}
        else:
            return {"fired": pd.DataFrame()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"fired": pd.DataFrame()}

    # Process and save new breakouts
    df_New = df.drop_duplicates(subset=['name']).copy()
    df_New = df_New[df_New['breakout_type'] != 'None'].copy()
    if df_New.empty:
        return {"fired": pd.DataFrame()}
    df_New['fired_timestamp'] = pd.Timestamp.now()
    df_New['momentum'] = df_New['momentum_score'].apply(lambda x: 'Bullish' if x > 0 else ('Bearish' if x < 0 else 'Neutral'))
    df_New['Strategy_Type'] = STRATEGY_TYPES['HIGH_POTENTIAL']
    df_New = df_New.replace([float('inf'), float('-inf')], None).where(pd.notnull(df_New), None)
    df_filtered = df_New[[col for col in UI_COLUMNS if col in df_New.columns]].copy()
    save_scan_results(df_filtered, db_name)
    return {"fired": df_filtered}

def identify_combined_breakout_timeframes(df: pd.DataFrame, timeframes: list) -> pd.DataFrame:
    """Identify and combine breakout timeframes."""
    if df is None or df.empty:
        return pd.DataFrame()
    squeeze_indicators, donchian_indicators = {}, {}
    for tf in timeframes:
        upper_breakout = (df[f'BB.upper[1]{tf}'] < df[f'KltChnl.upper[1]{tf}']) & (df[f'BB.upper{tf}'] >= df[f'KltChnl.upper{tf}'])
        lower_breakout = (df[f'BB.lower[1]{tf}'] > df[f'KltChnl.lower[1]{tf}']) & (df[f'BB.lower{tf}'] <= df[f'KltChnl.lower{tf}'])
        squeeze_indicators[tf] = upper_breakout | lower_breakout
        donchian_indicators[tf] = (df[f'DonchCh20.Upper{tf}'] > df[f'DonchCh20.Upper[1]{tf}']) | (df[f'DonchCh20.Lower{tf}'] < df[f'DonchCh20.Lower[1]{tf}'])
    squeeze_df, donchian_df = pd.DataFrame(squeeze_indicators, index=df.index), pd.DataFrame(donchian_indicators, index=df.index)
    combined_df = squeeze_df | donchian_df
    def get_breakout_type(row):
        is_squeeze, is_donchian = any(squeeze_df.loc[row.name]), any(donchian_df.loc[row.name])
        if is_squeeze and is_donchian: return 'Both'
        elif is_squeeze: return 'Squeeze'
        elif is_donchian: return 'Donchian'
        return 'None'
    df['breakout_type'] = df.apply(get_breakout_type, axis=1)
    df['Breakout_TFs'] = combined_df.apply(lambda row: ','.join([tf_display_map.get(tf, tf) for tf, is_breakout in row.items() if is_breakout]), axis=1)
    df['highest_tf'] = df['Breakout_TFs'].apply(lambda tfs: get_highest_timeframe(tfs, tf_order_map, tf_suffix_map))
    return df

def get_highest_timeframe(breakout_tfs_str, order_map, suffix_map):
    """Determine the highest timeframe from a comma-separated string."""
    if not isinstance(breakout_tfs_str, str) or not breakout_tfs_str:
        return 'Unknown'
    tfs = breakout_tfs_str.split(',')
    return max(tfs, key=lambda tf: order_map.get(suffix_map.get(tf, ''), 0))

def calculate_potency_score(df: pd.DataFrame, tf_order_map: dict, tf_suffix_map: dict) -> pd.DataFrame:
    """Calculate the potency score for each stock."""
    if df.empty:
        return df
    df['volume_score'], df['momentum_score'], df['ATR_score'] = 0.0, 0.0, 0.0
    for index, row in df.iterrows():
        for tf_display in row['Breakout_TFs'].split(','):
            tf = tf_suffix_map.get(tf_display)
            if tf and f'volume{tf}' in df.columns and f'average_volume_10d_calc{tf}' in df.columns:
                safe_avg_vol = row[f'average_volume_10d_calc{tf}'] if row[f'average_volume_10d_calc{tf}'] > 0 else 1
                df.loc[index, 'volume_score'] += (row[f'volume{tf}'] / safe_avg_vol) * tf_order_map.get(tf, 0)
            if tf and f'MACD.hist{tf}' in df.columns and f'MACD.hist[1]{tf}' in df.columns:
                value_to_add = (pd.to_numeric(row[f'MACD.hist{tf}'], errors='coerce') - pd.to_numeric(row[f'MACD.hist[1]{tf}'], errors='coerce')) / tf_order_map.get(tf, 1)
                df.loc[index, 'momentum_score'] += value_to_add
            if tf and f'ATRP{tf}' in df.columns:
                df.loc[index, 'ATR_score'] += row[f'ATRP{tf}']
    df['RVOL_Score'] = df['volume_score'] / len(timeframes)
    df['Breakout_Type_Score'] = df['breakout_type'].map(BREAKOUT_TYPE_SCORES).fillna(0)
    df['TF_Order'] = df['highest_tf'].apply(lambda tf_display_name: tf_order_map.get(tf_suffix_map.get(tf_display_name), 0))
    df['Potency_Score'] = ((df['relative_volume_10d_calc'] * POTENCY_WEIGHTS['RVOL']) + (df['TF_Order'] * POTENCY_WEIGHTS['TF_Order']) + (df['Breakout_Type_Score'] * POTENCY_WEIGHTS['Breakout_Type_Score']) + (df['RVOL_Score'] * 0.5) + abs((df['momentum_score'] * 0.5)) + (df['ATR_score']))
    df = df.sort_values(by='Potency_Score', ascending=False)
    df['Potency_Score'] = df['Potency_Score'].round(2)
    return df.drop(columns=['Breakout_Type_Score', 'TF_Order'], errors='ignore')

def calculate_trending_potency_score(row):
    """
    Calculate a potency score for trending/pullback setups using available TradingView fields.
    """
    score = 0
    
    # Helper to ensure scalar values
    def get_scalar_value(value, default=0):
        if hasattr(value, 'iloc'):
            return value.iloc[0] if not value.empty else default
        return value if pd.notna(value) else default

    # RSI and price data for trend analysis
    rsi_60m = get_scalar_value(row.get('RSI|60m'), 50)
    rsi_1d = get_scalar_value(row.get('RSI'), 50)
    close_60m = get_scalar_value(row.get('close|60m'), 0)
    close_1d = get_scalar_value(row.get('close'), 0)
    sma20_60m = get_scalar_value(row.get('SMA20|60m'), 0)
    sma20_1d = get_scalar_value(row.get('SMA20'), 0)

    # Trend Strength (40%)
    def calculate_trend_strength(rsi, close, sma20):
        if close > sma20:
            return max(0, min(100, (rsi - 50) * 2))  # Bullish
        return max(0, min(100, (50 - rsi) * 2))      # Bearish

    trend_score = max(calculate_trend_strength(rsi_60m, close_60m, sma20_60m), 
                      calculate_trend_strength(rsi_1d, close_1d, sma20_1d))
    score += trend_score * 0.4

    # Pullback Depth (30%)
    pullback_depth = abs(50 - rsi_1d if rsi_1d != 50 else rsi_60m)
    pullback_score = max(0, 100 - (abs(pullback_depth - 15) / 15 * 100)) if 5 <= pullback_depth <= 25 else 0
    score += pullback_score * 0.3

    # Volume Confirmation (20%)
    rvol = get_scalar_value(row.get('relative_volume_10d_calc'), 0)
    volume_score = min(rvol / 2.5, 1) * 100  # Normalize against a 2.5 RVOL benchmark
    score += volume_score * 0.2
    
    # Momentum (10%)
    momentum_score = 100 if rsi_1d > 55 or rsi_60m > 60 else (50 if 45 <= rsi_1d <= 55 else 0)
    score += momentum_score * 0.1

    return round(score, 1)

# --- MongoDB Functions ---
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
COLLECTION_NAME = "scan_results"

def get_mongo_collection(db_name):
    """Get the MongoDB collection."""
    client = MongoClient(MONGO_URI)
    return client[db_name][COLLECTION_NAME]

# def get_combined_breakout_timeframes_query():
#     """
#     Returns a query that identifies stocks with breakouts in multiple timeframes.
#     This is used to find high-potential squeeze and breakout setups.
#     """
#     return f"""
#     SELECT 
#         name, ticker, logoid,
#         relative_volume_10d_calc,
#         "price_52_week_high", "price_52_week_low", "Beta (1Y)",
#         "price_earnings_ttm", "price_book_fq", "market_cap_basic",
#         "Recommend.Other|1W", "Recommend.All|1W", "Recommend.MA|1W",
#         "Squeeze_15m", "Squeeze_30m", "Squeeze_60m", "Squeeze_1d",
#         "Donchian_15m", "Donchian_30m", "Donchian_60m", "Donchian_1d",
#         "breakout_type_15m", "breakout_type_30m", "breakout_type_60m", "breakout_type_1d",
#         "breakout_score_15m", "breakout_score_30m", "breakout_score_60m", "breakout_score_1d",
#         "momentum_15m", "momentum_30m", "momentum_60m", "momentum_1d",
#         "volume_spike_15m", "volume_spike_30m", "volume_spike_60m", "volume_spike_1d",
#         "volatility_increase_15m", "volatility_increase_30m", "volatility_increase_60m", "volatility_increase_1d",
#         "fired_timestamp_15m", "fired_timestamp_30m", "fired_timestamp_60m", "fired_timestamp_1d",
#         "fired_rvol_15m", "fired_rvol_30m", "fired_rvol_60m", "fired_rvol_1d"
#     FROM 
#         "SCREENER_INTRADAY"
#     WHERE
#         (
#             ("Squeeze_15m" = TRUE OR "Donchian_15m" = TRUE OR "breakout_type_15m" != 'None') OR
#             ("Squeeze_30m" = TRUE OR "Donchian_30m" = TRUE OR "breakout_type_30m" != 'None') OR
#             ("Squeeze_60m" = TRUE OR "Donchian_60m" = TRUE OR "breakout_type_60m" != 'None') OR
#             ("Squeeze_1d" = TRUE OR "Donchian_1d" = TRUE OR "breakout_type_1d" != 'None')
#         )
#         AND relative_volume_10d_calc >= {MIN_RVOL}
#         AND (close >= {MIN_PRICE} AND close <= {MAX_PRICE})
#         AND volume >= {MIN_VOLUME}
#         AND (close * volume) >= {MIN_VALUE_TRADED}
#     ORDER BY relative_volume_10d_calc DESC
#     """

# def get_trending_pullback_query():
#     """
#     Returns a query that identifies stocks in established trends with pullbacks.
#     This is used to find high-probability trending/pullback setups.
#     """
#     return f"""
#     SELECT 
#         name, ticker, logoid,
#         relative_volume_10d_calc,
#         "price_52_week_high", "price_52_week_low", "Beta (1Y)",
#         "price_earnings_ttm", "price_book_fq", "market_cap_basic",
#         "Recommend.Other|1W", "Recommend.All|1W", "Recommend.MA|1W",
#         "RSI", "RSI|60m",
#         "close", "close|60m",
#         "SMA20", "SMA20|60m", "SMA50", "SMA50|60m", "SMA200", "SMA200|60m",
#         "volume", "volume|60m",
#         "MACD.macd", "MACD.signal", "MACD.macd|60m", "MACD.signal|60m"
#     FROM 
#         "SCREENER_INTRADAY"
#     WHERE
#         (
#             ("RSI|60m" BETWEEN 40 AND 60 AND "close|60m" > "SMA20|60m" AND "volume|60m" > "average_volume_10d_calc|60m" * 1.2) OR
#             ("RSI" BETWEEN 45 AND 65 AND "close" > "SMA20" AND "volume" > "average_volume_10d_calc" * 1.5)
#         )
#         AND relative_volume_10d_calc >= 1.5
#         AND (close >= {MIN_PRICE} AND close <= {MAX_PRICE})
#         AND volume >= {MIN_VOLUME}
#         AND (close * volume) >= {MIN_VALUE_TRADED}
#         AND "Beta (1Y)" >= 0.8
#     ORDER BY 
#         CASE 
#             WHEN "RSI" BETWEEN 45 AND 65 AND "close" > "SMA20" AND "volume" > "average_volume_10d_calc" * 1.5 THEN 1
#             WHEN "RSI|60m" BETWEEN 40 AND 60 AND "close|60m" > "SMA20|60m" AND "volume|60m" > "average_volume_10d_calc|60m" * 1.2 THEN 2
#             ELSE 3
#         END,
#         relative_volume_10d_calc DESC
#     """

def save_scan_results(df: pd.DataFrame, db_name: str):
    """Save scan results to MongoDB, removing null values first."""
    if df.empty: return
    df.dropna(inplace=True)
    if df.empty: return
    collection = get_mongo_collection(db_name)
    for record in df.to_dict(orient='records'):
        query = {'name': record['name'], 'Breakout_TFs': record.get('Breakout_TFs')}
        update_doc = {**record, 'last_updated': pd.Timestamp.now()}
        original_timestamp = update_doc.pop('fired_timestamp', None)
        collection.update_one(query, {'$set': update_doc, '$setOnInsert': {'fired_timestamp': original_timestamp}}, upsert=True)

def load_scan_results(db_name: str) -> pd.DataFrame:
    """Load scan results from MongoDB."""
    collection = get_mongo_collection(db_name)
    records = list(collection.find({}))
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    if '_id' in df.columns: df = df.drop(columns=['_id'])
    if 'Potency_Score' in df.columns: df = df.sort_values(by='Potency_Score', ascending=False)
    return df