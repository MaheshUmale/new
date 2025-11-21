# Import necessary libraries
import traceback
from tradingview_screener import Query, col, And, Or
import pandas as pd
from pymongo import MongoClient
import os
import numpy as np
from tradingview_screener import Query, col, And, Or
import pandas as pd
from pymongo import MongoClient
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

# Set options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# --- Constants ---
# Define timeframes for each strategy type
HIGH_POTENTIAL_TFS = ['|15', '|30', '|60']
HIGH_PROBABILITY_TFS = ['|60', '|240', '']
PULLBACK_SETTINGS = {
    'depth': 0.05,
    'max_bars': 20,
    'max_pullback_pct': 0.01
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
# Define timeframes and their mappings for ordering, display, and suffixing
timeframes = ['|1', '|5', '|15', '|30', '|60', '|120', '|240', '', '|1W', '|1M']
tf_order_map = {'|1': 1, '|5': 2, '|15': 3, '|30': 4, '|60': 5, '|120': 6, '|240': 7, '': 8, '|1W': 9, '|1M': 10}
tf_display_map = {'|1': '1m', '|5': '5m', '|15': '15m', '|30': '30m', '|60': '1H', '|120': '2H', '|240': '4H', '': 'Daily', '|1W': 'Weekly', '|1M': 'Monthly'}
tf_suffix_map = {v: k for k, v in tf_display_map.items()}

# Define weights and scores for potency calculation
POTENCY_WEIGHTS = {'RVOL': 2.0, 'TF_Order': 1.5, 'Breakout_Type_Score': 1.0}
BREAKOUT_TYPE_SCORES = {'Both': 3, 'Squeeze': 2, 'Donchian': 1, 'None': 0}

# Define UI columns
UI_COLUMNS = ['name','close','sector','HTF_EMA20_Price','Upper_BB','Lower_BB','pullback_depth',  'ticker', 'logoid', 'relative_volume_10d_calc', 'Breakout_TFs', 'fired_timestamp', 'momentum', 'breakout_type', 'highest_tf', 'Potency_Score', 'Strategy_Type']

# Construct select columns for all timeframes
select_cols = ['name', 'logoid',  'relative_volume_10d_calc', 'beta_1_year','close|5','EMA9|5','EMA5|5',]

def get_query_for_TFs(settings, allTfs=timeframes):
    """Get the query for squeeze and breakout scans."""
    for tf in allTfs:
        select_cols.extend([
            f'KltChnl.lower{tf}', f'KltChnl.upper{tf}', f'BB.lower{tf}', f'BB.upper{tf}', f'DonchCh20.Upper{tf}', f'DonchCh20.Lower{tf}',
            # f'KltChnl.lower[1]{tf}', f'KltChnl.upper[1]{tf}', f'BB.lower[1]{tf}', f'BB.upper[1]{tf}', f'DonchCh20.Upper[1]{tf}', f'DonchCh20.Lower[1]{tf}',
            f'ATRP{tf}', f'SMA20{tf}', f'volume{tf}', f'average_volume_10d_calc{tf}', f'close{tf}', f'Value.Traded{tf}', f'MACD.hist{tf}', f'MACD.hist[1]{tf}',
            f'ADX{tf}', f'RSI{tf}', f'EMA20{tf}', f'SMA50{tf}'
        ])
    base_filters = [
        col('beta_1_year') > settings['beta_1_year'],
        col('is_primary') == True,
        col('typespecs').has(['', 'common', 'foreign-issuer']),
        col('type').isin(['dr', 'stock']),
        col('close').between(settings['min_price'], settings['max_price']),
        col('active_symbol') == True,
        col('Value.Traded|5') > settings['min_value_traded'],
        col('exchange').isin(['NSE' ]), 

    ]
    donchian_break = [Or(col(f'DonchCh20.Upper{tf}') > col(f'DonchCh20.Upper[1]{tf}'), col(f'DonchCh20.Lower{tf}') < col(f'DonchCh20.Lower[1]{tf}')) for tf in HIGH_POTENTIAL_TFS]
    squeeze_breakout = [Or(And(col(f'BB.upper[1]{tf}') < col(f'KltChnl.upper[1]{tf}'), col(f'BB.upper{tf}') >= col(f'KltChnl.upper{tf}')), And(col(f'BB.lower[1]{tf}') > col(f'KltChnl.lower[1]{tf}'), col(f'BB.lower{tf}') <= col(f'KltChnl.lower[1]{tf}'))) for tf in HIGH_POTENTIAL_TFS]
    vol_spike = [Or(col(f'volume{tf}').above_pct(col(f'average_volume_10d_calc{tf}'), settings['RVOL_threshold'] - 1), col(f'relative_volume_10d_calc{tf}') > settings['RVOL_threshold'], col('relative_volume_intraday|5') > settings['RVOL_threshold']) for tf in HIGH_POTENTIAL_TFS]

    filters = [And(*base_filters,*vol_spike)]####, Or(*squeeze_breakout,*donchian_break)]
    Q =  Query().select(*select_cols).where2(And(*filters))
    Q.set_markets(settings['market'])
    # Q.set_markets(['futures',  'options'])
    Q.limit(1000).set_property('symbols', {'query': {}})
    # print("Constructed Query:")
    # print(Q)
    return Q



# def get_squeeze_breakout_query(settings, high_potential_tfs=HIGH_POTENTIAL_TFS):
#     """Get the query for squeeze and breakout scans."""
#     for tf in high_potential_tfs:
#         select_cols.extend([
#             f'KltChnl.lower{tf}', f'KltChnl.upper{tf}', f'BB.lower{tf}', f'BB.upper{tf}', f'DonchCh20.Upper{tf}', f'DonchCh20.Lower{tf}',
#             f'KltChnl.lower[1]{tf}', f'KltChnl.upper[1]{tf}', f'BB.lower[1]{tf}', f'BB.upper[1]{tf}', f'DonchCh20.Upper[1]{tf}', f'DonchCh20.Lower[1]{tf}',
#             f'ATRP{tf}', f'SMA20{tf}', f'volume{tf}', f'average_volume_10d_calc{tf}', f'close{tf}', f'Value.Traded{tf}', f'MACD.hist{tf}', f'MACD.hist[1]{tf}',
#             f'ADX{tf}', f'RSI{tf}', f'EMA20{tf}', f'SMA50{tf}'
#         ])
#     base_filters = [
#         col('beta_1_year') > settings['beta_1_year'],
#         col('is_primary') == True,
#         col('typespecs').has(['', 'common', 'foreign-issuer']),
#         col('type').isin(['dr', 'stock']),
#         col('close').between(settings['min_price'], settings['max_price']),
#         col('active_symbol') == True,
#         col('Value.Traded|5') > settings['min_value_traded'],
#     ]
#     donchian_break = [Or(col(f'DonchCh20.Upper{tf}') > col(f'DonchCh20.Upper[1]{tf}'), col(f'DonchCh20.Lower{tf}') < col(f'DonchCh20.Lower[1]{tf}')) for tf in HIGH_POTENTIAL_TFS]
#     squeeze_breakout = [Or(And(col(f'BB.upper[1]{tf}') < col(f'KltChnl.upper[1]{tf}'), col(f'BB.upper{tf}') >= col(f'KltChnl.upper{tf}')), And(col(f'BB.lower[1]{tf}') > col(f'KltChnl.lower[1]{tf}'), col(f'BB.lower{tf}') <= col(f'KltChnl.lower[1]{tf}'))) for tf in HIGH_POTENTIAL_TFS]
#     vol_spike = [Or(col(f'volume{tf}').above_pct(col(f'average_volume_10d_calc{tf}'), settings['RVOL_threshold'] - 1), col(f'relative_volume_10d_calc{tf}') > settings['RVOL_threshold'], col('relative_volume_intraday|5') > settings['RVOL_threshold']) for tf in HIGH_POTENTIAL_TFS]

#     filters = [And(*base_filters), And(*vol_spike), Or(*squeeze_breakout, *donchian_break)]
#     return Query().select(*select_cols).where2(And(*filters)).set_markets(settings['market']).limit(1000).set_property('symbols', {'query': {}})

# def get_trending_pullback_query(settings, high_probability_tfs=HIGH_PROBABILITY_TFS):

#     """Get the query for trending and pullback scans."""
#     for tf in high_probability_tfs:
#         select_cols.extend([
#             f'KltChnl.lower{tf}', f'KltChnl.upper{tf}', f'BB.lower{tf}', f'BB.upper{tf}', f'DonchCh20.Upper{tf}', f'DonchCh20.Lower{tf}',
#             f'KltChnl.lower[1]{tf}', f'KltChnl.upper[1]{tf}', f'BB.lower[1]{tf}', f'BB.upper[1]{tf}', f'DonchCh20.Upper[1]{tf}', f'DonchCh20.Lower[1]{tf}',
#             f'ATRP{tf}', f'SMA20{tf}', f'volume{tf}', f'average_volume_10d_calc{tf}', f'close{tf}', f'Value.Traded{tf}', f'MACD.hist{tf}', f'MACD.hist[1]{tf}',
#             f'ADX{tf}', f'RSI{tf}', f'EMA20{tf}', f'SMA50{tf}'
#         ])
#     base_filters = [
#         col('beta_1_year') > settings['beta_1_year'],
#         col('is_primary') == True,
#         col('typespecs').has(['', 'common', 'foreign-issuer']),
#         col('type').isin(['dr', 'stock']),
#         col('close').between(settings['min_price'], settings['max_price']),
#         col('active_symbol') == True,
#         col('Value.Traded|5') > settings['min_value_traded'],
#     ]
#     trend_filters = [And(col(f'ADX{tf}').between(TREND_SETTINGS['adx_min'], TREND_SETTINGS['adx_max']),
#                          col(f'close{tf}') > col(f'SMA50{tf}'),  col(f'EMA20{tf}') > col(f'EMA50{tf}'),    # EMA alignment
#                         # Use RSI to identify pullback instead of percentage range
#                          col(f'RSI{tf}').between(30, 60)  # RSI in pullback zone
#                          ) for tf in HIGH_PROBABILITY_TFS]
# # Trend strength conditions - simplified to avoid multiplication issues
#     # trend_conditions = And(
#     #         col(f'ADX{tf}') >= TREND_SETTINGS['adx_min'],
#     #         col(f'ADX{tf}') <= TREND_SETTINGS['adx_max'],
#     #         col(f'close{tf}') > col(f'SMA50{tf}'),  # Above 50 SMA
          
#     #     )
#     filters = [And(*base_filters), Or(*trend_filters)]
#     return Query().select(*select_cols).where2(And(*filters)).set_markets(settings['market']).limit(1000).set_property('symbols', {'query': {}})

def run_scan(settings, cookies, db_name):
    """Run both scans, combine the results, and save to the database."""
    if cookies is None:
        return {"fired": pd.DataFrame()}

    try:
        all_tfs = set(HIGH_POTENTIAL_TFS + HIGH_PROBABILITY_TFS)
        _, df_raw = get_query_for_TFs(settings, allTfs=all_tfs).get_scanner_data(cookies=cookies)
        
        print(f"Scans completed.   Combined: {len(df_raw)}")

         
        if df_raw.empty:
            return {'fired': pd.DataFrame(), 'alerts': []}
        # 3. Clean and prepare data (remove N/A values, etc.)
        df_clean = df_raw.dropna(subset=['close'])   #.set_index('name')
        
        # 4. Calculate Potency Score and initial metadata
        df_setups = _calculate_potency_and_metadata(df_clean, settings)
        
        
        print(f"Setups after Potency Calculation: {len(df_setups)}")
        
        
        if df_setups.empty:
            return {'fired': pd.DataFrame(), 'alerts': []}
        print("DataFrame columns after potency calculation:")
        # print(df_setups.columns)
        # df_setups = _reorder_final_columns(df_setups)
        # 5. Filter for High Potential and High Probability
        high_potential_df = df_setups[df_setups['Breakout_TFs'].apply(lambda tfs: any(tf in tfs for tf in [tf_display_map[t] for t in HIGH_POTENTIAL_TFS]))].copy()
        high_probability_df = df_setups[df_setups['Breakout_TFs'].apply(lambda tfs: any(tf in tfs for tf in [tf_display_map[t] for t in HIGH_PROBABILITY_TFS]))].copy()

        # Assign Strategy Type for final display
        high_potential_df['Strategy_Type'] = 'High_Potential'
        high_probability_df['Strategy_Type'] = 'High_Probability'
        
        # Combine the two filtered sets, handling duplicates (e.g., if a stock meets both)
        final_df = pd.concat([high_potential_df, high_probability_df]).drop_duplicates(subset=['ticker', 'Breakout_TFs'], keep='first')
        
        if final_df.empty:
            return {'fired': pd.DataFrame(), 'alerts': []}

            # 6. Check for Real-Time Alerts
        # Alerts require C_5m and EMA9_5m, which are already present in df_clean/df_setups
        # print("Checking for alerts...")
        # print("---------------------------------------------")
        # print("Final DataFrame columns before alert check:")
        # # print(final_df.columns)
        # # print(final_df.head())
        # print("---------------------------------------------")
        current_alerts = _check_alerts(final_df, db_name)
        df_filtered = final_df[[col for col in UI_COLUMNS if col in final_df.columns]].copy()
        # 7. Save and return results
        save_scan_results(df_filtered, db_name)

        return {'fired': df_filtered, 'alerts': current_alerts}

         
    except Exception as e:
        import traceback
        traceback.print_exc()
    #     return {"fired": pd.DataFrame()}

    # df_New = df.drop_duplicates(subset=['name']).copy()
    # df_New = df_New[df_New['breakout_type'] != 'None'].copy()
    # if df_New.empty:
    #     return {"fired": pd.DataFrame()}
    # df_New['fired_timestamp'] = pd.Timestamp.now()
    # df_New['momentum'] = df_New['momentum_score'].apply(lambda x: 'Bullish' if x > 0 else ('Bearish' if x < 0 else 'Neutral'))
    # df_New = df_New.replace([float('inf'), float('-inf')], None).where(pd.notnull(df_New), None)
    # df_filtered = df_New[[col for col in UI_COLUMNS if col in df_New.columns]].copy()
    
    # save_scan_results(df_filtered, db_name)
    # return {"fired": df_filtered}

def _reorder_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorders the columns for clean storage and display."""
    
    required_cols = ['Breakout_TFs', 'HTF_EMA20_Price', 'Potency_Score', 'Strategy_Type', 'Upper_BB', 'Lower_BB', # Added Lower_BB
                     'breakout_type', 'close', 'fired_timestamp', 'highest_tf', 'logoid', 'momentum', 'name', 
                     'pullback_depth', 'relative_volume_10d_calc', 'rsi', 'ticker']
    
    # Ensure all required columns exist, adding them if they don't
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    return df[required_cols].sort_values(by='Potency_Score', ascending=False)


def _calculate_potency_and_metadata(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """Calculates Potency Score, Breakout Metadata, and Momentum."""
    if df.empty: return df

    print(f"Calculating Potency Score and Metadata...len(df)={len(df)}  ")
    # --- 1. Scoring & Filtering for Breakout Type ---
    # Prepare map for easy column referencing
    tf_suffix_map = {tf_display_map[k]: k for k in tf_display_map} # {'1m': '|1', '5m': '|5', ...}
    
    # Initialize scoring columns
    df['Breakout_Type_Score'] = 0
    df['TF_Order'] = 0

    # Identify duplicate column names
    duplicate_columns = df.columns.duplicated()

    # Create a new DataFrame keeping only the non-duplicated columns
    df_cleaned = df.loc[:, ~duplicate_columns].copy()
    df = df_cleaned.copy()
    
    try:
    # Iterate through all TFs to check for breakouts and score
        for tf in df.columns:
            if tf.startswith('BB.upper'): # Only check columns that are part of the scan TFs
                tf_suffix = tf.replace('BB.upper', '')
                
                # 1. Check for Bullish Breakout (Close > Upper BB)
                bullish_breakout_col = f"BULL_BREAK_{tf_suffix}"

                print(f'tf_suffix: {tf_suffix}')
                # Check the shape of the columns you are using (optional, for debugging)
                # print(df[f'close{tf_suffix}'].shape) 
                # print(df[f'BB.upper{tf_suffix}'].shape) 
                # Compare the underlying numpy arrays directly
                # comparison_result = (df[f'close{tf_suffix}'].values > df[f'BB.upper{tf_suffix}'].values)
                
                comparison_result = (df[f'close{tf_suffix}'] > df[f'BB.upper{tf_suffix}'])

                # Assign the result back to the DataFrame (pandas handles aligning this back correctly)
                df[bullish_breakout_col] = comparison_result.astype(int) * 10

                # df[bullish_breakout_col] = (df[f'close{tf_suffix}'] > df[f'BB.upper{tf_suffix}']).astype(int) * 10
                df['Breakout_Type_Score'] += df[bullish_breakout_col]

                # 2. Check for Bearish Breakout (Close < Lower BB) <-- NEW BEARISH LOGIC
                bearish_breakout_col = f"BEAR_BREAK_{tf_suffix}"
                df[bearish_breakout_col] = (df[f'close{tf_suffix}'] < df[f'BB.lower{tf_suffix}']).astype(int) * 10
                df['Breakout_Type_Score'] += df[bearish_breakout_col]
                
                # If any breakout happens, record the TF order (higher is better)
                df.loc[(df[bullish_breakout_col] > 0) | (df[bearish_breakout_col] > 0), 'TF_Order'] = df.loc[(df[bullish_breakout_col] > 0) | (df[bearish_breakout_col] > 0), 'TF_Order'].apply(lambda x: max(x, tf_order_map.get(tf_suffix, 0)))
    except Exception as e:
        print(f"Error during breakout scoring: {e}")
        # print(df.columns)
        traceback.print_exc()
    # Filter: Only keep setups with a breakout on at least one TF (Breakout_Type_Score > 0)
    # print(df[df['Breakout_Type_Score'] > 0])
    df_fired = df[df['Breakout_Type_Score'] > 0].copy()
    if df_fired.empty:
        return df_fired

    # --- 2. Final Metadata Calculation (For Display) ---

    # Map back suffixes to display names for final columns
    df_fired['C_5m'] = df_fired['close|5']
    df_fired['EMA5_5m'] = df_fired['EMA5|5']
    df_fired['Breakout_TFs'] = ''
    df_fired['highest_tf'] = ''
    df_fired['Upper_BB'] = 0.0 # Will hold the highest TF Upper BB
    df_fired['Lower_BB'] = 0.0 # Will hold the highest TF Lower BB <-- NEW
    df_fired['HTF_EMA20_Price'] = 0.0
    df_fired['pullback_depth'] = 0.0
    # df_fired['adx'] = 0.0
    # df_fired['rsi'] = 0.0
    df_fired['momentum'] = 'Neutral'
    df_fired['breakout_type'] = 'None'
    df_fired['Potency_Score'] = 0.0
    # if not exist fired_timestamp, create it
    if 'fired_timestamp' not in df_fired.columns:
        print("Creating 'fired_timestamp' column in df_fired.---------------------------------------------------------")
        df_fired['fired_timestamp'] = pd.Timestamp.now()
    # Iterate through fired setups to compile metadata
    for index, row in df_fired.iterrows():
        fired_tfs = []
        highest_order = 0
        highest_tf_name = ''
        
        bullish_count = 0
        bearish_count = 0
        
        for tf_name, tf_suffix in tf_suffix_map.items():
            bullish_col = f"BULL_BREAK_{tf_suffix}"
            bearish_col = f"BEAR_BREAK_{tf_suffix}"
            
            # Check if this TF triggered a breakout
            if row.get(bullish_col) or row.get(bearish_col):
                fired_tfs.append(tf_name)
                current_order = tf_order_map.get(tf_suffix)
                
                if current_order > highest_order:
                    highest_order = current_order
                    highest_tf_name = tf_name
                
                if row.get(bullish_col):
                    bullish_count += 1
                if row.get(bearish_col): # <-- Count bearish breakouts
                    bearish_count += 1

        row['Breakout_TFs'] = ','.join(fired_tfs)
        row['highest_tf'] = highest_tf_name
        
        # Determine breakout type
        if bullish_count > 0 and bearish_count > 0:
            row['breakout_type'] = 'Both'
        elif bullish_count > 0:
            row['breakout_type'] = 'Upside Squeeze'
        elif bearish_count > 0:
            row['breakout_type'] = 'Downside Squeeze' # <-- NEW
        else:
            row['breakout_type'] = 'Squeeze'
            
        # Get Key Levels from the Highest TF that fired
        if highest_tf_name:
            highest_tf_suffix = tf_suffix_map[highest_tf_name]
            row['Upper_BB'] = row[f'BB.upper{highest_tf_suffix}']
            row['Lower_BB'] = row[f'BB.lower{highest_tf_suffix}'] # <-- NEW
            row['HTF_EMA20_Price'] = row[f'EMA20{highest_tf_suffix}']
            
            # Determine overall momentum based on highest TF close vs EMA20
            # Also calculate pullback depth based on breakout direction
            
            # Bullish check (price > EMA20 and not Overbought on RSI)
            if row[f'close{highest_tf_suffix}'] > row['HTF_EMA20_Price'] and row[f'RSI{highest_tf_suffix}'] < TREND_SETTINGS['rsi_overbought']:
                row['momentum'] = 'Bullish'
                row['pullback_depth'] = (row['HTF_EMA20_Price'] - row['close']) / row['HTF_EMA20_Price'] # Positive means price is below EMA (pullback potential)

            # Bearish check (price < EMA20 and not Oversold on RSI)
            elif row[f'close{highest_tf_suffix}'] < row['HTF_EMA20_Price'] and row[f'RSI{highest_tf_suffix}'] > TREND_SETTINGS['rsi_oversold']: # <-- NEW BEARISH MOMENTUM
                row['momentum'] = 'Bearish'
                row['pullback_depth'] = (row['close'] - row['HTF_EMA20_Price']) / row['HTF_EMA20_Price'] # Positive means price is above EMA (downside pullback potential)

            else:
                row['momentum'] = 'Neutral'
                row['pullback_depth'] = 0.0

            # --- Potency Score Calculation ---
            # Potency is calculated here, I will use a simple example:
            # 1. Strength (ADX)
            # 2. RVOL
            # 3. Number of TFs fired
            
            adx_val = row[f'ADX{highest_tf_suffix}']
            rvol_val = row['relative_volume_10d_calc'] if pd.notna(row['relative_volume_10d_calc']) else 0.1
            
            # Score: ADX strength + RVOL factor + Number of TFs * 5
            score = (adx_val / TREND_SETTINGS['adx_max']) * 40 + \
                    min(rvol_val, 2) * 20 + \
                    len(fired_tfs) * 5 # Max 50, Total Max 100
            
            row['Potency_Score'] = score
            
        df_fired.loc[index] = row # Update the row in the final DF
        
    # df_fired['relative_volume_10d_calc'] = df_fired['RVOL_10D']

    # Final cleanup and sort
    print(f"Final fired setups: {len(df_fired)}")
    print("Columns in fired df:")
    # print(df_fired.columns)

    # print( df_fired[['Breakout_TFs','Potency_Score']].head(10))

    df_fired = df_fired.sort_values(by='Potency_Score', ascending=False)
    df_fired['Potency_Score'] = df_fired['Potency_Score'].round(2)
    # df_fired = _reorder_final_columns(df_fired)
    return df_fired.drop(columns=['Breakout_Type_Score', 'TF_Order'], errors='ignore')


def _check_alerts(df_merged: pd.DataFrame, db_name: str) -> List[Dict]:
    """
    Checks for real-time trigger alerts based on the merged setup data.
    Adds Downside Pullback Fade logic.
    """
    print("Checking for alerts.. AFTER SCANNING QUERY ------------------------.")
    fired_alerts = []
   
    # 1. Breakout Confirmation (High Potential)
    # Trigger: Price is above Upper BB (Bullish) or below Lower BB (Bearish) on the current 5m candle
    breakout_potential = df_merged[df_merged['Strategy_Type'] == 'High_Potential']
    print("_check_alerts trigger conditions on merged data...")
    # print(breakout_potential.columns)
    # print(breakout_potential.head())
    # Bullish Confirmation: Close > Upper BB
    upside_breakout_fired = breakout_potential[
        (breakout_potential['momentum'] != 'Bearish') &
        (breakout_potential['C_5m'] > breakout_potential['Upper_BB'])
    ]
    for index, row in upside_breakout_fired.iterrows():
        alert = {
            'symbol': row['ticker'],
            'trigger_type': 'Upside Breakout',
            'setup_tf': row['Breakout_TFs'],
            'current_price': row['C_5m'],
            'trigger_level': row['Upper_BB'],
            'details': f"Price closed above the {row['Breakout_TFs']} Upper BB: {row['Upper_BB']:.2f}"
        }
        save_alert_result(alert, db_name)
        fired_alerts.append(alert)

    # Bearish Confirmation: Close < Lower BB <-- NEW BEARISH CONFIRMATION
    downside_breakout_fired = breakout_potential[
        (breakout_potential['momentum'] != 'Bullish') &
        (breakout_potential['C_5m'] < breakout_potential['Lower_BB'])
    ]
    for index, row in downside_breakout_fired.iterrows():
        alert = {
            'symbol': row['ticker'],
            'trigger_type': 'Downside Breakout', # Clear trigger name
            'setup_tf': row['Breakout_TFs'],
            'current_price': row['C_5m'],
            'trigger_level': row['Lower_BB'],
            'details': f"Price closed below the {row['Breakout_TFs']} Lower BB: {row['Lower_BB']:.2f}"
        }
        save_alert_result(alert, db_name)
        fired_alerts.append(alert)

    # 2. Pullback Bounce / Fade (High Probability)
    pullback_potential = df_merged[df_merged['Strategy_Type'] == 'High_Probability']
    
    # print("Pullback potential momentum:")
    # print(pullback_potential[['momentum', 'C_5m', 'HTF_EMA20_Price', 'EMA5_5m']])
    # --- BULLISH PULLBACK BOUNCE (Price pulls back to support and bounces) ---
    is_bullish_pullback = (pullback_potential['momentum'] == 'Bullish') & \
                          (pullback_potential['C_5m'] < pullback_potential['HTF_EMA20_Price'] * (1 + PULLBACK_SETTINGS['max_pullback_pct'])) & \
                          (pullback_potential['C_5m'] > pullback_potential['HTF_EMA20_Price'] * (1 - PULLBACK_SETTINGS['max_pullback_pct'])) & \
                          (pullback_potential['C_5m'] > pullback_potential['EMA5_5m']) # 5m close above 5m EMA5 (momentum confirmation)
    
    bullish_pullback_fired = pullback_potential[is_bullish_pullback]

    for index, row in bullish_pullback_fired.iterrows():
        alert = {
            'symbol': row['ticker'],
            'trigger_type': 'Upside Pullback Bounce',
            'setup_tf': row['Breakout_TFs'],
            'current_price': row['C_5m'],
            'trigger_level': row['HTF_EMA20_Price'],
            'details': f"Bounced off EMA20 support at {row['HTF_EMA20_Price']:.2f} with 5m momentum confirmation."
        }
        print("Bullish pullback alert:")
        print(alert)
        save_alert_result(alert, db_name)
        fired_alerts.append(alert)

    # --- BEARISH PULLBACK FADE (Price pulls back to resistance and fades) --- <-- NEW BEARISH PULLBACK
    is_bearish_pullback = (pullback_potential['momentum'] == 'Bearish') & \
                          (pullback_potential['C_5m'] > pullback_potential['HTF_EMA20_Price'] * (1 - PULLBACK_SETTINGS['max_pullback_pct'])) & \
                          (pullback_potential['C_5m'] < pullback_potential['HTF_EMA20_Price'] * (1 + PULLBACK_SETTINGS['max_pullback_pct'])) & \
                          (pullback_potential['C_5m'] < pullback_potential['EMA5_5m']) # 5m close below 5m EMA5 (downside momentum confirmation)

    bearish_pullback_fired = pullback_potential[is_bearish_pullback]

    for index, row in bearish_pullback_fired.iterrows():
        alert = {
            'symbol': row['ticker'],
            'trigger_type': 'Downside Pullback Fade',
            'setup_tf': row['Breakout_TFs'],
            'current_price': row['C_5m'],
            'trigger_level': row['HTF_EMA20_Price'],
            'details': f"Faded from EMA20 resistance at {row['HTF_EMA20_Price']:.2f} with 5m momentum confirmation."
        }
        print("Bearish pullback alert:")
        print(alert)
        save_alert_result(alert, db_name)
        fired_alerts.append(alert)

    return fired_alerts




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
    df['highest_tf_suffix'] = df['Breakout_TFs'].apply(lambda tfs: get_highest_timeframe_suffix(tfs, tf_suffix_map))
    df['rsi'] = df.apply(lambda row: row[f'RSI{row["highest_tf_suffix"]}'] if row["highest_tf_suffix"] in tf_suffix_map.values() else None, axis=1)
    df['adx'] = df.apply(lambda row: row[f'ADX{row["highest_tf_suffix"]}'] if row["highest_tf_suffix"] in tf_suffix_map.values() else None, axis=1)
    ##set SMA50{row["highest_tf_suffix"] to 1 ide if missing to avoid NaN in pullback_depth calculation
    try:
     df['pullback_depth'] = df.apply(lambda row: (row[f'SMA50{row["highest_tf_suffix"]}'] - row[f'close{row["highest_tf_suffix"]}']) / row[f'SMA50{row["highest_tf_suffix"]}'] if row["highest_tf_suffix"] in tf_suffix_map.values() else 0, axis=1)
    except Exception as e:
        print(f"Error calculating pullback_depth: {e}")
        # df['pullback_depth'] = df['pullback_depth'].replace([np.inf, -np.inf], np.nan).fillna(0)
        #print stack trace
        import traceback
        traceback.print_exc()
    df['HTF_EMA20_Price'] = df.apply(lambda row: row[f'EMA20{row["highest_tf_suffix"]}'] if row["highest_tf_suffix"] in tf_suffix_map.values() else None, axis=1)
    df['Upper_BB'] = df.apply(lambda row: row[f'BB.upper{row["highest_tf_suffix"]}'] if row["highest_tf_suffix"] in tf_suffix_map.values() else None, axis=1)   
    df['Lower_BB'] = df.apply(lambda row: row[f'BB.lower{row["highest_tf_suffix"]}'] if row["highest_tf_suffix"] in tf_suffix_map.values() else None, axis=1)
      
    return df
def get_highest_timeframe(breakout_tfs_str, order_map, suffix_map):
    """Determine the highest timeframe from a comma-separated string."""
    if not isinstance(breakout_tfs_str, str) or not breakout_tfs_str:
        return 'Unknown'
    tfs = breakout_tfs_str.split(',')
    return max(tfs, key=lambda tf: order_map.get(suffix_map.get(tf, ''), 0))
##get highest_timeframe suffix from breakout_tfs_str
def get_highest_timeframe_suffix(breakout_tfs_str, suffix_map):
    """Get the highest timeframe suffix from a comma-separated string."""
    if not isinstance(breakout_tfs_str, str) or not breakout_tfs_str:
        return 'Unknown'
    tfs = breakout_tfs_str.split(',')
    return max([suffix_map.get(tf, '') for tf in tfs], key=lambda s: tf_order_map.get(s, 0))


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

# --- MongoDB Functions ---
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
COLLECTION_NAME = "scan_results"

TRIGGER_ALERTS_COLLECTION_NAME = "fired_alerts" # New collection for triggers


def get_mongo_collection(db_name, collection_name=COLLECTION_NAME):
    """Get the MongoDB collection."""
    client = MongoClient(MONGO_URI)
    return client[db_name][collection_name]

def save_scan_results(df: pd.DataFrame, db_name: str):
    """Save scan results to MongoDB, removing null values first."""
    if df.empty: return
    """Save scan results to MongoDB, ensuring required trigger levels are present."""
    # if df.empty: return
    # IMPORTANT: These fields must be calculated in your full run_scan implementation
    df['Upper_BB'] = df.get('Upper_BB', np.nan).fillna(-1)
    df['Lower_BB'] = df.get('Lower_BB', np.nan).fillna(-1)
    df['HTF_EMA20_Price'] = df.get('HTF_EMA20_Price', np.nan).fillna(-1)
    print("Saving scan results to MongoDB...")
    # print(df.head() )
    # df.dropna(subset=['name'], inplace=True)
    

    collection = get_mongo_collection(db_name, COLLECTION_NAME)
    # print(df.columns)
    for record in df.to_dict(orient='records'):
        query = {'name': record['name']}
        update_doc = {**record, 'last_updated': pd.Timestamp.now()}
        original_timestamp = update_doc.pop('fired_timestamp', None)
        collection.update_one(query, {'$set': update_doc, '$setOnInsert': {'fired_timestamp': original_timestamp}}, upsert=True)

def load_scan_results(db_name: str) -> pd.DataFrame:
    """Load scan results from MongoDB."""
    collection = get_mongo_collection(db_name, COLLECTION_NAME)
    records = list(collection.find({}))
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    if '_id' in df.columns: df = df.drop(columns=['_id'])
    if 'Potency_Score' in df.columns: df = df.sort_values(by='Potency_Score', ascending=False)
    # df['price'] = df['close']
    return df

def save_alert_result(alert: Dict, db_name: str):
    """Save a fired trigger alert to the dedicated alerts collection (Step 2C)."""
    alert['triggered_at'] = datetime.now()
    collection = get_mongo_collection(db_name, TRIGGER_ALERTS_COLLECTION_NAME)
    # Use symbol and trigger type to update, ensuring we don't spam the DB with the same active alert
    collection.update_one(
        {'symbol': alert['symbol'], 'trigger_type': alert['trigger_type']},
        {'$set': alert, '$setOnInsert': {'first_triggered': alert['triggered_at']}},
        upsert=True
    )
    # print(f"ALERT FIRED: {alert['symbol']} - {alert['trigger_type']} at {alert['current_price']:.2f}")

def load_fired_alerts(db_name: str) -> pd.DataFrame:
    """Load current fired alerts from MongoDB (Step 2C)."""
    collection = get_mongo_collection(db_name, TRIGGER_ALERTS_COLLECTION_NAME)
    # Only load alerts triggered in the last 24 hours
    cutoff = datetime.now() - pd.Timedelta(days=1)
    records = list(collection.find({'triggered_at': {'$gte': cutoff}}))
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    df['triggered_at'] = pd.to_datetime(df['triggered_at'])
    return df.sort_values(by='triggered_at', ascending=False)



# --- Trigger Data Fetching (Simulation of Real-Time Check) ---

def get_trigger_data_from_tv(symbol_list: List[str], cookies: Optional[Dict]) -> pd.DataFrame:
    """
    Fetches required 5m data (OHLC, EMA9, EMA20) for the watchlist 
    just before the candle closes using tradingview_screener.
    
    FIXED: Using set_tickers for efficient watchlist filtering.
    """
    if not symbol_list:
        return pd.DataFrame()
    
    tf = '|5'
    query = Query().select(
        f'close{tf}',
        f'EMA5{tf}',
        f'EMA20{tf}',
        f'name' # Ensure 'name' is selected for merging later
    )
    
    # --- CORRECT IMPLEMENTATION: Use set_tickers for specific watchlist ---
    query.set_tickers(*symbol_list)
    # ---------------------------------------------------------------------
    
    try:
        count,data = query.get_scanner_data(cookies=cookies)
        df = pd.DataFrame(data)
        df.rename(columns={
            f'close{tf}': 'C_5m',
            f'EMA5{tf}': 'EMA5_5m',
            f'EMA20{tf}': 'EMA20_5m',
        }, inplace=True)
        return df.set_index('name')
    except Exception as e:
        print(f"Error fetching trigger data from TV: {e}")
        #print stack trace
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# --- Automated Trigger Logic (Step 2B) ---

def check_trigger_conditions(df_setups: pd.DataFrame, df_realtime: pd.DataFrame, db_name: str):
    """
    Applies trigger rules to stocks that have an active setup.
    """
    print("Checking trigger conditions...")
    if df_setups.empty or df_realtime.empty:
        print("No active setups or real-time data available.")
        return []
    
    fired_alerts = []
    
    # Merge the setup data (HTF levels) with the 5m real-time data
    # df_merged = df_setups.join(df_realtime, on=['name','ticker'], how='inner')
    df_merged = pd.merge(df_setups, df_realtime, on=['name', 'ticker'], how='inner')
    if df_merged.empty:
        print("No merged data available for trigger conditions.")   
        return []

    return _check_alerts(df_merged, db_name)

    # print("Checking trigger conditions on merged data...")
    # # print(df_merged.columns)
    # # print(df_merged.head())
    # # 1. Squeeze Fire (High Potential)
    # # Trigger: C_5m closes above the HTF Upper_BB and below Lower_BB (stored in setup)
    # squeeze_potential = df_merged[df_merged['Strategy_Type'] == 'High_Potential']
    # squeeze_firedUpper = squeeze_potential[
    #     (squeeze_potential['C_5m'] > squeeze_potential['Upper_BB']) & 
    #     (squeeze_potential['Upper_BB'] > 0) # Ensure Upper_BB was stored correctly
    # ]
    # squeeze_firedLower = squeeze_potential[
    #     (squeeze_potential['C_5m'] < squeeze_potential['Lower_BB']) & 
    #     (squeeze_potential['Lower_BB'] > 0) # Ensure Lower_BB was stored correctly
    # ]
    # for symbol, row in  squeeze_firedUpper.iterrows():
    #     alert = {
    #         'symbol': symbol,
    #         'trigger_type': 'Squeeze Fire',
    #         'setup_tf': row['Breakout_TFs'],
    #         'current_price': row['C_5m'],
    #         'setup_level': row['Upper_BB'],
    #         'details': f"Price closed above the {row['Breakout_TFs']} Upper BB: {row['Upper_BB']:.2f}"
    #     }
    #     save_alert_result(alert, db_name)
    #     fired_alerts.append(alert)

    # for symbol, row in squeeze_firedLower.iterrows():
    #     alert = {
    #         'symbol': symbol,
    #         'trigger_type': 'Squeeze Fire',
    #         'setup_tf': row['Breakout_TFs'],
    #         'current_price': row['C_5m'],
    #         'setup_level': row['Lower_BB'],
    #         'details': f"Price closed below the {row['Breakout_TFs']} Lower BB: {row['Lower_BB']:.2f}"
    #     }
    #     save_alert_result(alert, db_name)
    #     fired_alerts.append(alert)

    # # 2. Pullback Bounce (High Probability)
    # # Trigger: Price is near the HTF_EMA20 AND 5m C_5m > 5m EMA9 (momentum confirmation)
    # pullback_potential = df_merged[df_merged['Strategy_Type'] == 'High_Probability']
    
    # # Condition A: Price is near EMA20 (within 1.0% range) - Confirms the setup is still valid/active
    # is_near_ema20 = (pullback_potential['C_5m'] < pullback_potential['HTF_EMA20_Price'] * 1.02) & \
    #                 (pullback_potential['C_5m'] > pullback_potential['HTF_EMA20_Price'] * 0.98)
    
    # # Condition B: Momentum Reversal (5m close is above 5m EMA9)
    # momentum_reversal = pullback_potential['C_5m'] > pullback_potential['EMA5_5m']
    
    # pullback_fired = pullback_potential[is_near_ema20 & momentum_reversal]
    
    # for symbol, row in pullback_fired.iterrows():
    #     alert = {
    #         'symbol': symbol,
    #         'trigger_type': 'Pullback Bounce',
    #         'setup_tf': row['Breakout_TFs'],
    #         'current_price': row['C_5m'],
    #         'setup_level': row['HTF_EMA20_Price'],
    #         'details': f"Bounced off {row['Breakout_TFs']} EMA20 ({row['HTF_EMA20_Price']:.2f}) with 5m momentum shift (C > EMA9)."
    #     }
    #     save_alert_result(alert, db_name)
    #     fired_alerts.append(alert)

    # return fired_alerts

