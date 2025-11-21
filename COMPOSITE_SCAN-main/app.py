import traceback
from flask import Flask, jsonify, render_template, request
import pandas as pd
import json
import rookiepy
import threading
import time
from datetime import datetime, timedelta
# Import the new functions from scan.py
from scan import run_scan, load_scan_results, get_trigger_data_from_tv, check_trigger_conditions, load_fired_alerts

app = Flask(__name__)

# --- In-memory cache for scan results and alerts ---
global scan_results_cache, fired_alerts_cache

scan_results_cache = pd.DataFrame()
fired_alerts_cache = pd.DataFrame()

# --- Scanner Settings (Assume these are managed globally or via DB) ---
SCANNER_SETTINGS = {
    'beta_1_year': 1.2,
    'min_price': 10,
    'max_price': 50000,
    'min_value_traded': 10000000,
    'RVOL_threshold': 2,
    'market': 'india'
}

# --- Database Name (Consistent across scan.py and app.py) ---
DB_NAME = 'composite_scan'

# --- Background Scanner Thread (Slow, for Setups) ---
def run_scanner_periodically():
    """Runs the full market scan periodically (slower, for setups)."""
    while True:
        cookies = None
        try:
            cookies = rookiepy.to_cookiejar(rookiepy.brave(['.tradingview.com']))
        except Exception as e:
            print(f"Could not get cookies: {e}")

        try:
            # run_scan calculates and saves all high-quality setups and trigger levels
            results = run_scan(SCANNER_SETTINGS, cookies, DB_NAME)
            global scan_results_cache
            scan_results_cache = results['fired']
            print(f"Full Scan complete. Found {len(scan_results_cache)} setups.")
        except Exception as e:
            print(f"An error occurred during scanning: {e}")
        time.sleep(60)  # Run every 1 minute

# --- Real-Time Trigger Detector Thread (Fast, for Triggers) ---
def run_trigger_detector_periodically():
    """
    Runs trigger check ~20 seconds before every 5-minute candle closes, 
    using TV data to simulate final closing values (Step 2B).
    """
    global fired_alerts_cache
    
    while True:
        now = datetime.now()
        # Calculate seconds until the next 5-minute boundary (0, 5, 10, 15...)
        seconds_to_next_5m = 300 - (now.minute % 5 * 60 + now.second)
        
        # We want to check 20 seconds before the 5m candle close.
        check_offset = 30
        
        if seconds_to_next_5m <= 300 and seconds_to_next_5m > check_offset:
            # Sleep until the exact check time (20 seconds before the close)
            ##FIXME INTENTIAL SPEEDUP FOR TESTING
            sleep_time = seconds_to_next_5m - check_offset
            print(f"Trigger Detector: Sleeping for {sleep_time:.1f}s to wait for pre-close window...")
            time.sleep(sleep_time)
            
            # --- EXECUTE TRIGGER CHECK ---
            
            # 1. Get watchlist from the latest setup scan results (DB is primary source)
            watchlist_df = load_scan_results(DB_NAME)
            print(f"Trigger Detector: Loaded watchlist with {len(watchlist_df)} stocks.")
            watchlist_symbols = watchlist_df['ticker'].unique().tolist()
            # print(f"Trigger Detector: Watchlist symbols: {watchlist_symbols}")
            if not watchlist_symbols:
                print("Trigger Detector: Watchlist empty. Skipping check.")
                time.sleep(check_offset)
                continue

            # 2. Get 5m data for the watchlist from TV (simulates current candle close)
            cookies = None
            try:
                cookies = rookiepy.to_cookiejar(rookiepy.brave(['.tradingview.com']))
            except Exception as e:
                print(f"Could not get cookies for trigger check: {e}")
                
            df_realtime = get_trigger_data_from_tv(watchlist_symbols, cookies)
            print(f"Trigger Detector: Retrieved real-time data for {len(df_realtime)} stocks.")
            # 3. Apply the trigger logic and save results to the alerts collection
            try:
                fired = check_trigger_conditions(watchlist_df, df_realtime, DB_NAME)
                print(f"Trigger Detector: Checked {len(watchlist_symbols)} stocks. {len(fired)} alerts fired.")
                
                # Update the alerts cache for the dashboard
                fired_alerts_cache = load_fired_alerts(DB_NAME)
                
            except Exception as e:
                print(f"An error occurred during trigger checking: {e}")
                traceback.print_exc()
                
            # Sleep for the remaining 20 seconds of the cycle
            time.sleep(check_offset) 

        else:
            # If we missed the window, wait for 1 second and re-evaluate
            time.sleep(1) 


# --- API Endpoints ---
@app.route('/')
def index():
    """Serves the main alerts."""
    return render_template('index.html')

@app.route('/alerts_dashboard')
def alerts_dashboard():
    """Serves the main alerts."""
    return render_template('alerts.html')

@app.route('/data')
def data():
    """Serves the latest full scan results (setups)."""
    # ... existing logic to serve scan_results_cache ...
    global scan_results_cache
    if not scan_results_cache.empty:
        return jsonify(scan_results_cache.to_dict(orient='records'))
    else:
        db_results = load_scan_results(DB_NAME)
        if not db_results.empty:
            
            scan_results_cache = db_results
            return jsonify(db_results.to_dict(orient='records'))
        else:
            return jsonify([])

@app.route('/alerts')
def alerts():
    """Serves the latest fired alerts (triggers) (Step 2C)."""
    global fired_alerts_cache
    
    # Load from DB to ensure it's fresh
    fired_alerts_cache = load_fired_alerts(DB_NAME)

    if not fired_alerts_cache.empty:
        # Clean up Pandas Timestamp objects for JSON serialization
        df = fired_alerts_cache.copy()
        # df['triggered_at'] = df['triggered_at'].dt.isoformat()
        
        df['triggered_at'] = df['triggered_at'].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        
        if '_id' in df.columns: df = df.drop(columns=['_id'])

        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify([])

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    # ... existing settings management ...
    if request.method == 'POST':
        # Update settings from the form
        return jsonify({"message": "Settings updated successfully"})
    return jsonify(SCANNER_SETTINGS)


if __name__ == '__main__':
    # 1. Start the background scanner thread (Slow, for Setups)
    scanner_thread = threading.Thread(target=run_scanner_periodically, daemon=True)
    scanner_thread.start()
    
    # 2. Start the trigger detector thread (Fast, for Pinpoint Triggers)
    trigger_detector_thread = threading.Thread(target=run_trigger_detector_periodically, daemon=True)
    trigger_detector_thread.start()
    
    # 3. Start the Flask Web Server
    app.run(debug=False, port=5001, use_reloader=False)