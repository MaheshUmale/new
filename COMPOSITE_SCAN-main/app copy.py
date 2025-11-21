from flask import Flask, jsonify, render_template, request
import pandas as pd
import json
from scan import run_scan, load_scan_results
import rookiepy
import threading
import time

app = Flask(__name__)

# --- In-memory cache for scan results ---
scan_results_cache = pd.DataFrame()

# --- Scanner Settings ---
# Default settings
SCANNER_SETTINGS = {
    'beta_1_year': 1.5,
    'min_price': 10,
    'max_price': 50000,
    'min_value_traded': 1000000,
    'RVOL_threshold': 2,
    'market': 'america'
}

# --- Background Scanner Thread ---
def run_scanner_periodically():
    """Runs the scanner periodically."""
    while True:
        cookies = None
        try:
            cookies = rookiepy.to_cookiejar(rookiepy.brave(['.tradingview.com']))
        except Exception as e:
            print(f"Could not get cookies: {e}")

        try:
            results = run_scan(SCANNER_SETTINGS, cookies, 'composite_scan')
            global scan_results_cache
            scan_results_cache = results['fired']
            print(f"Scan complete. Found {len(scan_results_cache)} results.")
        except Exception as e:
            print(f"An error occurred during scanning: {e}")
        time.sleep(60)  # Run every 60 seconds

@app.route('/')
def index():
    """Serves the main dashboard."""
    return render_template('index.html')

@app.route('/data')
def data():
    """Provides the combined scan results as JSON."""
    if not scan_results_cache.empty:
        return jsonify(scan_results_cache.to_dict(orient='records'))
    else:
        # If the cache is empty, try loading from the database
        db_results = load_scan_results('composite_scan')
        if not db_results.empty:
            return jsonify(db_results.to_dict(orient='records'))
        else:
            return jsonify([])

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Manages scanner settings."""
    if request.method == 'POST':
        # Update settings from the form
        SCANNER_SETTINGS['beta_1_year'] = float(request.form.get('beta_1_year', SCANNER_SETTINGS['beta_1_year']))
        SCANNER_SETTINGS['min_price'] = int(request.form.get('min_price', SCANNER_SETTINGS['min_price']))
        SCANNER_SETTINGS['max_price'] = int(request.form.get('max_price', SCANNER_SETTINGS['max_price']))
        SCANNER_SETTINGS['min_value_traded'] = int(request.form.get('min_value_traded', SCANNER_SETTINGS['min_value_traded']))
        SCANNER_SETTINGS['RVOL_threshold'] = float(request.form.get('RVOL_threshold', SCANNER_SETTINGS['RVOL_threshold']))
        SCANNER_SETTINGS['market'] = request.form.get('market', SCANNER_SETTINGS['market'])
        return jsonify({"message": "Settings updated successfully"})
    return jsonify(SCANNER_SETTINGS)

if __name__ == '__main__':
    # Start the background scanner thread
    scanner_thread = threading.Thread(target=run_scanner_periodically)
    scanner_thread.daemon = True
    scanner_thread.start()
    app.run(debug=False, port=5001)
