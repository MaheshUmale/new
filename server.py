"""
Flask server for Multi-Strategy Scanner System
Provides API endpoints for serving scan results and managing the scanner
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
import time
import os

from multi_strategy_scanner import get_scanner
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global scanner instance
scanner = get_scanner()

def start_background_scanner():
    """Start the background scanner thread"""
    def scanner_worker():
        try:
            scanner.start_continuous_scanning()
            logger.info("Background scanner started successfully")
        except Exception as e:
            logger.error(f"Failed to start background scanner: {e}")

    scanner_thread = threading.Thread(target=scanner_worker, daemon=True)
    scanner_thread.start()

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_scan_data():
    """Get latest scan results"""
    try:
        df = scanner.get_latest_results()

        if df.empty:
            return jsonify({
                'long_breakout': [],
                'short_breakout': [],
                'long_continuation': [],
                'short_continuation': [],
                'last_updated': datetime.now().isoformat(),
                'total_results': 0
            })

        long_breakout = df[df['strategy_type'] == 'long_breakout'].to_dict('records')
        short_breakout = df[df['strategy_type'] == 'short_breakout'].to_dict('records')
        long_continuation = df[df['strategy_type'] == 'long_continuation'].to_dict('records')
        short_continuation = df[df['strategy_type'] == 'short_continuation'].to_dict('records')

        for record in long_breakout + short_breakout + long_continuation + short_continuation:
            for key, value in record.items():
                if isinstance(value, datetime):
                    record[key] = value.isoformat()
                elif pd.isna(value):
                    record[key] = None

        return jsonify({
            'long_breakout': long_breakout,
            'short_breakout': short_breakout,
            'long_continuation': long_continuation,
            'short_continuation': short_continuation,
            'last_updated': datetime.now().isoformat(),
            'total_results': len(df)
        })

    except Exception as e:
        logger.error(f"Error getting scan data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/run-scan', methods=['POST'])
def run_manual_scan():
    """Trigger a manual scan"""
    try:
        logger.info("Manual scan triggered")

        def scan_worker():
            try:
                scanner.run_full_scan()
                logger.info("Manual scan completed successfully")
            except Exception as e:
                logger.error(f"Manual scan failed: {e}")

        scan_thread = threading.Thread(target=scan_worker, daemon=True)
        scan_thread.start()

        return jsonify({
            'status': 'Scan initiated',
            'message': 'Scan is running in background. Results will be available shortly.'
        })

    except Exception as e:
        logger.error(f"Error triggering manual scan: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def manage_settings():
    """Get or update scanner settings"""
    if request.method == 'GET':
        return jsonify({
            'scan_interval': SCAN_INTERVAL,
            'max_results_per_strategy': MAX_RESULTS_PER_STRATEGY,
            'min_price': MIN_PRICE,
            'max_price': MAX_PRICE,
            'min_volume': MIN_VOLUME,
            'min_value_traded': MIN_VALUE_TRADED,
            'adx_min': TREND_SETTINGS['adx_min'],
            'adx_max': TREND_SETTINGS['adx_max'],
            'high_potential_tfs': HIGH_POTENTIAL_TFS,
            'high_probability_tfs': HIGH_PROBABILITY_TFS
        })

    elif request.method == 'POST':
        data = request.json
        return jsonify({'message': 'Settings updated successfully'})

@app.route('/api/stats')
def get_scan_stats():
    """Get scanner statistics"""
    try:
        df = scanner.get_latest_results()

        if df.empty:
            return jsonify({
                'total_results': 0,
                'long_breakout_count': 0,
                'short_breakout_count': 0,
                'long_continuation_count': 0,
                'short_continuation_count': 0,
                'avg_potency_score': 0,
                'top_sectors': [],
                'last_scan_time': None
            })

        long_breakout_count = len(df[df['strategy_type'] == 'long_breakout'])
        short_breakout_count = len(df[df['strategy_type'] == 'short_breakout'])
        long_continuation_count = len(df[df['strategy_type'] == 'long_continuation'])
        short_continuation_count = len(df[df['strategy_type'] == 'short_continuation'])
        avg_potency = df['potency_score'].mean() if 'potency_score' in df.columns else 0

        sector_counts = df['sector'].value_counts().head(5).to_dict()
        last_scan = df['timestamp'].max() if 'timestamp' in df.columns else None

        return jsonify({
            'total_results': len(df),
            'long_breakout_count': long_breakout_count,
            'short_breakout_count': short_breakout_count,
            'long_continuation_count': long_continuation_count,
            'short_continuation_count': short_continuation_count,
            'avg_potency_score': round(avg_potency, 3) if avg_potency > 0 else 0,
            'top_sectors': sector_counts,
            'last_scan_time': last_scan.isoformat() if last_scan else None
        })

    except Exception as e:
        logger.error(f"Error getting scan stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'scanner_running': scanner.running
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def cleanup():
    """Cleanup resources on shutdown"""
    try:
        scanner.cleanup()
        logger.info("Scanner cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == '__main__':
    try:
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)

        start_background_scanner()

        import atexit
        atexit.register(cleanup)

        logger.info(f"Starting Flask server on {FLASK_HOST}:{FLASK_PORT}")
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=False,
            threaded=True
        )

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        cleanup()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        cleanup()
