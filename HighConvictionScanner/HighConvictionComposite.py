HTML_TEMPLATE ="""

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>High-Conviction Intraday Scanner</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0d1117; color: #c9d1d9; }
        .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); }
        .table-header th { background-color: #30363d; color: #f0f6fc; }
     
        .alert-success { color: #3fb950; font-weight: 600; }
        .alert-danger { color: #f85149; font-weight: 600; }
        .collapsible { cursor: pointer; padding: 12px; width: 100%; text-align: left; background-color: #21262d; border-bottom: 1px solid #30363d; border-radius: 8px; }
        /* Apply border-radius only to the button, not the content */
        .card .collapsible:first-child { border-top-left-radius: 8px;
        border-top-right-radius: 8px; border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
        .content { padding: 0 18px;
        max-height: 0; overflow: hidden; transition: max-height 0.2s ease-out; background-color: #161b22;
        }
        .active, .collapsible:hover { background-color: #30363d;
        }
        table { border-collapse: separate; border-spacing: 0; width: 100%;
        }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #30363d;
        }
    </style>
</head>
<body class="p-6">
    <div class="max-w-7xl mx-auto space-y-8">
        <header class="text-center">
            <h1 class="text-3xl font-bold text-white mb-2">High-Conviction Intraday Scanner</h1>
            <p class="text-sm text-gray-400">Last Scan: {{ RESULTS.last_scan_time }} (Scans every {{ SCAN_INTERVAL_SECONDS }} seconds)</p>
        </header>

        <div class="card p-6">
            <h2 class="text-xl font-semibold mb-3">Scanner Settings</h2>
   
            <form action="{{ url_for('update_settings') }}" method="post" class="flex items-center space-x-4">
                <label for="rvol" class="text-gray-400">Min RVOL (5m) for Entry Trigger:</label>
                <input type="number" step="0.1" id="rvol_override" name="rvol_override" value="{{ RESULTS.rvol_override }}" 
                       class="p-2 card bg-gray-800 border-gray-600 w-24 text-white" min="0.5" required>
       
                <label for="min_price" class="text-gray-400">Price Range (₹):</label>
                <input type="number" step="0.1" id="min_price" name="min_price" value="{{ RESULTS.base_filter_settings.min_price }}" 
                       class="p-2 card bg-gray-800 border-gray-600 w-24 text-white" min="0" required>
                <label for="max_price" class="text-gray-400">to</label>
               
                <input type="number" step="0.1" id="max_price" name="max_price" value="{{ RESULTS.base_filter_settings.max_price }}" 
                       class="p-2 card bg-gray-800 border-gray-600 w-24 text-white" min="0" required>
                <label for="min_volume_5m_cr" class="text-gray-400">Min 5m Value Traded (Crores):</label>
                <input type="number" step="0.1" id="min_volume_5m_cr" name="min_volume_5m_cr" value="{{ RESULTS.base_filter_settings.min_volume_5m_cr }}" 
               
                class="p-2 card bg-gray-800 border-gray-600 w-24 text-white" min="0" required>
                <label for="min_beta" class="text-gray-400">Min Beta:</label>
                <input type="number" step="0.1" id="min_beta" name="min_beta" value="{{ RESULTS.base_filter_settings.min_beta }}" 
                       class="p-2 card bg-gray-800 border-gray-600 w-24 text-white" min="0" required>
            
                <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-150">Update Filter</button>
            </form>
            <p class="text-sm text-gray-500 mt-3">Current Price Filter: ₹{{ RESULTS.base_filter_settings.min_price }} to ₹{{ RESULTS.base_filter_settings.max_price }} |
             Min Beta: {{ RESULTS.base_filter_settings.min_beta }}</p>
        </div>

        {% for category, stocks in RESULTS.items() %}
        {% if category in ['long_breakout', 'short_breakout', 'long_continuation', 'short_continuation'] %}
        <div class="card overflow-hidden">
            {% set title = category.replace('_', ' ').title() %}
            {% set is_long = 'long' in category %}
          
            {# ADDED data-listener and setting active for the first card by default #}
            <button type="button" class="collapsible text-lg font-semibold {% if is_long %} text-green-400 {% else %} text-red-400 {% endif %} {% if loop.index == 1 %} active {% endif %}" data-listener="false">
                {{ title }} ({{ stocks | length }} Stock{{ 's' if stocks | length != 1 else '' }})
            </button>
 
            
            <div class="content" {% if loop.index == 1 %} style="max-height: 1000px;"
            {% endif %}>
                {% if stocks %}
                <table class="min-w-full">
                    <thead class="table-header">
                        <tr>
                 
                            <th>Stock</th>
                            <th>Score</th>
                            <th>Entry (E)</th>
                            <th>SL ({{ WEIGHTS.ATR_SL_MULTIPLE }}x 
                            ATR)</th>
                            <th>Target (T) (1:2 R:R)</th>
                            <th>Pos Size (Shares)</th>
                            <th>Trade Value (₹)</th>
         
                            <th>Context</th>
                        </tr>
                    </thead>
                    <tbody>
                 
                        {% for stock in stocks %}

                        <tr>
                            <td class="font-bold"> 
                                <a href="https://in.tradingview.com/chart/?symbol={{ stock.ticker|replace('&', '_') 
                                }}" target="_blank" class="hover:text-blue-400">
                                    {{ stock.ticker }} ({{ 'Long' if is_long else 'Short' }})
                                </a>
                     
                            </td>
                            <td>{{ "%.2f"|format(stock.potency_score) }}</td>
                            <td>{{ "%.2f"|format(stock.entry_price) }}</td>
                            <td class="{% if is_long %} 
                            alert-danger {% else %} alert-success {% endif %}">{{ "%.2f"|format(stock.stop_loss) }}</td>
                            <td class="{% if is_long %} alert-success {% else %} alert-danger {% endif %}">{{ "%.2f"|format(stock.target_price) }}</td>
                            <td>{{ stock.position_size }}</td>
                   
                            <td>{{ "%.2f"|format(stock.total_trade_value) }}</td>
                            <td>
                                RVOL (5m): {{ "%.2f"|format(stock['rvol_5m']) }} |
                                ATR (5m): {{ "%.2f"|format(stock['ATR_5m']) }}
                            </td>
                        </tr>
                        {% endfor %}
                  
                        </tbody>
                </table>
                {% else %}
                <p class="p-4 text-gray-500">No stocks currently meet this high-conviction criteria.</p>
                {% endif %}
            </div>
        </div>
  
        {% endif %}
        {% endfor %}

        <footer class="text-center text-gray-600 text-sm pt-4">
            <p>Logic based on Multi-Timeframe Squeeze Breakout and Trend Continuation.</p>
            <p>Risk controlled at ₹{{ WEIGHTS.MAX_RISK_RS }} per trade, Max Capital ₹{{ WEIGHTS.MAX_CAPITAL_RS }}.</p>
        </footer>
    </div>

    <div id="historical-alerts-section" class="max-w-7xl mx-auto space-y-8 mt-10">
        <header class="text-center">
            <h2 class="text-3xl font-bold text-white mb-2">Historical Scan Alerts (MongoDB)</h2>
            <p class="text-sm text-gray-400">Review of the last 50 past trade setups, grouped by Ticker, sorted by Latest Alert Time.</p>
        </header>
        <div id="historical-alerts-controls" class="text-right">
            <button id="toggle-all-historical" 
                    class="bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-lg transition duration-150 text-sm">
                Expand All
            </button>
        </div>
        <div id="historical-alerts-content">
            <div class="card p-6 text-center text-gray-500">
                <p>Fetching historical data...</p>
            </div>
        </div>
    </div>
    <script>
    // Refactored to a function to be used for both static and dynamic content
    function initCollapsibles() {
        var coll = document.getElementsByClassName("collapsible");
        for (let i = 0; i < coll.length; i++) {
            // Check if listener is already attached to avoid duplicates
            if (coll[i].getAttribute('data-listener') === 'false' || !coll[i].getAttribute('data-listener')) {
                coll[i].addEventListener("click", function() {
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    if (content.style.maxHeight){
                        content.style.maxHeight = null;
                    } else {
                        // Use scrollHeight to expand to the full content size
                        content.style.maxHeight = content.scrollHeight + "px";
                    } 
                });
                coll[i].setAttribute('data-listener', 'true');
            }
        }
    }
    
    // Utility function to convert ISO string to a readable format
    function formatScanTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString('en-IN', {
            year: 'numeric', month: 'numeric', day: 'numeric',
            hour: '2-digit', minute: '2-digit', second: '2-digit',
            hour12: false
        });
    }
    
    // Function to toggle all historical alert collapsibles
    function toggleAllHistoricalAlerts() {
        const coll = document.getElementById('historical-alerts-content').getElementsByClassName("collapsible");
        const isAllExpanded = Array.from(coll).every(c => c.classList.contains('active'));
        const newState = isAllExpanded ? 'collapse' : 'expand';
        
        for (let i = 0; i < coll.length; i++) {
            const button = coll[i];
            const content = button.nextElementSibling;

            if (newState === 'expand') {
                button.classList.add("active");
                content.style.maxHeight = content.scrollHeight + "px";
            } else {
                button.classList.remove("active");
                content.style.maxHeight = null;
            }
        }
        
        // Update the button text
        const toggleButton = document.getElementById('toggle-all-historical');
        if (toggleButton) {
            toggleButton.textContent = isAllExpanded ?
            'Expand All' : 'Collapse All';
        }
    }

    // Function to fetch data from the Flask API and render the HTML
    async function fetchAndRenderHistoricalAlerts() {
        const contentDiv = document.getElementById('historical-alerts-content');
        const apiUrl = '/api/historical_alerts?limit=150'; // Fetch last 150 records
        const controlButton = document.getElementById('toggle-all-historical');

        try {
            const response = await fetch(apiUrl);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const scans = await response.json();
            let html = '';

            if (scans.length === 0) {
                html = `
                    <div class="card p-6 text-center text-gray-500">
                        <p>No historical scan data found in the database.
                        Run the scanner in the background to start logging alerts.</p>
                    </div>
                `;
            } else {
                // --- Grouping Logic: Aggregate alerts by Ticker ---
                const groupedAlerts = {};

                scans.forEach(scan => {
                    const scanTime = formatScanTime(scan.scan_time);
                    
                    scan.alerts.forEach(alert => {
                        const ticker = alert.ticker;

                        if (!groupedAlerts[ticker]) {
                            groupedAlerts[ticker] = {
                                ticker: ticker,
                                alerts: [] 
                            };
                        }

                        // Add the alert details, including the scan time
                        groupedAlerts[ticker].alerts.push({
                            scan_time: scanTime,
                            ...alert 
                        });
                    });
                });
                
                // --- Sorting by Latest Scan Time (Most Recent First) ---
                const sortedGroupedAlerts = Object.values(groupedAlerts)
                    .sort((a, b) => {
                        // Ensure alerts within the group are sorted by time (descending) 
                        // before using the first element for comparison.
                        a.alerts.sort((alertA, alertB) => new Date(alertB.scan_time) - new Date(alertA.scan_time));
                        b.alerts.sort((alertA, alertB) => new Date(alertB.scan_time) - new Date(alertA.scan_time));

                        const latestTimeA = new Date(a.alerts[0].scan_time); 
                        const latestTimeB = new Date(b.alerts[0].scan_time); 
                        
                        // Sort descending (b - a)
                        return latestTimeB - latestTimeA; 
                    });

                // --- HTML Generation (Grouped by Ticker) ---
                sortedGroupedAlerts.forEach((group) => {
                    const ticker = group.ticker;
                    const alertCount = group.alerts.length;
                    
                    // The alerts inside 'group.alerts' are already sorted by time (most recent first) 
                    // from the sorting step above.
                    const firstAlert = group.alerts[0];

                    // Start of the collapsible card for a single Ticker
                    html += `
                        <div class="card overflow-hidden">
                            <button type="button" class="collapsible text-lg font-semibold text-blue-400" data-listener="false">
                            
                                <a href="https://in.tradingview.com/chart/?symbol=${ticker.replace(/&/g, "_")}" target="_blank" class="hover:text-blue-400">
                                    ${ticker} (${alertCount} Alert${alertCount !== 1 ? 's' : ''})
                                </a>
                                <span class="text-sm text-gray-500 ml-4">Latest Setup: ${firstAlert.alert_type.replace('_', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</span>
                            </button>
                            
                            <div class="content">
                    `;
      
                    if (alertCount > 0) {
                        // Start of the alerts table
                        html += `
                            <table class="min-w-full">
                                <thead class="table-header">
                                    <tr>
                                        <th>Scan Time</th>
                                        <th>Type</th>
                                        <th>Score</th>
                                        <th>Entry (E)</th>
                                        <th>SL</th>
                                        <th>Target</th>
                                        <th>Context</th>
                                    </tr>
                                </thead>
                                <tbody>
                        `;
                        
                        // Table rows for each alert instance for this ticker
                        group.alerts.forEach((alert) => {
                            const isLong = alert.alert_type.includes('long');
                            const slClass = isLong ? 'alert-danger' : 'alert-success';
                            const targetClass = isLong ? 'alert-success' : 'alert-danger';

                            html += `
                                <tr>
                                    <td>${alert.scan_time}</td>
                                    <td class="font-bold ${isLong ? 'text-green-400' : 'text-red-400'}">
                                        ${alert.alert_type.replace('_', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                                    </td>
                                    <td>${alert.potency_score.toFixed(2)}</td>
                                    <td>${alert.entry_price.toFixed(2)}</td>
                                    <td class="${slClass}">${alert.stop_loss.toFixed(2)}</td>
                                    <td class="${targetClass}">${alert.target_price.toFixed(2)}</td>
                                    <td>
                                        Trend: ${alert.daily_trend} | Setup: ${alert.setup_type} | RVOL (5m): ${alert.rvol_5m.toFixed(2)}
                                    </td>
                                </tr>
                            `;
                        });
                        
                        // Close table body and table
                        html += `
                                </tbody>
                            </table>
                        `;
                    } else {
                        html += `
                            <p class="p-4 text-gray-500">Error: No alerts found for this ticker.</p>
                        `;
                    }
                    
                    // Close collapsible content div and card
                    html += `
                            </div>
                        </div>
                    `;
                });
            }

            // Insert the generated HTML into the content div
            contentDiv.innerHTML = html;
            // Re-initialize collapsibles for the newly added HTML
            initCollapsibles();
            
            // Attach event listener to the new button after content is rendered
            if (controlButton) {
                controlButton.addEventListener('click', toggleAllHistoricalAlerts);
                controlButton.textContent = 'Expand All';
                controlButton.style.display = 'inline-block';
            }


        } catch (error) {
            console.error('Error fetching historical data:', error);
            contentDiv.innerHTML = `
                <div class="card p-6 text-center text-red-500">
                    <p>Error loading historical alerts.
                    Check server console for MongoDB connection issues.</p>
                </div>
            `;
            // HIDE the button on error
            if (controlButton) {
                controlButton.style.display = 'none';
            }
        }
    }
    
    // Auto-refresh logic (fixed 'this' usage)
    function startAutoRefresh() {
        fetchAndRenderHistoricalAlerts(); // Initial fetch
        // Refresh every 30 seconds
        setInterval(() => fetchAndRenderHistoricalAlerts(), 30000);
        setInterval(function() { window.location.reload();     }, 30 * 1000);
    }
    
    // Call the fetch function and start the auto-refresh loop
    document.addEventListener('DOMContentLoaded', startAutoRefresh);
</script>


</body>
</html>


"""
import time
import os
import json
import math
from threading import Thread
import traceback
from flask import Flask, render_template_string, request, redirect, url_for
from tradingview_screener import Query, Column as col, And, Or 
from datetime import datetime
import pandas as pd
# --- 1. MongoDB Imports ---
from pymongo import MongoClient
# --------------------------

# --- Global Configuration and Data Store ---
app = Flask(__name__)
# The interval at which the background scanner runs
SCAN_INTERVAL_SECONDS = 60 
market = 'india'  # Market to scan

# --- 2. MongoDB Configuration ---
# Replace with your actual MongoDB connection string
MONGO_URI = "mongodb://localhost:27017/" 
MONGO_DB_NAME = "HighConvictionScans"
MONGO_COLLECTION_NAME = "IntradayAlerts"
# --------------------------------

# Dictionary to store the latest scan results and settings
RESULTS = {
    'long_breakout': [],
    'long_continuation': [],
    'short_breakout': [],
    'short_continuation': [],
    'last_scan_time': 'Never',
    'rvol_override': 3.0,
    'base_filter_settings': {
        'min_price': 10,
        'max_price': 100000,
        'min_volume_5m_cr': 3.0, # Value Traded in Crores
        'min_beta': 1.2,
    }
}

# --- 3. MongoDB Connection Function ---
def connect_to_mongo():
    """Connects to MongoDB and returns the collection object."""
    try:
        # Connect to the MongoDB client
        client = MongoClient(MONGO_URI)
        # Access the database
        db = client[MONGO_DB_NAME]
        # Access the collection
        collection = db[MONGO_COLLECTION_NAME]
        print(f"Successfully connected to MongoDB: {MONGO_DB_NAME}/{MONGO_COLLECTION_NAME}")
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

MONGO_COLLECTION = connect_to_mongo()
# --------------------------------------

# --- 1. Define Common Columns for Selection (Field Name Corrections Applied) ---
# FIX 1: Removed unused 'BB.upper|5' and 'BB.lower|5'
SELECT_COLS = (\
    'name', 'logoid', 'close', 'close|5', 'Value.Traded', \
    'relative_volume_10d_calc|5', \
    # Indicators for Potency/Context\
    'ATR', 'ATR|5', 'ADX|60', \
    'BB.upper', 'BB.lower', \
    'BB.upper|15', 'BB.upper|30', 'BB.upper|60', \
    'BB.lower|15', 'BB.lower|30', 'BB.lower|60', \
    'KltChnl.upper|15', 'KltChnl.upper|30', 'KltChnl.upper|60', \
    'KltChnl.lower|15', 'KltChnl.lower|30', 'KltChnl.lower|60', \
    'EMA20', 'EMA200', 'EMA5|5', \
    'beta_1_year' \
)

# --- 2. Potency Score Weights and Risk Management Constants ---
# These constants define the importance of factors and the trade management rules.
WEIGHTS = {
    'W_RVOL': 25,
    'W_TF': 20,
    'W_SETUP': 10,
    'MAX_RISK_RS': 1000.0, # Maximum risk per trade
    'MAX_CAPITAL_RS': 100000.0, # Max total capital used for all trades
    'RR_RATIO': 2.0, # Risk-Reward Ratio (Target distance = RR_RATIO * SL distance)
    'ATR_SL_MULTIPLE': 2.0, # Stop-loss distance is 2x ATR(5m)
    'SQZ_WIDTH_FACTOR': 0.75 # New logical correction: BB Width must be < 0.75 * KltChnl Width
}

# --- 3. Scanner Logic and Filtering Functions ---

    # def stage1_filters(min_price, max_price, min_volume_5m_cr, min_beta, rvol_min):
    #     """
    #     Stage 1: Implements mandatory Liquidity and Context Filters (Universal Pre-Scan).
    #     Filters on Price, Beta, Value Traded, and minimum 5m RVOL.
    #     """
    #     # Convert Value Traded in Crores to actual currency units (1 Crore = 10,000,000)
    #     min_value_traded = min_volume_5m_cr * 10000000

    #     return 

def calculate_potency(stock):
    """
    Stage 2: Calculates the Potency Score based on RVOL and Setup quality.
    Applies the new, precise Squeeze condition post-data retrieval.
    """
    potency = 0
    tf_order_score = 0
    setup_type_score = 0
    setup_type = None
    
    # 1. RVOL Factor Score (Max 25 points)
    # Capped at 3.0 to prevent outlier RVOLs from skewing the score
    rvol_factor = min(stock['relative_volume_10d_calc|5'], 3.0)
    potency += WEIGHTS['W_RVOL'] * rvol_factor 
    
    # 2. Setup Identification (Precise Squeeze and Pullback)

    # Check for Squeeze condition on MTFs (60m > 30m > 15m)
    # New Logic: BB Width < 0.75 * KltChnl Width
    squeeze_found = False
    
    for tf in [60, 30, 15]:
        bb_upper = stock[f'BB.upper|{tf}']
        bb_lower = stock[f'BB.lower|{tf}']
        klt_upper = stock[f'KltChnl.upper|{tf}']
        klt_lower = stock[f'KltChnl.lower|{tf}']
        
        # Calculate widths
        bb_width = bb_upper - bb_lower
        klt_width = klt_upper - klt_lower
        
        # Check for both price integrity (non-zero width) and the squeeze condition
        if klt_width > 0 and (bb_width < WEIGHTS['SQZ_WIDTH_FACTOR'] * klt_width):
            squeeze_found = True
            
            # Prioritize the highest TF found
            if tf == 60 and tf_order_score < 3:
                tf_order_score = 3
            elif tf == 30 and tf_order_score < 2:
                tf_order_score = 2
            elif tf == 15 and tf_order_score < 1:
                tf_order_score = 1
            
            # If 60m squeeze is found, we stop looking at lower TFs for scoring
            if tf_order_score == 3:
                break 

    stock['is_squeeze_ready'] = squeeze_found
    
    # Trend Pullback: Strong directional trend (ADX > 25 on 60m)
    stock['is_pullback_ready'] = (stock['ADX|60'] >= 25.0)
    
    # 3. Apply Potency Score based on setups

    if stock['is_pullback_ready']:
        setup_type = 'Pullback'
        setup_type_score = 8 
        # Pullback is always scored 3 for TF order (requires 60m ADX > 25)
        if tf_order_score < 3:
             tf_order_score = 3 

    if stock['is_squeeze_ready']:
        # If squeeze is found, it is generally higher conviction than a pullback
        # and overrides the setup_type/score if a squeeze was found on any TF.
        setup_type = 'Breakout'
        setup_type_score = 10 
        # tf_order_score for squeeze is already set above (3, 2, or 1)

    # Final Potency Calculation
    if setup_type is not None:
        potency += WEIGHTS['W_TF'] * tf_order_score
        potency += setup_type_score
    
    return round(potency, 2), setup_type, tf_order_score

def calculate_trade_management(stock, entry_price):
    """
    Calculates SL, Target, and Position Size based on 1:2 R:R and Max Risk (₹1000).
    """
    atr_5m = stock.get('ATR|5', 0.0)
    
    # Risk per Share (SL Distance): 2x ATR(5m)
    risk_per_share = WEIGHTS['ATR_SL_MULTIPLE'] * atr_5m
    
    # Target Distance: Risk * R:R Ratio (e.g., 2x ATR * 2.0 = 4x ATR)
    target_distance = risk_per_share * WEIGHTS['RR_RATIO']
    
    # Safety check: Prevent division by zero if ATR is non-existent or zero
    if risk_per_share == 0 or entry_price == 0:
        return 0, 0, 0, 0.0, 0.0
    
    # 1. Calculate Position Size based on Max Risk (₹1000)
    max_shares_by_risk = math.floor(WEIGHTS['MAX_RISK_RS'] / risk_per_share)
    
    # 2. Check Capital Constraint (Max Capital ₹100,000)
    max_shares_by_capital = math.floor(WEIGHTS['MAX_CAPITAL_RS'] / entry_price)
    
    # Final Position Size is the minimum of the two constraints
    position_size = int(min(max_shares_by_risk, max_shares_by_capital))
    
    # Actual Trade Value and Actual Risk
    total_trade_value = position_size * entry_price
    actual_risk = position_size * risk_per_share
    
    return risk_per_share, target_distance, position_size, total_trade_value, actual_risk

def run_scan():
    """
    Executes the multi-stage scanner logic and updates the global RESULTS dictionary.
    """
    global RESULTS

    rvol_min = RESULTS['rvol_override']
    settings = RESULTS['base_filter_settings']


    stage1_filters =And(
        # A. Liquidity & Volatility (Price, Value Traded, Beta)
        
        col('beta_1_year') >= settings['min_beta'],
        col('is_primary') == True,
        col('typespecs').has('common'),
        col('type') == 'stock', 
        col('close').between(settings['min_price'], settings['max_price']),
        col('active_symbol') == True,
        col('exchange')== 'NSE',
            
        col('Value.Traded|5') >= settings['min_volume_5m_cr'] * 10000000,
        # B. RVOL Filter
         
        Or(
            #EMA10 above EMA20 to filter out extremely volatile stocks and CLOSE price above both EMAs for trend direction Or INVERSE EMA10 below EMA20 and CLOSE below both EMAs for downtrend
            And(
                col('EMA10|5') < col('EMA20|5'),
                col('close|5') < col('EMA10|5'),
                col('close|5') < col('close[1]|5'),
                Or(
                    col('relative_volume_intraday|5') > rvol_min,
                    col('volume|5').above_pct(col('average_volume_10d_calc|5'), rvol_min )
                ) 
            ),
            And(
                col('close|5') > col('EMA10|5'),
                col('EMA10|5') > col('EMA20|5'),
                col('close|5') > col('close[1]|5'),
                Or(
                    col('relative_volume_intraday|5') > rvol_min,
                    col('volume|5').above_pct(col('average_volume_10d_calc|5'), rvol_min )
                )
            ),
        )       
    )
    
 
    try:
        # --- Stage 1: Build the Universal Pre-Scan Query ---
         
        
        # --- Stage 2: Add Setup Identification Conditions ---
        # FIX 1: Updated Query Squeeze condition to be two-sided (BB must be inside Keltner)
        squeeze_condition = Or(
            And(col('BB.upper|60') < col('KltChnl.upper|60'), col('BB.lower|60') > col('KltChnl.lower|60')),
            And(col('BB.upper|30') < col('KltChnl.upper|30'), col('BB.lower|30') > col('KltChnl.lower|30')),
            And(col('BB.upper|15') < col('KltChnl.upper|15'), col('BB.lower|15') > col('KltChnl.lower|15')),
        )
        
        # Trend Pullback: Strong directional trend (ADX > 25 on 60m)
        pullback_condition = col('ADX|60') >= 25.0
        
        # Combine Setup Conditions: At least one setup must be ready
        setup_condition = And(stage1_filters, Or(squeeze_condition, pullback_condition))

        # Finalize Query
        query = Query().select(*SELECT_COLS).where2(setup_condition)
       
        query = query.set_markets(market) 
        # --- Execute the Query ---
        total, df = query.get_scanner_data()
        
        if df.empty:
            print("Scan complete. No setups found.")
            RESULTS['long_breakout'] = []
            RESULTS['short_breakout'] = []
            RESULTS['long_continuation'] = []
            RESULTS['short_continuation'] = []
            RESULTS['last_scan_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return

        print(f"Scan complete. {len(df)} stocks passed Stage 1 filters (Liquidity, RVOL) and Pullback filter.")
        
        # Prepare final result lists
        final_long_breakouts = []
        final_short_breakouts = []
        final_long_continuations = []
        final_short_continuations = []

        # --- Stage 3: Post-Processing, Potency Scoring, and Entry Trigger Check ---
        stocks_to_keep = []
        for _, stock in df.iterrows():
            
            # Calculate Potency Score (which now also sets is_squeeze_ready and is_pullback_ready)
            potency, setup_type, tf_order = calculate_potency(stock)
            
            # Only proceed if a valid setup was found (Squeeze OR Pullback)
            if setup_type is None:
                continue

            # Higher TF Trend Context Check (Daily TF)
            is_bullish_trend = (stock['close'] > stock['EMA20']) and (stock['EMA20'] > stock['EMA200'])
            is_bearish_trend = (stock['close'] < stock['EMA20']) and (stock['EMA20'] < stock['EMA200'])

            stock_data = {
                'ticker': stock['ticker'],
                'close_5m': stock['close|5'],
                'potency_score': potency,
                'setup_type': setup_type,
                'tf_order_score': tf_order,
                'rvol_5m': stock['relative_volume_10d_calc|5'],
                'ATR_5m': stock.get('ATR|5', 0.0), # Use .get for safer access
                'daily_trend': 'Bullish' if is_bullish_trend else ('Bearish' if is_bearish_trend else 'Neutral'),
                'raw_data': stock.to_dict() 
            }
            
            trade_ready = False
            trade_type = None

            # 1. Long Breakout/Continuation Check (Requires Bullish Trend)
            if is_bullish_trend and (stock['is_squeeze_ready'] or stock['is_pullback_ready']):
                
                # Long Breakout Trigger (Price > Daily BB Upper)
                if stock['close|5'] > stock['BB.upper']:
                    trade_ready = True
                    trade_type = 'long_breakout'
                
                # Long Continuation Trigger (Price > 5m EMA5)
                elif stock['close|5'] > stock['EMA5|5']:
                    trade_ready = True
                    trade_type = 'long_continuation'

            # 2. Short Breakout/Continuation Check (Requires Bearish Trend)
            elif is_bearish_trend and (stock['is_squeeze_ready'] or stock['is_pullback_ready']):

                # Short Breakout Trigger (Price < Daily BB Lower)
                if stock['close|5'] < stock['BB.lower']:
                    trade_ready = True
                    trade_type = 'short_breakout'

                # Short Continuation Trigger (Price < 5m EMA5)
                elif stock['close|5'] < stock['EMA5|5']:
                    trade_ready = True
                    trade_type = 'short_continuation'

            
            # --- Trade Management Calculation for Ready Trades ---
            if trade_ready:
                entry_price = stock['close|5']
                risk_per_share, target_distance, pos_size, total_trade_value, actual_risk = calculate_trade_management(stock, entry_price)

                if pos_size > 0:
                    
                    if trade_type in ['long_breakout', 'long_continuation']:
                        sl_price = entry_price - risk_per_share
                        target_price = entry_price + target_distance
                    else: # Short setups
                        sl_price = entry_price + risk_per_share
                        target_price = entry_price - target_distance

                    # Keys used here must match the HTML template (stop_loss, target_price, entry_price)
                    stock_data.update({
                        'entry_price': entry_price,
                        'stop_loss': round(sl_price, 2),
                        'target_price': round(target_price, 2),
                        'position_size': pos_size,
                        'risk_per_share': round(risk_per_share, 2),
                        'target_distance': round(target_distance, 2),
                        'actual_risk': round(actual_risk, 2),
                        'total_trade_value': round(total_trade_value, 2),
                        'actual_rr': round(target_distance / risk_per_share, 2) if risk_per_share > 0 else 0.0,
                    })
                    
                    # Append to the correct list
                    if trade_type == 'long_breakout':
                        final_long_breakouts.append(stock_data)
                    elif trade_type == 'long_continuation':
                        final_long_continuations.append(stock_data)
                    elif trade_type == 'short_breakout':
                        final_short_breakouts.append(stock_data)
                    elif trade_type == 'short_continuation':
                        final_short_continuations.append(stock_data)

        # Sort results by Potency Score (descending)
        final_long_breakouts.sort(key=lambda x: x['potency_score'], reverse=True)
        final_short_breakouts.sort(key=lambda x: x['potency_score'], reverse=True)
        final_long_continuations.sort(key=lambda x: x['potency_score'], reverse=True)
        final_short_continuations.sort(key=lambda x: x['potency_score'], reverse=True)


        # Update global RESULTS
        RESULTS['long_breakout'] = final_long_breakouts
        RESULTS['long_continuation'] = final_long_continuations
        RESULTS['short_breakout'] = final_short_breakouts
        RESULTS['short_continuation'] = final_short_continuations
        RESULTS['last_scan_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Scan complete. Found {len(final_long_breakouts) + len(final_short_breakouts) + len(final_long_continuations) + len(final_short_continuations)} potential trades.")

    except Exception as e:
        print(f"An error occurred during scanning: {e}")
        traceback.print_exc()


# --- 4. MongoDB Storage Function ---
def save_scan_results_to_db(results_dict):
    """
    Stores the final filtered and scored scan results into the MongoDB collection.
    It prepares a single document containing all results from the current scan cycle.
    """
    if MONGO_COLLECTION is None:
        print("Skipping database save: MongoDB connection failed.")
        return

    # Create a timestamped record for the entire scan run
    timestamp = datetime.now()
    
    # Structure the document for MongoDB
    db_record = {
        'scan_time': timestamp,
        'scan_interval_seconds': SCAN_INTERVAL_SECONDS,
        'base_filters': results_dict['base_filter_settings'],
        'rvol_override': results_dict['rvol_override'],
        # Combine all setups into a single list for simpler storage
        'alerts': [],
    }

    alert_types = [
        'long_breakout', 'long_continuation', 
        'short_breakout', 'short_continuation'
    ]
    
    for alert_type in alert_types:
        for alert in results_dict[alert_type]:
            # Add the specific type to the alert document
            alert_document = alert.copy()
            alert_document['alert_type'] = alert_type
            db_record['alerts'].append(alert_document)
            
    try:
        if db_record['alerts']:
            # Insert the entire scan result document
            insert_result = MONGO_COLLECTION.insert_one(db_record)
            print(f"Stored {len(db_record['alerts'])} alerts from {db_record['scan_time']} (ID: {insert_result.inserted_id})")
        else:
            # If no alerts, still log the scan time
            # db_record['alerts_count'] = 0
            # MONGO_COLLECTION.insert_one(db_record)
            print(f"Scan completed at {timestamp}, 0 alerts found. Logged scan time. NO update to db")
            
    except Exception as e:
        print(f"Error saving data to MongoDB: {e}")

# --- 5. Background Thread Function ---
def background_scanner():
    """
    The main thread function that continuously runs the scanner logic.
    """
    print("Starting background scanner thread...")
    print(f"Background scanner started. Running every {SCAN_INTERVAL_SECONDS} seconds.")
    while True:
        try:
            # Check if market is open (simplified check, you may need a more robust one)
            now = datetime.now()
            # Basic check for market hours (e.g., 9:15 to 15:30 IST Mon-Fri)
            # Keeping the user's wide time range for now: 9 <= now.hour < 24
            if 9 <= now.hour < 24 and now.weekday() < 5: 
                print(f"Running scan at {now.strftime('%H:%M:%S')}...")
                run_scan()
                # --- Crucial Addition: Save results to DB after each scan ---
                save_scan_results_to_db(RESULTS)
                # -----------------------------------------------------------
                time.sleep(SCAN_INTERVAL_SECONDS)
            else:
                # Sleep longer if market is closed
                print(f"Market closed. Sleeping for 5 minutes. Last scan: {RESULTS['last_scan_time']}")
                time.sleep(300) # 5 minutes
        except Exception as e:
            print(f"Error in background scanner loop: {e}")
            traceback.print_exc()
            time.sleep(60) # Sleep for a minute after error

# --- Flask Web Server and HTML Template ---

@app.route('/')
def index():
    """Renders the main dashboard page."""
    # Ensure all trade lists are available in the RESULTS dictionary
    data = RESULTS
    
    # Calculate total number of trades found
    total_trades = sum(len(data.get(k, [])) for k in ['long_breakout', 'long_continuation', 'short_breakout', 'short_continuation'])
    
    # Format current time for display
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template_string(HTML_TEMPLATE, 
        DATA=data,
        WEIGHTS=WEIGHTS,
        TOTAL_TRADES=total_trades,
        CURRENT_TIME=current_time,
        SCAN_INTERVAL_SECONDS=SCAN_INTERVAL_SECONDS,
        RESULTS=RESULTS
    )

@app.route('/settings', methods=['POST'])
def update_settings():
    """Updates global scanner settings from the web form."""
    try:
        #print all form data for debugging
        print("Received settings update:", request.form)
        # Update RVOL Override
        rvol_override = float(request.form.get('rvol_override'))
        RESULTS['rvol_override'] = rvol_override
        
        #add below fields to the form and update logic
        # Update Base Filter Settings if provided
        if request.form.get('min_price'):
            RESULTS['base_filter_settings']['min_price'] = float(request.form.get('min_price'))
        if request.form.get('max_price'):
            RESULTS['base_filter_settings']['max_price'] = float(request.form.get('max_price'))
        if request.form.get('min_volume_5m_cr'):
            RESULTS['base_filter_settings']['min_volume_5m_cr'] = float(request.form.get('min_volume_5m_cr'))
        if request.form.get('min_beta'):
            RESULTS['base_filter_settings']['min_beta'] = float(request.form.get('min_beta'))

        run_scan()  # Re-run scan with new settings immediately


    except Exception as e:
        print(f"Error updating settings: {e}")
        traceback.print_exc()
    
    return redirect(url_for('index'))

from flask import jsonify # ADD THIS IMPORT

# ... (other code)

def serialize_scan_record(record):
    """Converts a MongoDB BSON record to a JSON-serializable dictionary."""
    # Convert MongoDB's ObjectId to string
    record['_id'] = str(record['_id'])
    # Convert datetime objects to ISO format string
    record['scan_time'] = record['scan_time'].isoformat()
    return record

@app.route('/api/historical_alerts', methods=['GET'])
def get_historical_alerts():
    """Fetches the last N historical scan records from MongoDB."""
    if MONGO_COLLECTION is None:
        return jsonify({'error': 'MongoDB connection failed.'}), 500
    
    # Get the 'limit' query parameter, default to 10 records
    try:
        limit = int(request.args.get('limit', 10))
    except ValueError:
        limit = 10

    try:
        # Fetch the last 'limit' records, sorted by scan_time descending
        historical_scans = list(
            MONGO_COLLECTION.find({
                                    'alerts': {
                                        '$ne': []
                                    }
                                })
            .sort('scan_time', -1)
            .limit(limit)
        )
        
        # Serialize the records before sending as JSON
        serialized_scans = [serialize_scan_record(record) for record in historical_scans]
        
        return jsonify(serialized_scans)
    
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return jsonify({'error': 'Failed to retrieve data from database.'}), 500

# ... (rest of your existing Flask routes and main execution block)



# --- 7. Run the Application ---
if __name__ == '__main__':
    # Start the background scanning thread
    scanner_thread = Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    
    # Start the Flask web server
    # Set host='0.0.0.0' to make it accessible externally
    app.run(host='0.0.0.0', port=9090, debug=False)