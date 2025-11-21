# COMPOSITE_SCAN
USE https://github.com/MaheshUmale/scan as base


CREATE 




To effectively manage the trade-offs between squeeze/breakout and trending stocks, and between smaller and higher timeframes (TF), you can structure your scanning process into two distinct, yet complementary, stages: The Watchlist Builder (Higher TF) and The Entry Timer (Smaller TF).

1. The Watchlist Builder: Focus on Potential (Higher TFs - 15m/30m/60m)
This stage addresses your concern about missing bigger potential. You want to identify stocks with the potential for a significant move before they become obvious trends.

Primary Scanner: Squeeze (e.g., Keltner Channels/Bollinger Bands) and High-Volume Consolidation/DC Breakout.

Goal: To find stocks that are currently basing, consolidating, or in a tight trading range on a Higher Timeframe (15-minute or 30-minute). These are your "coiled springs" with the biggest potential for explosive moves.

Action: Add these stocks to a "Pre-Breakout Watchlist." You are identifying the potential setup.

Secondary Scanner: Already Trending Stocks (High Relative Strength/ADX).

Goal: To find stocks that are already moving strongly on the Daily or 60-minute TF. These stocks offer high-probability continuation trades after a pullback.

Action: Add these to a "Momentum Watchlist."
Scanner Type,Timeframe,Focus,Benefit
Squeeze/Consolidation,15m / 30m,Bigger Potential,Early entry on explosive moves.
Trending/Momentum,60m / Daily,High Probability,Continuation trades with established direction.

The Entry Timer: Pinpoint Timing (Smaller TFs - 1m/3m/5m)
This stage addresses your concern about pinpointing entries without missing the initial move. This is where you use the smaller TFs for execution only.

Once you have your watchlists, you do not trade directly off the higher TF alert. Instead, you wait for confirmation on a smaller TF.

For the "Pre-Breakout Watchlist" (Squeeze/Consolidation):

Wait for the stock to break the consolidation level (DC Breakout) on the Higher TF (e.g., the 15m candle closes above the range).

Immediately switch to the Smallest Timeframe (1-minute or 3-minute).

Pinpoint Entry: Enter only when the smaller TF confirms the direction—look for a high-volume candle, a flag break, or a moving average cross in the direction of the breakout. This avoids false higher TF breaks.

For the "Momentum Watchlist" (Trending Stocks):

Wait for the stock to pull back to a key support level on the Higher TF (e.g., 9 or 20 period EMA on the 15m chart).

Switch to the Smallest Timeframe (3-minute or 5-minute).

Pinpoint Entry: Enter when the smaller TF shows a reversal candlestick pattern (e.g., hammer, engulfing) or a moving average flip at that higher TF support level. This ensures you're entering at a high-value, lower-risk zone during an established trend.

Timeframe Use,Action,Resolution
Higher TF (15m+),Identify the Setup (Squeeze or Pullback),Prevents small moves hitting big TF resistance.
Smaller TF (1m/3m/5m),Pinpoint the Entry (Confirmation/Reversal),Prevents missed opportunity and filters false breaks.

Intraday Scanner Indicators
1. 🎯 Squeeze and Breakout Scanners (Finding Potential)
The goal here is to identify periods of low volatility (the squeeze) on a higher timeframe, which often precedes an explosive move (the breakout).

Indicator,Timeframe,Scanner Criteria,Purpose
TTM Squeeze (or Bollinger Bands/Keltner Channel Combo),15m or 30m,"Look for the ""squeeze on"" signal (red dots in TTM, or Bollinger Bands inside Keltner Channels).","Identify Stocks Coiling: This is your primary ""Pre-Breakout Watchlist"" builder, spotting stocks before the major move happens."
High Volume Consolidation,15m,"Current volume below 20-period Average True Range (ATR), but with a Doji or Inside Bar pattern at a key price level (e.g., previous day's high/low).","Identify Stocks Resting: Finds stocks building energy near a breakout level, ensuring they have liquidity but are currently tightening."
Donchian Channel (DC) Breakout,30m,Current price has crossed either the 20-period Highest High or Lowest Low by a small margin.,Alert the Firing: Acts as the alert that the consolidation range is beginning to break. You then use the smaller TF to confirm the follow-through.

2. 🚀 Trending and Momentum Scanners (Finding Continuation)
The goal here is to identify stocks already moving strongly on the highest timeframes, which you will then wait for a pullback in to enter.
Indicator,Timeframe,Scanner Criteria,Purpose
Average Directional Index (ADX),60m or Daily,ADX > 25 (ideally moving up toward 40).,"Confirm Strong Trend: This filters for stocks with a strong, established trend, distinguishing true momentum from choppy price action."
Relative Strength (RS),Daily,Stock's 20-day Relative Strength Index (RSI) > 60 (for uptrends) or < 40 (for downtrends).,Filter for Strength: Ensures the stock is currently outperforming or underperforming the general market or its sector.
Moving Average (MA) Stack,60m,5-period EMA > 20-period EMA > 50-period SMA.,"Trend Alignment: This visually confirms a strong, clean trend where shorter-term momentum is aligned with intermediate-term direction."

Multi-Timeframe Execution Blueprint
Here's how you use these indicators to resolve your conflict using the multi-timeframe approach:

Scenario A: Trading a Squeeze/Breakout (Higher Potential)
Scanner (15m/30m): TTM Squeeze signals a squeeze is ON (red dots).

Higher TF Setup (15m): Wait for the price to break out of the range (e.g., Donchian Channel break, or the first green TTM dot/Bollinger Band expansion).

Lower TF Entry (3m/5m): Switch to the 3-minute chart. Do not enter immediately. Wait for one of the following confirmations:

A high-volume candle break in the direction of the higher TF breakout.

A pullback to the 9-period EMA or the top of the breakout range, followed by a bullish engulfing/hammer candlestick.

This pinpoints your entry, avoiding a false breakout on the 15m chart.

Scenario B: Trading a Trend Continuation (Higher Probability)
Scanner (60m/Daily): ADX > 25 and MA Stack is aligned (strong uptrend).

Higher TF Setup (15m): Wait for the price to pull back to a key support MA (e.g., 20-period EMA on the 15m chart).

Lower TF Entry (3m/5m): Switch to the 3-minute chart. Do not enter immediately. Wait for a clear reversal signal at that 15m support level:

RSI on the 3m chart drops to near 30 and then crosses back up above 50.

A reversal candlestick pattern (e.g., bullish engulfing) forms with a volume spike at the 15m support level.

This gives you the best-risk-reward entry in an established, confirmed trend.

By dividing the scanning process into finding setups on the higher TFs and finding entries on the smaller TFs, you effectively capture the potential of the squeeze and the reliability of the trend without the timing pitfalls of a single timeframe.



these are 2 scanners (appSQZ.py and scan.py)
Ref Links:https://shner-elmo.github.io/TradingView-Screener/fields/stocks.html
https://shner-elmo.github.io/TradingView-Screener/3.0.0/tradingview_screener.html
https://github.com/shner-elmo/TradingView-Screener

Generate CODE for Multi-Timeframe, Multi-Strategy Filter system based on above.

# Multi-Strategy Trading Scanner System

This guide outlines the architecture, logic, and implementation steps for an enhanced stock scanning system that combines high-potential squeeze detection with high-probability trend-following pullback identification.

The goal is to resolve the conflict between "early-stage, big-move potential" (Squeezes) and "already-trending, reliable continuation" (Pullbacks) by running both strategies simultaneously and presenting them in a unified, segmented dashboard.

## 1. Context and Problem Statement

| Old Scanner Type | Goal/Issue | Resolution |
| :--- | :--- | :--- |
| **Squeeze Scanner** (`appSQZ.py`) | Find stocks with high volatility compression (potential for large moves). *Issue: Many don't fire or fail quickly.* | **High Potential Setups:** Run on lower timeframes (15m, 30m) for early alerts. |
| **General Scanner** (`scan.py`) | Basic trend/volume filtering. *Issue: Misses stocks already strongly trending or at a continuation point.* | **High Probability Setups:** Implement a dedicated **Trending/Pullback** scan on higher timeframes (60m, Daily). |

## 2. System Architecture (3 Components)

| Component | Role | Files Involved | Key Change |
| :--- | :--- | :--- | :--- |
| **Logic Layer (Python)** | Executes external market queries, runs dual-strategy filtering, scores, and saves results to MongoDB. | \`scan.py\` | **Introduces \`get_trending_pullback_query\`**. Tags results by \`Strategy_Type\`. |
| **Server Layer (Python)** | Flask server for API endpoints (\`/data\` for results, \`/settings\`). Handles background scheduling. | \`appSQZ.py\` | Minimal changes; focuses on serving the new data structure. |
| **UI/Dashboard (HTML/JS)** | Fetches combined data, separates it into two distinct display panels, and provides visual cues. | \`index.html\` | **Critical:** Displays "Squeeze Heatmap" and "Pullback Table" separately. |

## 3. Step-by-Step Implementation Guide

Follow these steps to upgrade your system.

### Step 1: Update the Scanning Logic (\`scan.py\`)

The most critical change is in \`scan.py\`. We must introduce the **Trending/Pullback** filter to capture continuation trades.

* **New Constants:** Define timeframes for each strategy type.
    * **High Potential (Squeeze):** \`HIGH_POTENTIAL_TFS = ['|15', '|30', '|60']\`
    * **High Probability (Trend):** \`HIGH_PROBABILITY_TFS = ['|60', '|240', '']\` (60m, 4h, Daily)
* **New Query Function:** Implement the \`get_trending_pullback_query\`. This query identifies stocks where:
    1.  The trend is strong (\`ADX > 25\`).
    2.  The price is above the 50-period SMA (trend direction confirmed).
    3.  The price is currently near the 20-period EMA (healthy pullback zone).
* **Consolidation:** The main scanning loop must run **both** query functions, combine the results into a single DataFrame, and add a **\`Strategy_Type\`** column (\`High_Potential\` or \`High_Probability\`).

### Step 2: Update the Server Layer (\`appSQZ.py\`)

Ensure your Flask server continues to run the scanner thread and serve the data.

* **Data Handling:** The \`/data\` endpoint must now return the combined DataFrame, including the new \`Strategy_Type\` column. The Python server logic already uses the \`scan.py\` functions to fetch and save data to MongoDB, so minimal changes are expected here, provided \`scan.py\` returns the correctly structured DataFrame.

### Step 3: Implement the Unified UI (\`index.html\`)

The dashboard must now handle the dual-strategy data and display it clearly.

* **Data Fetching:** The JavaScript fetches the single, combined dataset.
* **Client-Side Filtering:** The JavaScript logic filters the data based on the new \`Strategy_Type\` column.
    * **Squeeze Data:** \`data.filter(d => d.Strategy_Type === 'High_Potential')\`
    * **Pullback Data:** \`data.filter(d => d.Strategy_Type === 'High_Probability')\`
* **Display Segregation:**
    * **Top Panel:** Display the **Squeeze/Breakout** stocks using the existing **Heatmap** view (showing multiple timeframes).
    * **Bottom Panel:** Display the **Trending/Pullback** stocks in a clean, high-priority **Table** view. This table should explicitly show the confirmation timeframe (e.g., 60m or Daily) and the relevant MA value.

## 4. Trading Strategy Summary (How to Use the New System)

This combined view dictates two different, high-conviction trading approaches:

| Strategy | Setup Timeframe (HTF) | Entry Timeframe (LTF) | Action Required |
| :--- | :--- | :--- | :--- |
| **High Potential (Squeeze)** | 15m, 30m, 60m | 3m, 5m | **Watchlist:** Set price alerts near the breakout levels. Only enter if LTF confirms the momentum shift (e.g., strong volume spike, reversal candle). |
| **High Probability (Pullback)** | 60m, 4h, Daily | 3m, 5m | **High Priority:** These stocks are already at a strong support/entry zone. Monitor the LTF for a tight, high-volume rejection candle at the key MA (e.g., EMA 20). |

This new system ensures you capture early-stage potential *and* reliable continuation trades, providing a complete view of the market's high-quality setups.

additional details are given in "Multi-Strategy Scanner System Readme.md" file
