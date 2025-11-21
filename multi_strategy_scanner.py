"""
Multi-Timeframe, Multi-Strategy Scanner System
Combines High-Potential Squeeze Detection with High-Probability Trend Pullback Identification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tradingview_screener import Query, Column, And, Or, col
import pymongo
from pymongo import MongoClient
import time
import threading

from config import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Timeframe mapping for TradingView
TIMEFRAME_MAP = {
    '1m': '|1',
    '5m': '|5',
    '15m': '|15',
    '30m': '|30',
    '60m': '|60',
    '1h': '|60',
    '2h': '|120',
    '4h': '|240',
    '1d': '',
    '1W': '|1W',
    '1M': '|1M',
}

import math


@dataclass
class ScanResult:
    """Data class for scan results"""
    ticker: str
    strategy_type: str  # 'High_Potential' or 'High_Probability'
    timeframe: str
    price: float
    volume: float
    market_cap: float
    sector: str
    exchange: str
    squeeze_status: Optional[str] = None
    trend_strength: Optional[float] = None
    pullback_depth: Optional[float] = None
    adx: Optional[float] = None
    rsi: Optional[float] = None
    sma50: Optional[float] = None
    ema20: Optional[float] = None
    bb_position: Optional[str] = None
    volume_ratio: Optional[float] = None
    potency_score: Optional[float] = None
    timestamp: Optional[datetime] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    position_size: Optional[int] = None
    total_trade_value: Optional[float] = None

class MultiStrategyScanner:
    """Main scanner class that implements both strategies"""
    
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.running = False
        self.scan_thread = None
    
    def _safe_float(self, row: pd.Series, key: str) -> Optional[float]:
        """Safely extract a single float value from a row. Handles duplicate column labels
        (returns the first non-null), Series values, and non-convertible types."""
        try:
            val = row.get(key)
            if isinstance(val, pd.Series):
                non_null = val.dropna()
                if len(non_null) == 0:
                    return None
                val = non_null.iloc[0]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)
        except Exception:
            return None
        
    def get_squeeze_scan_query(self, timeframe: str) -> Query:
        """Generate query for squeeze breakout patterns"""
        tf_suffix = TIMEFRAME_MAP.get(timeframe, '')
        
        squeeze_condition = Or(
            And(col(f'BB.upper{tf_suffix}') < col(f'KltChnl.upper{tf_suffix}'), col(f'BB.lower{tf_suffix}') > col(f'KltChnl.lower{tf_suffix}')),
        )
        
        base_conditions = And(
            col('beta_1_year') >= MIN_BETA,
            col('is_primary') == True,
            col('typespecs').has('common'),
            col('type') == 'stock',
            col('close').between(MIN_PRICE, MAX_PRICE),
            col('active_symbol') == True,
            col('exchange')== DEFAULT_EXCHANGE,
            col(f'Value.Traded{tf_suffix}') >= MIN_VALUE_TRADED,
        )
        
        query = Query().select(
            'name', 'close', 'volume', 'market_cap_basic', 'sector', 'exchange', 'logoid', 'beta_1_year', 'relative_volume_10d_calc|5',
            'ATR', 'ATR|5', 'ADX|60', 'BB.upper', 'BB.lower', 'EMA20', 'EMA200', 'EMA5|5',
            f'KltChnl.lower{tf_suffix}', f'KltChnl.upper{tf_suffix}', f'BB.lower{tf_suffix}', f'BB.upper{tf_suffix}',
            f'volume{tf_suffix}', f'SMA20{tf_suffix}', f'close{tf_suffix}', f'RSI{tf_suffix}',
            f'open{tf_suffix}', f'high{tf_suffix}', f'low{tf_suffix}', f'ATR{tf_suffix}', f'Value.Traded{tf_suffix}'
        ).where2(
            And(base_conditions, squeeze_condition)
        ).order_by(
            f'volume{tf_suffix}', ascending=False
        ).limit(MAX_RESULTS_PER_STRATEGY)
        
        return query
    
    def get_trending_pullback_query(self, timeframe: str) -> Query:
        """Generate query for trending stocks with pullback opportunities"""
        tf_suffix = TIMEFRAME_MAP.get(timeframe, '')
        
        trend_conditions = And(
            col(f'ADX{tf_suffix}').between(TREND_SETTINGS['adx_min'], TREND_SETTINGS['adx_max']),
            col(f'close{tf_suffix}') > col(f'SMA50{tf_suffix}'),
            col(f'EMA20{tf_suffix}') > col(f'EMA50{tf_suffix}'),
            col(f'RSI{tf_suffix}').between(30, 60)
        )
        
        base_conditions = And(
            col('beta_1_year') >= MIN_BETA,
            col('is_primary') == True,
            col('typespecs').has('common'),
            col('type') == 'stock',
            col('close').between(MIN_PRICE, MAX_PRICE),
            col('active_symbol') == True,
            col('exchange')== DEFAULT_EXCHANGE,
            col(f'Value.Traded{tf_suffix}') >= MIN_VALUE_TRADED,
        )

        query = Query().select(
            'name', 'close', 'volume', 'market_cap_basic', 'sector', 'exchange', 'logoid', 'beta_1_year', 'relative_volume_10d_calc|5',
            'ATR', 'ATR|5', 'ADX|60', 'BB.upper', 'BB.lower', 'EMA20', 'EMA200', 'EMA5|5',
            f'ADX{tf_suffix}', f'RSI{tf_suffix}', f'SMA20{tf_suffix}', f'SMA50{tf_suffix}',
            f'EMA20{tf_suffix}', f'EMA50{tf_suffix}', f'close{tf_suffix}', f'open{tf_suffix}', f'volume{tf_suffix}',
            f'high{tf_suffix}', f'low{tf_suffix}', f'ATR{tf_suffix}', f'Value.Traded{tf_suffix}'
        ).where2(
            And(base_conditions, trend_conditions)
        ).order_by(
            f'ADX{tf_suffix}', ascending=False
        ).limit(MAX_RESULTS_PER_STRATEGY)
        
        return query
    
    def calculate_potency_score(self, stock: pd.Series) -> (float, str, int):
        """
        Calculates the Potency Score based on RVOL and Setup quality.
        Applies the new, precise Squeeze condition post-data retrieval.
        """
        potency = 0
        tf_order_score = 0
        setup_type_score = 0
        setup_type = None

        # 1. RVOL Factor Score (Max 25 points)
        rvol_factor = min(self._safe_float(stock, 'relative_volume_10d_calc|5') or 0, 3.0)
        potency += WEIGHTS['W_RVOL'] * rvol_factor

        # 2. Setup Identification (Precise Squeeze and Pullback)
        squeeze_found = False
        for tf in [60, 30, 15]:
            bb_upper = self._safe_float(stock, f'BB.upper|{tf}')
            bb_lower = self._safe_float(stock, f'BB.lower|{tf}')
            klt_upper = self._safe_float(stock, f'KltChnl.upper|{tf}')
            klt_lower = self._safe_float(stock, f'KltChnl.lower|{tf}')

            if all(v is not None for v in [bb_upper, bb_lower, klt_upper, klt_lower]):
                bb_width = bb_upper - bb_lower
                klt_width = klt_upper - klt_lower
                if klt_width > 0 and (bb_width < WEIGHTS['SQZ_WIDTH_FACTOR'] * klt_width):
                    squeeze_found = True
                    if tf == 60 and tf_order_score < 3: tf_order_score = 3
                    elif tf == 30 and tf_order_score < 2: tf_order_score = 2
                    elif tf == 15 and tf_order_score < 1: tf_order_score = 1
                    if tf_order_score == 3: break

        stock['is_squeeze_ready'] = squeeze_found
        stock['is_pullback_ready'] = (self._safe_float(stock, 'ADX|60') or 0) >= 25.0

        if stock['is_pullback_ready']:
            setup_type = 'Pullback'
            setup_type_score = 8
            if tf_order_score < 3: tf_order_score = 3

        if stock['is_squeeze_ready']:
            setup_type = 'Breakout'
            setup_type_score = 10

        if setup_type is not None:
            potency += WEIGHTS['W_TF'] * tf_order_score
            potency += setup_type_score

        return round(potency, 2), setup_type, tf_order_score

    def calculate_trade_management(self, stock: pd.Series, entry_price: float) -> (float, float, int, float, float):
        """
        Calculates SL, Target, and Position Size based on 1:2 R:R and Max Risk.
        """
        atr_5m = self._safe_float(stock, 'ATR|5') or 0.0
        risk_per_share = WEIGHTS['ATR_SL_MULTIPLE'] * atr_5m
        target_distance = risk_per_share * WEIGHTS['RR_RATIO']

        if risk_per_share == 0 or entry_price == 0:
            return 0, 0, 0, 0.0, 0.0

        max_shares_by_risk = math.floor(WEIGHTS['MAX_RISK_RS'] / risk_per_share)
        max_shares_by_capital = math.floor(WEIGHTS['MAX_CAPITAL_RS'] / entry_price)
        position_size = int(min(max_shares_by_risk, max_shares_by_capital))

        total_trade_value = position_size * entry_price
        actual_risk = position_size * risk_per_share

        return risk_per_share, target_distance, position_size, total_trade_value, actual_risk
    
    def process_scan_results(self, df: pd.DataFrame, strategy_type: str, timeframe: str) -> List[ScanResult]:
        """Process scan results and convert to ScanResult objects"""
        results = []
        
        for _, row in df.iterrows():
            try:
                potency, setup_type, _ = self.calculate_potency_score(row)
                if setup_type is None:
                    continue

                entry_price = self._safe_float(row, 'close|5')
                risk_per_share, target_distance, position_size, total_trade_value, _ = self.calculate_trade_management(row, entry_price)

                if position_size > 0:
                    is_long = strategy_type in ['long_breakout', 'long_continuation']
                    stop_loss = entry_price - risk_per_share if is_long else entry_price + risk_per_share
                    target_price = entry_price + target_distance if is_long else entry_price - target_distance

                    result = ScanResult(
                        ticker=row.get('name', 'Unknown'),
                        strategy_type=strategy_type,
                        timeframe=timeframe,
                        price=self._safe_float(row, 'close') or 0.0,
                        volume=self._safe_float(row, 'volume') or 0.0,
                        market_cap=row.get('market_cap_basic', 0),
                        sector=row.get('sector', 'Unknown'),
                        exchange=row.get('exchange', 'Unknown'),
                        squeeze_status='Active' if setup_type == 'Breakout' else None,
                        trend_strength=self._safe_float(row, 'ADX'),
                        pullback_depth=(self._safe_float(row, 'EMA20') - self._safe_float(row, 'close')) / self._safe_float(row, 'EMA20') if self._safe_float(row, 'EMA20') else None,
                        adx=self._safe_float(row, 'ADX'),
                        rsi=self._safe_float(row, 'RSI'),
                        sma50=self._safe_float(row, 'SMA50'),
                        ema20=self._safe_float(row, 'EMA20'),
                        potency_score=potency,
                        timestamp=datetime.now(),
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target_price=target_price,
                        position_size=position_size,
                        total_trade_value=total_trade_value,
                    )
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing row for {row.get('name', 'Unknown')}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        return results
    
    def run_strategy_scan(self, strategy_type: str, timeframes: List[str]) -> List[ScanResult]:
        """Run scan for a specific strategy across multiple timeframes"""
        all_results = []
        
        for timeframe in timeframes:
            try:
                logger.info(f"Running {strategy_type} scan for {timeframe} timeframe...")
                
                if strategy_type == 'High_Potential':
                    query = self.get_squeeze_scan_query(timeframe)
                else:
                    query = self.get_trending_pullback_query(timeframe)
                
                _, df = query.get_scanner_data()
                
                if df.empty:
                    logger.info(f"No results found for {strategy_type} {timeframe}")
                    continue
                
                # Process results
                results = self.process_scan_results(df, strategy_type, timeframe)
                all_results.extend(results)
                
                logger.info(f"Found {len(results)} {strategy_type} opportunities in {timeframe}")
                
            except Exception as e:
                logger.error(f"Error running {strategy_type} scan for {timeframe}: {e}")
                continue
        
        return all_results
    
    def save_results_to_db(self, results: List[ScanResult]) -> None:
        """Save scan results to MongoDB"""
        try:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame([vars(result) for result in results])
            
            if df.empty:
                return
            
            # Add metadata
            df['scan_id'] = f"scan_{int(time.time())}"
            df['created_at'] = datetime.now()
            
            # Convert to records and save
            records = df.to_dict('records')
            self.collection.insert_many(records)
            
            # Clean up old results (keep only last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.collection.delete_many({'timestamp': {'$lt': cutoff_time}})
            
            logger.info(f"Saved {len(records)} results to database")
            
        except Exception as e:
            logger.error(f"Error saving results to database: {e}")
    
    def get_latest_results(self) -> pd.DataFrame:
        """Get latest scan results from database"""
        try:
            # Get results from last scan
            cursor = self.collection.find().sort('created_at', -1).limit(1000)
            df = pd.DataFrame(list(cursor))
            if '_id' in df.columns: df = df.drop(columns=['_id'])
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert timestamp and sort
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(['strategy_type', 'potency_score'], ascending=[True, False])
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving results from database: {e}")
            return pd.DataFrame()
    
    def run_full_scan(self) -> pd.DataFrame:
        """Run complete multi-strategy scan"""
        logger.info("Starting full multi-strategy scan...")
        
        long_breakout_results = self.run_strategy_scan('long_breakout', HIGH_POTENTIAL_TFS)
        short_breakout_results = self.run_strategy_scan('short_breakout', HIGH_POTENTIAL_TFS)
        long_continuation_results = self.run_strategy_scan('long_continuation', HIGH_PROBABILITY_TFS)
        short_continuation_results = self.run_strategy_scan('short_continuation', HIGH_PROBABILITY_TFS)
        
        all_results = long_breakout_results + short_breakout_results + long_continuation_results + short_continuation_results
        
        if all_results:
            self.save_results_to_db(all_results)
            df = pd.DataFrame([vars(result) for result in all_results])
            logger.info(f"Full scan completed. Found {len(all_results)} total opportunities.")
            return df
        else:
            logger.warning("No results found in full scan")
            return pd.DataFrame()
    
    def start_continuous_scanning(self):
        """Start continuous scanning in background thread"""
        self.running = True
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
        logger.info("Started continuous scanning")
    
    def stop_continuous_scanning(self):
        """Stop continuous scanning"""
        self.running = False
        if self.scan_thread:
            self.scan_thread.join(timeout=10)
        logger.info("Stopped continuous scanning")
    
    def _scan_loop(self):
        """Main scanning loop"""
        while self.running:
            try:
                self.run_full_scan()
                time.sleep(SCAN_INTERVAL)
            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_continuous_scanning()
        self.client.close()

# Global scanner instance
scanner = MultiStrategyScanner()

def get_scanner():
    """Get global scanner instance"""
    return scanner