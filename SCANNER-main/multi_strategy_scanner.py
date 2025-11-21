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
        
        # Squeeze conditions - use And() for complex conditions
        squeeze_condition = And(
            col(f'KltChnl.lower{tf_suffix}') > col(f'BB.lower{tf_suffix}'),  # Squeeze condition
            col(f'KltChnl.upper{tf_suffix}') < col(f'BB.upper{tf_suffix}'),  # Squeeze condition
            col(f'volume{tf_suffix}') >= MIN_VOLUME,  # Volume filter
            col(f'close{tf_suffix}') > col(f'SMA20{tf_suffix}')  # Price above 20 SMA
        )
        
        # Volume and price filters
        base_conditions = And(
            col('close') >= MIN_PRICE,
            col('close') <= MAX_PRICE,
            col('volume') >= MIN_VOLUME,
            col(f'Value.Traded{tf_suffix}') >= MIN_VALUE_TRADED,
        )
        
        # Combine conditions
        query = Query().select(
            'name', 'close', 'volume', 'market_cap_basic', 'sector', 'exchange',
            f'KltChnl.lower{tf_suffix}', f'KltChnl.upper{tf_suffix}', f'BB.lower{tf_suffix}', f'BB.upper{tf_suffix}',
            f'volume{tf_suffix}', f'SMA20{tf_suffix}', f'close{tf_suffix}', f'RSI{tf_suffix}',
            f'open{tf_suffix}', f'high{tf_suffix}', f'low{tf_suffix}', f'ATR{tf_suffix}', f'Value.Traded{tf_suffix}'
        ).where2(
            And(squeeze_condition, base_conditions)
        ).order_by(
            f'volume{tf_suffix}', ascending=False
        ).limit(MAX_RESULTS_PER_STRATEGY)
        
        return query
    
    def get_trending_pullback_query(self, timeframe: str) -> Query:
        """Generate query for trending stocks with pullback opportunities"""
        tf_suffix = TIMEFRAME_MAP.get(timeframe, '')
        
        # Trend strength conditions - simplified to avoid multiplication issues
        trend_conditions = And(
            col(f'ADX{tf_suffix}') >= TREND_SETTINGS['adx_min'],
            col(f'ADX{tf_suffix}') <= TREND_SETTINGS['adx_max'],
            col(f'close{tf_suffix}') > col(f'SMA50{tf_suffix}'),  # Above 50 SMA
            col(f'EMA20{tf_suffix}') > col(f'EMA50{tf_suffix}'),    # EMA alignment
            # Use RSI to identify pullback instead of percentage range
            col(f'RSI{tf_suffix}').between(30, 60)  # RSI in pullback zone
        )
        
        # Volume and price filters
        base_conditions = And(
            col('close') >= MIN_PRICE,
            col('close') <= MAX_PRICE,
            col('volume') >= MIN_VOLUME,
            col(f'Value.Traded{tf_suffix}') >= MIN_VALUE_TRADED,
        )
        
        # Combine conditions
        query = Query().select(
            'name', 'close', 'volume', 'market_cap_basic', 'sector', 'exchange',
            f'ADX{tf_suffix}', f'RSI{tf_suffix}', f'SMA20{tf_suffix}', f'SMA50{tf_suffix}',
            f'EMA20{tf_suffix}', f'EMA50{tf_suffix}', f'close{tf_suffix}', f'open{tf_suffix}', f'volume{tf_suffix}',
            f'high{tf_suffix}', f'low{tf_suffix}', f'ATR{tf_suffix}', f'Value.Traded{tf_suffix}'
        ).where2(
            And(trend_conditions, base_conditions)
        ).order_by(
            f'ADX{tf_suffix}', ascending=False
        ).limit(MAX_RESULTS_PER_STRATEGY)
        
        return query
    
    def calculate_potency_score(self, row: pd.Series, strategy_type: str) -> float:
        """Calculate potency score based on strategy type with robust scalar handling"""
        score = 0.0

        # Common metrics
        close_val = self._safe_float(row, 'close')
        atr_val = self._safe_float(row, 'ATR')
        rsi_val = self._safe_float(row, 'RSI')
        adx_val = self._safe_float(row, 'ADX')

        if strategy_type == 'High_Potential':
            # Volume ratio contribution (prefer provided, else fallback if possible)
            vr = row.get('volume_ratio')
            if isinstance(vr, pd.Series):
                vr = vr.dropna().iloc[0] if len(vr.dropna()) else None
            if vr is None:
                vol = self._safe_float(row, 'volume')
                avg_vol = self._safe_float(row, 'average_volume_10d_calc')
                if vol is not None and avg_vol is not None and avg_vol > 0:
                    vr = vol / avg_vol
            if vr is not None:
                try:
                    score += min(float(vr) * 0.2, 0.2)
                except Exception:
                    pass

            # RSI contribution if present
            if rsi_val is not None:
                rsi_score = 1.0 - abs(rsi_val - 50.0) / 50.0
                score += rsi_score * 0.3

            # Volatility contribution based on ATR/close
            if atr_val is not None and close_val is not None and close_val > 0:
                volatility_ratio = atr_val / close_val
                score += min(volatility_ratio * 10.0, 0.3)

            score += 0.2  # Base score for squeeze potential

        elif strategy_type == 'High_Probability':
            # ADX contribution
            if adx_val is not None:
                adx_score = min(adx_val / 50.0, 1.0)
                score += adx_score * 0.4

            # RSI contribution
            if rsi_val is not None:
                if rsi_val <= PULLBACK_SETTINGS['rsi_oversold']:
                    score += 0.3
                elif rsi_val <= 50.0:
                    score += 0.2

            # Volume ratio contribution if available
            vr = row.get('volume_ratio')
            if isinstance(vr, pd.Series):
                vr = vr.dropna().iloc[0] if len(vr.dropna()) else None
            if vr is None:
                vol = self._safe_float(row, 'volume')
                avg_vol = self._safe_float(row, 'average_volume_10d_calc')
                if vol is not None and avg_vol is not None and avg_vol > 0:
                    vr = vol / avg_vol
            if vr is not None:
                try:
                    score += min(float(vr) * 0.1, 0.2)
                except Exception:
                    pass

            # Pullback depth contribution (compute if not provided)
            pdpth = row.get('pullback_depth')
            if isinstance(pdpth, pd.Series):
                pdpth = pdpth.dropna().iloc[0] if len(pdpth.dropna()) else None
            if pdpth is None:
                ema20 = self._safe_float(row, 'EMA20')
                if ema20 is not None and close_val is not None and ema20 > 0:
                    pdpth = (ema20 - close_val) / ema20
            if pdpth is not None:
                try:
                    if PULLBACK_SETTINGS['min_pullback_depth'] <= pdpth <= PULLBACK_SETTINGS['max_pullback_depth']:
                        score += 0.1
                except Exception:
                    pass

        return min(score, 1.0)
    
    def process_scan_results(self, df: pd.DataFrame, strategy_type: str, timeframe: str) -> List[ScanResult]:
        """Process scan results and convert to ScanResult objects"""
        results = []
        
        for _, row in df.iterrows():
            try:
                # Calculate derived metrics
                if strategy_type == 'High_Probability':
                    pullback_depth = None
                    close_value = self._safe_float(row, 'close')
                    ema20_value = self._safe_float(row, 'EMA20')
                    if ema20_value is not None and close_value is not None and ema20_value > 0:
                        pullback_depth = (ema20_value - close_value) / ema20_value
                    volume_ratio = None
                    vol = self._safe_float(row, 'volume')
                    avg_vol = self._safe_float(row, 'average_volume_10d_calc')
                    if vol is not None and avg_vol is not None and avg_vol > 0:
                        volume_ratio = vol / avg_vol
                else:
                    pullback_depth = None
                    # keep provided ratio if exists
                    volume_ratio = row.get('volume_ratio', None)
                
                # Create ScanResult object
                result = ScanResult(
                    ticker=row.get('name', 'Unknown'),
                    strategy_type=strategy_type,
                    timeframe=timeframe,
                    price=self._safe_float(row, 'close') or 0.0,
                    volume=self._safe_float(row, 'volume') or 0.0,
                    market_cap=row.get('market_cap_basic', 0),
                    sector=row.get('sector', 'Unknown'),
                    exchange=row.get('exchange', 'Unknown'),
                    squeeze_status='Active' if strategy_type == 'High_Potential' else None,
                    trend_strength=self._safe_float(row, 'ADX'),
                    pullback_depth=pullback_depth,
                    adx=self._safe_float(row, 'ADX'),
                    rsi=self._safe_float(row, 'RSI'),
                    sma50=self._safe_float(row, 'SMA50'),
                    ema20=self._safe_float(row, 'EMA20'),
                    bb_position=None,
                    volume_ratio=volume_ratio,
                    potency_score=self.calculate_potency_score(row, strategy_type),
                    timestamp=datetime.now()
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
        
        # Run both strategies
        squeeze_results = self.run_strategy_scan('High_Potential', HIGH_POTENTIAL_TFS)
        trending_results = self.run_strategy_scan('High_Probability', HIGH_PROBABILITY_TFS)
        
        # Combine results
        all_results = squeeze_results + trending_results
        
        if all_results:
            # Save to database
            self.save_results_to_db(all_results)
            
            # Return as DataFrame
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