# Multi-Strategy Trading Scanner System

A comprehensive multi-timeframe, multi-strategy trading scanner that combines high-potential squeeze detection with high-probability trending pullback identification.

## 🎯 System Overview

This system implements a dual-strategy approach to stock scanning:

1. **High-Potential Strategy (Squeeze/Breakout)**: Identifies stocks with volatility compression on lower timeframes (15m, 30m, 60m) that are poised for explosive moves.

2. **High-Probability Strategy (Trending/Pullback)**: Finds stocks with strong established trends on higher timeframes (60m, 4h, Daily) that are experiencing healthy pullbacks for continuation entries.

## 🏗️ Architecture

### Core Components

- **`multi_strategy_scanner.py`**: Main scanning engine with dual-strategy implementation
- **`server.py`**: Flask server with REST API endpoints
- **`config.py`**: Configuration settings and constants
- **`templates/dashboard.html`**: Unified web dashboard with heatmap and table views
- **`requirements.txt`**: Python dependencies

### Key Features

- **Multi-Timeframe Analysis**: Scans across multiple timeframes simultaneously
- **Potency Scoring**: Calculates confidence scores for each opportunity
- **Real-time Updates**: Continuous background scanning with configurable intervals
- **Unified Dashboard**: Single interface showing both strategy results
- **REST API**: Full API for data access and scanner control
- **MongoDB Integration**: Persistent storage of scan results

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install MongoDB

Ensure MongoDB is installed and running locally, or update the `MONGODB_URI` in `config.py`.

### 3. Start the System

```bash
python server.py
```

### 4. Access Dashboard

Open your browser to `http://localhost:5000`

## 📊 Dashboard Features

### Squeeze Heatmap (Top Panel)
- Visual grid showing high-potential squeeze opportunities
- Color-coded potency scores
- Clickable tiles for quick Yahoo Finance lookup
- Real-time updates

### Pullback Table (Bottom Panel)
- Detailed table of trending pullback opportunities
- Key metrics: ADX, RSI, Pullback Depth, Potency Score
- Sortable by potency score
- Sector information included

### Control Panel
- Manual scan trigger
- Auto-refresh indicator
- Connection status
- Statistics overview

## 🔧 Configuration

Edit `config.py` to customize:

- **Timeframes**: `HIGH_POTENTIAL_TFS`, `HIGH_PROBABILITY_TFS`
- **Market Filters**: Price ranges, volume requirements
- **Strategy Settings**: ADX thresholds, RSI levels, pullback parameters
- **Scanning Intervals**: Update frequency
- **Database Settings**: MongoDB connection

## 📡 API Endpoints

- `GET /` - Main dashboard
- `GET /api/data` - Latest scan results
- `POST /api/run-scan` - Trigger manual scan
- `GET /api/settings` - Get current settings
- `POST /api/settings` - Update settings
- `GET /api/stats` - Scanner statistics
- `GET /api/health` - Health check

## 🎯 Trading Strategy Implementation

### High-Potential (Squeeze) Strategy
1. **Setup**: Identify stocks in volatility compression on 15m/30m/60m
2. **Alert**: Wait for breakout confirmation on higher timeframe
3. **Entry**: Switch to 3m/5m for precise entry timing
4. **Confirmation**: Look for volume spike or reversal pattern

### High-Probability (Trending/Pullback) Strategy
1. **Setup**: Find strong trending stocks (ADX > 25) on 60m/Daily
2. **Alert**: Wait for pullback to key support (EMA20/SMA50)
3. **Entry**: Switch to 3m/5m for reversal confirmation
4. **Confirmation**: Look for rejection candle or RSI bounce

## 🔍 Technical Implementation

### Scanning Logic
- Uses TradingView Screener API for market data
- Implements custom queries for each strategy
- Calculates potency scores based on multiple factors
- Stores results in MongoDB with timestamps

### Data Processing
- Converts API responses to structured DataFrames
- Calculates derived metrics (pullback depth, volume ratios)
- Applies filtering and sorting
- Handles data cleanup and validation

### Error Handling
- Comprehensive error handling throughout
- Graceful degradation on API failures
- Automatic retry mechanisms
- Detailed logging for debugging

## 📈 Performance Optimization

- Background threading for continuous scanning
- Efficient database queries with indexing
- Caching of recent results
- Configurable scan intervals to manage API usage

## 🔒 Security Considerations

- Input validation on all API endpoints
- CORS configuration for web access
- No sensitive data exposure in responses
- Rate limiting ready for production deployment

## 🛠️ Troubleshooting

### Common Issues

1. **MongoDB Connection Errors**
   - Ensure MongoDB is running: `mongod --dbpath /path/to/data`
   - Check connection string in `config.py`

2. **TradingView API Errors**
   - Verify internet connectivity
   - Check API rate limits
   - Review query parameters

3. **No Results Found**
   - Adjust filter criteria in config
   - Check market hours
   - Verify timeframe availability

### Logs

Check console output for detailed logging information. Logs include:
- Scan execution details
- Error messages with stack traces
- Performance metrics
- API response information

## 🚀 Future Enhancements

- Additional strategy types (momentum, mean reversion)
- Advanced filtering and sorting options
- Historical data analysis
- Alert notifications (email, SMS)
- Mobile app integration
- Machine learning integration for score prediction
- Social sentiment analysis
- Options flow integration

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Verify all dependencies are installed correctly
4. Ensure MongoDB is running and accessible

---

**Happy Trading!** 🚀📈