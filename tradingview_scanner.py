from tradingview_screener import Query, col
import pandas as pd
import numpy as np

def get_screener_data(filters=None):
    """
    Gets screener data from TradingView.
    """
    query = Query()

    if filters:
        conditions = []
        for key, value in filters.items():
            if value:
                if key == 'performance':
                    if value == '1':
                        conditions.append(col('change') > 0)
                    elif value == '-1':
                        conditions.append(col('change') < 0)
                elif key == 'market_cap':
                    conditions.append(col('market_cap_basic') > float(value))
                elif key == 'volatility':
                    conditions.append(col('Volatility.D') > float(value))
                elif key == 'rsi':
                    if value == '70':
                        conditions.append(col('RSI') > 70)
                    elif value == '30':
                        conditions.append(col('RSI') < 30)
                elif key == 'gap':
                    if value == '1':
                        conditions.append(col('gap') > 0)
                    elif value == '-1':
                        conditions.append(col('gap') < 0)
        if conditions:
            query = query.where(*conditions)

    query = query.select(
        'name',
        'close',
        'volume',
        'market_cap_basic'
    )

    _, df = query.get_scanner_data()

    if df is not None:
        df = df.reset_index()
        df.replace(np.nan, None, inplace=True)


    return df

if __name__ == '__main__':
    data = get_screener_data({'performance': '1'})
    if data is not None:
        print(f"Successfully retrieved {len(data)} rows.")
        print("Final Columns:", data.columns)
        print(data.head())
    else:
        print("No data retrieved.")
