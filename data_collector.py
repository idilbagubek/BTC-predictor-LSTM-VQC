import sqlite3
import time
import os
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import pandas as pd

from config import data_config, DB_PATH


def init_database() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER NOT NULL,
            PRIMARY KEY (symbol, interval, open_time)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_candles_time 
        ON candles (symbol, interval, open_time DESC)
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def fetch_candles_from_binance(
    symbol: str = None,
    interval: str = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000,
    max_retries: int = 5
) -> List[Dict]:
    
    import random
    
    symbol = symbol or data_config.symbol
    interval = interval or data_config.interval
    
    url = f"{data_config.exchange_api_base}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)
    }
    
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited (429). Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            
            if response.status_code >= 500:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Server error ({response.status_code}). Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            candles = []
            for row in data:
                candles.append({
                    "symbol": symbol,
                    "interval": interval,
                    "open_time": row[0],
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": row[6]
                })
            
            return candles
        
        except requests.exceptions.Timeout as e:
            last_exception = e
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Request timeout. Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait_time)
            
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Connection error. Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait_time)
            
        except requests.exceptions.RequestException as e:
            last_exception = e
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Request error: {e}. Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait_time)
    
    print(f"Failed to fetch candles after {max_retries} attempts. Last error: {last_exception}")
    return []


def store_candles(candles: List[Dict]) -> int:
    if not candles:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    inserted = 0
    for candle in candles:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO candles 
                (symbol, interval, open_time, open, high, low, close, volume, close_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candle["symbol"],
                candle["interval"],
                candle["open_time"],
                candle["open"],
                candle["high"],
                candle["low"],
                candle["close"],
                candle["volume"],
                candle["close_time"]
            ))
            if cursor.rowcount > 0:
                inserted += 1
        except sqlite3.Error as e:
            print(f"Error inserting candle: {e}")
    
    conn.commit()
    conn.close()
    return inserted


def get_latest_candle_time(symbol: str = None, interval: str = None) -> Optional[int]:
    """Get the most recent candle timestamp in database"""
    symbol = symbol or data_config.symbol
    interval = interval or data_config.interval
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT MAX(open_time) FROM candles
        WHERE symbol = ? AND interval = ?
    """, (symbol, interval))
    
    result = cursor.fetchone()[0]
    conn.close()
    return result


def get_oldest_candle_time(symbol: str = None, interval: str = None) -> Optional[int]:
    """Get the oldest candle timestamp in database"""
    symbol = symbol or data_config.symbol
    interval = interval or data_config.interval
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT MIN(open_time) FROM candles
        WHERE symbol = ? AND interval = ?
    """, (symbol, interval))
    
    result = cursor.fetchone()[0]
    conn.close()
    return result


def get_candle_count(symbol: str = None, interval: str = None) -> int:
    """Get total number of candles in database"""
    symbol = symbol or data_config.symbol
    interval = interval or data_config.interval
    
    if not os.path.exists(DB_PATH):
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT COUNT(*) FROM candles
            WHERE symbol = ? AND interval = ?
        """, (symbol, interval))
        
        result = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        result = 0
    
    conn.close()
    return result


def load_candles_as_dataframe(
    symbol: str = None,
    interval: str = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:

    symbol = symbol or data_config.symbol
    interval = interval or data_config.interval
    
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT open_time, open, high, low, close, volume, close_time
        FROM candles
        WHERE symbol = ? AND interval = ?
    """
    params = [symbol, interval]
    
    if start_time:
        query += " AND open_time >= ?"
        params.append(start_time)
    if end_time:
        query += " AND open_time <= ?"
        params.append(end_time)
    
    query += " ORDER BY open_time ASC"
    
    if limit:
        query = f"""
            SELECT * FROM (
                SELECT open_time, open, high, low, close, volume, close_time
                FROM candles
                WHERE symbol = ? AND interval = ?
                ORDER BY open_time DESC
                LIMIT ?
            ) ORDER BY open_time ASC
        """
        params = [symbol, interval, limit]
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
    
    return df


def fetch_historical_data(
    days: int = 30,
    symbol: str = None,
    interval: str = None,
    verbose: bool = True
) -> int:
    symbol = symbol or data_config.symbol
    interval = interval or data_config.interval
    
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    oldest = get_oldest_candle_time(symbol, interval)
    latest = get_latest_candle_time(symbol, interval)
    
    total_fetched = 0

    if oldest is None or start_time < oldest:
        if verbose:
            print(f"Fetching historical data from {days} days ago...")
        
        current_end = oldest if oldest else end_time
        current_start = start_time
        
        while current_start < current_end:
            candles = fetch_candles_from_binance(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000
            )
            
            if not candles:
                break
            
            inserted = store_candles(candles)
            total_fetched += inserted
            
            if verbose:
                dt = datetime.fromtimestamp(candles[-1]["open_time"]/1000, tz=timezone.utc)
                print(f"  Fetched {len(candles)} candles up to {dt.strftime('%Y-%m-%d %H:%M')}, "
                      f"inserted {inserted} new")
            
            # Move window forward
            current_start = candles[-1]["open_time"] + 1
            time.sleep(0.1)  # Rate limiting
    
    # Fetch recent data to fill gaps 
    if latest:
        if verbose:
            print("Fetching recent data to fill gaps...")
        
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        interval_ms = get_interval_ms(interval)
        end_time = now_ms - interval_ms  # Exclude currently-forming candle
        
        candles = fetch_candles_from_binance(
            symbol=symbol,
            interval=interval,
            start_time=latest + 1,
            end_time=end_time,
            limit=1000
        )
        
        if candles:
            candles = filter_closed_candles(candles, interval)
            inserted = store_candles(candles)
            total_fetched += inserted
            if verbose:
                print(f"  Fetched {len(candles)} recent candles, inserted {inserted} new")
    
    if verbose:
        count = get_candle_count(symbol, interval)
        print(f"Total candles in database: {count}")
    
    return total_fetched


def get_interval_ms(interval: str) -> int:
    """Convert interval string to milliseconds"""
    multipliers = {
        's': 1000,
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        'd': 24 * 60 * 60 * 1000,
        'w': 7 * 24 * 60 * 60 * 1000
    }
    
    unit = interval[-1]
    value = int(interval[:-1])
    
    return value * multipliers.get(unit, 60 * 1000)


def filter_closed_candles(candles: List[Dict], interval: str = None) -> List[Dict]:
    
    interval = interval or data_config.interval
    interval_ms = get_interval_ms(interval)
    now_ms = int(time.time() * 1000)
    
    # A candle is closed if close_time < now - small buffer 
    cutoff_time = now_ms - 1000
    
    return [c for c in candles if c['close_time'] <= cutoff_time]


def update_candles(symbol: str = None, interval: str = None, max_iterations: int = 10) -> int:
    symbol = symbol or data_config.symbol
    interval = interval or data_config.interval
    interval_ms = get_interval_ms(interval)
    
    total_inserted = 0
    
    for iteration in range(max_iterations):
        latest = get_latest_candle_time(symbol, interval)
        now_ms = int(time.time() * 1000)
    
        end_time = now_ms - interval_ms
        
        if latest:
            if latest >= end_time - interval_ms:
                break
                
            candles = fetch_candles_from_binance(
                symbol=symbol,
                interval=interval,
                start_time=latest + 1,
                end_time=end_time,
                limit=1000
            )
        else:
            candles = fetch_candles_from_binance(
                symbol=symbol,
                interval=interval,
                end_time=end_time,
                limit=1000
            )
        
        if not candles:
            break
        
        candles = filter_closed_candles(candles, interval)
        
        if not candles:
            break
        
        inserted = store_candles(candles)
        total_inserted += inserted
        
        if inserted == 0 or len(candles) < 1000:
            break
    
    return total_inserted


if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    
    print("\nFetching historical data...")
    fetch_historical_data(days=7, verbose=True)
    
    print("\nLoading sample data...")
    df = load_candles_as_dataframe(limit=10)
    print(df)
