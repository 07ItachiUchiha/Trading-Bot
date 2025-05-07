import sqlite3
import pandas as pd
import datetime
import json
import os
import sys
from pathlib import Path
import bcrypt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "trading_bot.db"

def ensure_db_exists():
    """Create database and tables if they don't exist"""
    if not DB_PATH.parent.exists():
        os.makedirs(DB_PATH.parent)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Create trades table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL,
        stop_loss REAL,
        targets TEXT,
        size REAL NOT NULL,
        entry_time TIMESTAMP NOT NULL,
        exit_time TIMESTAMP,
        pnl REAL,
        status TEXT NOT NULL
    )
    ''')
    
    # Create users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE,
        api_key TEXT,
        api_secret TEXT,
        last_login TIMESTAMP,
        role TEXT
    )
    ''')
    
    # Create admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if cursor.fetchone() is None:
        hashed_password = bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cursor.execute("""
            INSERT INTO users (username, password, email, role) 
            VALUES (?, ?, ?, ?)
            """, 
            ("admin", hashed_password, "admin@tradingbot.com", "admin")
        )
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get a connection to the database"""
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def get_trades_from_db(symbol=None, status=None):
    """Get trades from the database with optional filtering"""
    conn = get_db_connection()
    
    query = "SELECT * FROM trades"
    params = []
    
    if symbol or status:
        query += " WHERE"
        
        if symbol:
            query += " symbol = ?"
            params.append(symbol)
            
        if symbol and status:
            query += " AND"
            
        if status:
            query += " status = ?"
            params.append(status)
    
    query += " ORDER BY entry_time DESC"
    
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    # Convert timestamp strings to datetime
    if not df.empty:
        df['entry_time'] = pd.to_datetime(df['entry_time'], format='mixed', errors='coerce')
        df['exit_time'] = pd.to_datetime(df['exit_time'], format='mixed', errors='coerce')
        # Convert targets from JSON string to list
        df['targets'] = df['targets'].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x else []
        )
    return df

def add_trade_to_db(trade_data):
    """Add a new trade to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Convert targets list to JSON string
    if 'targets' in trade_data and isinstance(trade_data['targets'], list):
        trade_data['targets'] = json.dumps(trade_data['targets'])
    
    # Ensure consistent timestamp format
    if isinstance(trade_data.get('entry_time'), datetime.datetime):
        trade_data['entry_time'] = trade_data['entry_time'].strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
    INSERT INTO trades (
        symbol, direction, entry_price, stop_loss, 
        targets, size, entry_time, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        trade_data['symbol'],
        trade_data['direction'],
        trade_data['entry_price'],
        trade_data['stop_loss'],
        trade_data['targets'],
        trade_data['size'],
        trade_data['entry_time'],
        trade_data['status']
    ))
    
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return trade_id

def update_trade_in_db(trade_id, update_data):
    """Update an existing trade in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ensure consistent timestamp format
    if 'exit_time' in update_data and isinstance(update_data.get('exit_time'), datetime.datetime):
        update_data['exit_time'] = update_data['exit_time'].strftime("%Y-%m-%d %H:%M:%S")
    
    set_clauses = []
    params = []
    
    for key, value in update_data.items():
        set_clauses.append(f"{key} = ?")
        params.append(value)
    
    params.append(trade_id)
    
    query = f"UPDATE trades SET {', '.join(set_clauses)} WHERE id = ?"
    cursor.execute(query, params)
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0

def delete_trade_from_db(trade_id):
    """Delete a trade from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0

def get_user_from_db(username):
    """Get user data from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    conn.close()
    
    if user:
        return dict(user)
    
    return None

def add_user_to_db(user_data):
    """Add a new user to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO users (
            username, password, email, api_key, api_secret
        ) VALUES (?, ?, ?, ?, ?)
        ''', (
            user_data['username'],
            user_data['password'],
            user_data.get('email', ''),
            user_data.get('api_key', ''),
            user_data.get('api_secret', '')
        ))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None
    
def update_user_in_db(username, update_data):
    """Update an existing user in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    set_clauses = []
    params = []
    
    for key, value in update_data.items():
        set_clauses.append(f"{key} = ?")
        params.append(value)
    
    params.append(username)
    
    query = f"UPDATE users SET {', '.join(set_clauses)} WHERE username = ?"
    cursor.execute(query, params)
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0
    
def update_login_timestamp(username):
    """Update the last login timestamp for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Use consistent datetime format
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute(
        "UPDATE users SET last_login = ? WHERE username = ?",
        (current_time, username)
    )
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0

def get_trade_stats():
    """Get overall trading statistics"""
    conn = get_db_connection()
    
    # Get total trades
    total_trades = pd.read_sql("SELECT COUNT(*) as count FROM trades", conn).iloc[0]['count']
    
    # Get closed trades stats
    closed_trades_df = pd.read_sql(
        "SELECT * FROM trades WHERE status = 'closed'",
        conn
    )
    
    stats = {
        'total_trades': total_trades,
        'closed_trades': len(closed_trades_df),
        'open_trades': total_trades - len(closed_trades_df)
    }
    
    if not closed_trades_df.empty:
        win_trades = closed_trades_df[closed_trades_df['pnl'] > 0]
        loss_trades = closed_trades_df[closed_trades_df['pnl'] <= 0]
        
        stats.update({
            'total_pnl': closed_trades_df['pnl'].sum(),
            'win_trades': len(win_trades),
            'loss_trades': len(loss_trades),
            'win_rate': len(win_trades) / len(closed_trades_df) * 100 if len(closed_trades_df) > 0 else 0,
            'avg_win': win_trades['pnl'].mean() if len(win_trades) > 0 else 0,
            'avg_loss': loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        })
    
    conn.close()
    return stats