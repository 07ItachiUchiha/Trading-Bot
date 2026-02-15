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


def _bootstrap_admin_if_configured(cursor):
    """Create a bootstrap admin only when explicit env vars are provided."""
    username = os.environ.get("TRADING_BOT_BOOTSTRAP_ADMIN_USERNAME", "").strip()
    password = os.environ.get("TRADING_BOT_BOOTSTRAP_ADMIN_PASSWORD", "").strip()
    email = os.environ.get("TRADING_BOT_BOOTSTRAP_ADMIN_EMAIL", "admin@localhost").strip()

    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    if user_count > 0:
        return

    if not username or not password:
        print(
            "No users exist yet. Set TRADING_BOT_BOOTSTRAP_ADMIN_USERNAME and "
            "TRADING_BOT_BOOTSTRAP_ADMIN_PASSWORD to create the first admin account."
        )
        return

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    cursor.execute(
        """
        INSERT INTO users (username, password, email, role)
        VALUES (?, ?, ?, ?)
        """,
        (username, hashed_password, email, "admin"),
    )

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
    
    _bootstrap_admin_if_configured(cursor)
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get a connection to the database"""
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def get_trades_from_db(symbol=None, status=None, limit=100):
    """Fetch trades with optional symbol/status filter."""
    conn = None
    try:
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
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        df = pd.read_sql(query, conn, params=params)
        
        # Convert timestamp strings to datetime
        if not df.empty:
            df['entry_time'] = pd.to_datetime(df['entry_time'], format='mixed', errors='coerce')
            df['exit_time'] = pd.to_datetime(df['exit_time'], format='mixed', errors='coerce')
            # Convert targets from JSON string to list
            df['targets'] = df['targets'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x else []
            )
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"Error fetching trades from database: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.close()
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'id', 'symbol', 'direction', 'entry_price', 'exit_price',
            'stop_loss', 'targets', 'size', 'pnl', 'entry_time', 'exit_time', 'status'
        ])

def add_trade_to_db(trade_data):
    """Insert a new trade record."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Print debug info
        print(f"Adding trade: {trade_data}")
        
        # Format dates properly
        if isinstance(trade_data.get('entry_time'), datetime.datetime):
            entry_time = trade_data['entry_time'].strftime("%Y-%m-%d %H:%M:%S")
        else:
            entry_time = trade_data.get('entry_time')
            
        # Convert targets list to a string
        targets_str = json.dumps(trade_data.get('targets', []))
        
        # Validate required fields
        if not trade_data.get('symbol') or not entry_time:
            print("Error: Missing required fields for trade")
            return False
        
        cursor.execute(
            """
            INSERT INTO trades (
                symbol, direction, entry_price, exit_price, 
                stop_loss, targets, size, pnl, 
                entry_time, exit_time, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_data.get('symbol', 'UNKNOWN'),
                trade_data.get('direction', 'long'),
                float(trade_data.get('entry_price', 0.0)),
                None,  # exit_price starts as NULL
                float(trade_data.get('stop_loss', 0.0)),
                targets_str,
                float(trade_data.get('size', 0.0)),
                0.0,   # pnl starts at 0
                entry_time,
                None,  # exit_time starts as NULL
                trade_data.get('status', 'open')
            )
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding trade to database: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.close()
        return False

def update_trade_in_db(trade_id, update_data):
    """Update fields on an existing trade."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if trade exists
        cursor.execute("SELECT id FROM trades WHERE id = ?", (trade_id,))
        if not cursor.fetchone():
            print(f"Trade with ID {trade_id} not found")
            conn.close()
            return False
        
        # Ensure consistent timestamp format
        if 'exit_time' in update_data and isinstance(update_data.get('exit_time'), datetime.datetime):
            update_data['exit_time'] = update_data['exit_time'].strftime("%Y-%m-%d %H:%M:%S")
        
        # Format numeric values
        if 'exit_price' in update_data:
            update_data['exit_price'] = float(update_data['exit_price'])
        if 'pnl' in update_data:
            update_data['pnl'] = float(update_data['pnl'])
        
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
    except Exception as e:
        print(f"Error updating trade: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.close()
        return False

def delete_trade_from_db(trade_id):
    """Remove a trade by ID."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting trade: {e}")
        if conn:
            conn.close()
        return False

def get_user_from_db(username):
    """Look up a user by username."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        conn.close()
        
        if user:
            return dict(user)
        
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        if conn:
            conn.close()
        return None

def add_user_to_db(user_data):
    """Create a new user record."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
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
        print("User already exists or constraint violation")
        if conn:
            conn.close()
        return None
    except Exception as e:
        print(f"Error adding user: {e}")
        if conn:
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
