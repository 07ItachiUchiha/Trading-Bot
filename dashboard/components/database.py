import sqlite3
import pandas as pd
import datetime
import json
import os
import sys
import shutil
from pathlib import Path
import bcrypt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Database paths
DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "prediction_platform.db"


def _bootstrap_database_file():
    """Move/copy legacy DB into the new prediction DB location when needed."""
    target_dir = DB_PATH.parent
    if not target_dir.exists():
        os.makedirs(target_dir)

    legacy_db_path = target_dir / "trading_bot.db"

    if not DB_PATH.exists() and legacy_db_path.exists():
        shutil.copy2(legacy_db_path, DB_PATH)


def _legacy_trades_table_exists(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
    )
    return cursor.fetchone() is not None


def _migrate_legacy_trades_to_prediction_events(cursor):
    """Migrate legacy `trades` rows into `prediction_events` once."""
    if not _legacy_trades_table_exists(cursor):
        return

    cursor.execute("SELECT COUNT(*) FROM prediction_events")
    prediction_events_count = cursor.fetchone()[0]
    if prediction_events_count > 0:
        return

    cursor.execute(
        """
        INSERT INTO prediction_events (
            id, symbol, predicted_direction, baseline_price, resolved_price,
            reference_threshold, targets, exposure_size, signal_time,
            resolution_time, outcome_delta, outcome_status
        )
        SELECT
            id, symbol, direction, entry_price, exit_price,
            stop_loss, targets, size, entry_time,
            exit_time, pnl, status
        FROM trades
        """
    )


def _bootstrap_admin_if_configured(cursor):
    """Create a bootstrap admin only when explicit env vars are provided."""
    username = os.environ.get(
        "TRADING_BOT_BOOTSTRAP_ADMIN_USERNAME",
        os.environ.get("PREDICTION_PLATFORM_BOOTSTRAP_ADMIN_USERNAME", ""),
    ).strip()
    password = os.environ.get(
        "TRADING_BOT_BOOTSTRAP_ADMIN_PASSWORD",
        os.environ.get("PREDICTION_PLATFORM_BOOTSTRAP_ADMIN_PASSWORD", ""),
    ).strip()
    email = os.environ.get(
        "TRADING_BOT_BOOTSTRAP_ADMIN_EMAIL",
        os.environ.get("PREDICTION_PLATFORM_BOOTSTRAP_ADMIN_EMAIL", "admin@localhost"),
    ).strip()

    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    if user_count > 0:
        return

    if not username or not password:
        print(
            "No users exist yet. Set PREDICTION_PLATFORM_BOOTSTRAP_ADMIN_USERNAME and "
            "PREDICTION_PLATFORM_BOOTSTRAP_ADMIN_PASSWORD to create the first admin account."
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
    _bootstrap_database_file()
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Create prediction events table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        predicted_direction TEXT NOT NULL,
        baseline_price REAL NOT NULL,
        resolved_price REAL,
        reference_threshold REAL,
        targets TEXT,
        exposure_size REAL NOT NULL,
        signal_time TIMESTAMP NOT NULL,
        resolution_time TIMESTAMP,
        outcome_delta REAL,
        outcome_status TEXT NOT NULL
    )
    ''')

    _migrate_legacy_trades_to_prediction_events(cursor)
    
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

def get_prediction_events_from_db(symbol=None, outcome_status=None, limit=100):
    """Fetch prediction events with optional symbol/status filter."""
    conn = None
    try:
        conn = get_db_connection()
        
        query = "SELECT * FROM prediction_events"
        params = []
        
        if symbol or outcome_status:
            query += " WHERE"
            
            if symbol:
                query += " symbol = ?"
                params.append(symbol)
                
            if symbol and outcome_status:
                query += " AND"
                
            if outcome_status:
                query += " outcome_status = ?"
                params.append(outcome_status)
        
        query += " ORDER BY signal_time DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        df = pd.read_sql(query, conn, params=params)
        
        # Convert timestamp strings to datetime
        if not df.empty:
            df['signal_time'] = pd.to_datetime(df['signal_time'], format='mixed', errors='coerce')
            df['resolution_time'] = pd.to_datetime(df['resolution_time'], format='mixed', errors='coerce')
            # Convert targets from JSON string to list
            df['targets'] = df['targets'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x else []
            )
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"Error fetching prediction events from database: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.close()
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'id', 'symbol', 'predicted_direction', 'baseline_price', 'resolved_price',
            'reference_threshold', 'targets', 'exposure_size', 'outcome_delta',
            'signal_time', 'resolution_time', 'outcome_status'
        ])


def get_trades_from_db(symbol=None, status=None, limit=100):
    """Backward-compatible alias for legacy callers."""
    return get_prediction_events_from_db(symbol=symbol, outcome_status=status, limit=limit)

def add_prediction_event_to_db(event_data):
    """Insert a new prediction event record."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Print debug info
        print(f"Adding prediction event: {event_data}")
        
        # Format dates properly
        if isinstance(event_data.get('signal_time'), datetime.datetime):
            signal_time = event_data['signal_time'].strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(event_data.get('entry_time'), datetime.datetime):
            signal_time = event_data['entry_time'].strftime("%Y-%m-%d %H:%M:%S")
        else:
            signal_time = event_data.get('signal_time') or event_data.get('entry_time')
            
        # Convert targets list to a string
        targets_str = json.dumps(event_data.get('targets', []))
        
        # Validate required fields
        if not event_data.get('symbol') or not signal_time:
            print("Error: Missing required fields for prediction event")
            return False
        
        cursor.execute(
            """
            INSERT INTO prediction_events (
                symbol, predicted_direction, baseline_price, resolved_price, 
                reference_threshold, targets, exposure_size, outcome_delta, 
                signal_time, resolution_time, outcome_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_data.get('symbol', 'UNKNOWN'),
                event_data.get('predicted_direction', event_data.get('direction', 'neutral')),
                float(event_data.get('baseline_price', event_data.get('entry_price', 0.0))),
                None,  # resolved_price starts as NULL
                float(event_data.get('reference_threshold', event_data.get('stop_loss', 0.0))),
                targets_str,
                float(event_data.get('exposure_size', event_data.get('size', 0.0))),
                0.0,   # outcome_delta starts at 0
                signal_time,
                None,  # resolution_time starts as NULL
                event_data.get('outcome_status', event_data.get('status', 'open'))
            )
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding prediction event to database: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.close()
        return False


def add_trade_to_db(trade_data):
    """Backward-compatible alias for legacy callers."""
    return add_prediction_event_to_db(trade_data)

def update_prediction_event_in_db(event_id, update_data):
    """Update fields on an existing prediction event."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if event exists
        cursor.execute("SELECT id FROM prediction_events WHERE id = ?", (event_id,))
        if not cursor.fetchone():
            print(f"Prediction event with ID {event_id} not found")
            conn.close()
            return False
        
        # Ensure consistent timestamp format
        if 'resolution_time' in update_data and isinstance(update_data.get('resolution_time'), datetime.datetime):
            update_data['resolution_time'] = update_data['resolution_time'].strftime("%Y-%m-%d %H:%M:%S")
        if 'exit_time' in update_data and isinstance(update_data.get('exit_time'), datetime.datetime):
            update_data['resolution_time'] = update_data['exit_time'].strftime("%Y-%m-%d %H:%M:%S")
            update_data.pop('exit_time', None)
        
        # Format numeric values
        if 'resolved_price' in update_data:
            update_data['resolved_price'] = float(update_data['resolved_price'])
        if 'exit_price' in update_data:
            update_data['resolved_price'] = float(update_data['exit_price'])
            update_data.pop('exit_price', None)
        if 'outcome_delta' in update_data:
            update_data['outcome_delta'] = float(update_data['outcome_delta'])
        if 'pnl' in update_data:
            update_data['outcome_delta'] = float(update_data['pnl'])
            update_data.pop('pnl', None)
        if 'status' in update_data:
            update_data['outcome_status'] = update_data['status']
            update_data.pop('status', None)
        if 'direction' in update_data:
            update_data['predicted_direction'] = update_data['direction']
            update_data.pop('direction', None)
        
        set_clauses = []
        params = []
        
        for key, value in update_data.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
        
        params.append(event_id)
        
        query = f"UPDATE prediction_events SET {', '.join(set_clauses)} WHERE id = ?"
        cursor.execute(query, params)
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating prediction event: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.close()
        return False

def update_trade_in_db(trade_id, update_data):
    """Backward-compatible alias for legacy callers."""
    return update_prediction_event_in_db(trade_id, update_data)


def delete_prediction_event_from_db(event_id):
    """Remove a prediction event by ID."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM prediction_events WHERE id = ?", (event_id,))
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting prediction event: {e}")
        if conn:
            conn.close()
        return False


def delete_trade_from_db(trade_id):
    """Backward-compatible alias for legacy callers."""
    return delete_prediction_event_from_db(trade_id)

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

def get_prediction_stats():
    """Get overall prediction outcome statistics."""
    conn = get_db_connection()
    
    # Get total prediction events
    total_events = pd.read_sql("SELECT COUNT(*) as count FROM prediction_events", conn).iloc[0]['count']
    
    # Get closed outcome stats
    closed_events_df = pd.read_sql(
        "SELECT * FROM prediction_events WHERE outcome_status = 'closed'",
        conn
    )
    
    stats = {
        'total_events': total_events,
        'closed_events': len(closed_events_df),
        'open_events': total_events - len(closed_events_df)
    }
    
    if not closed_events_df.empty:
        positive_events = closed_events_df[closed_events_df['outcome_delta'] > 0]
        negative_events = closed_events_df[closed_events_df['outcome_delta'] <= 0]
        
        stats.update({
            'net_outcome': closed_events_df['outcome_delta'].sum(),
            'positive_events': len(positive_events),
            'negative_events': len(negative_events),
            'positive_rate': len(positive_events) / len(closed_events_df) * 100 if len(closed_events_df) > 0 else 0,
            'avg_positive_outcome': positive_events['outcome_delta'].mean() if len(positive_events) > 0 else 0,
            'avg_negative_outcome': negative_events['outcome_delta'].mean() if len(negative_events) > 0 else 0
        })
    
    conn.close()
    return stats


def get_trade_stats():
    """Backward-compatible alias for legacy callers."""
    return get_prediction_stats()
