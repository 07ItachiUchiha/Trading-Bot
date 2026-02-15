import streamlit as st
import datetime
import jwt
import bcrypt
import sqlite3
import os
from pathlib import Path

# Define DB_PATH properly
DB_PATH = os.path.join(Path(__file__).parent.parent, "data", "trading_bot.db")

# Secret key for JWT token
SECRET_KEY = os.environ.get("JWT_SECRET", "")

def login(username, password):
    """Authenticate a user"""
    # Ensure directory exists
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE,
        role TEXT NOT NULL DEFAULT 'user'
    )
    ''')
    
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    
    # Check password
    if user and bcrypt.checkpw(password.encode("utf-8"), user[2].encode("utf-8")):
        conn.close()
        return {"username": username, "role": user[4] if len(user) > 4 else "user"}
    
    conn.close()
    return None

def create_token(user):
    """Create JWT token for authenticated user"""
    if not SECRET_KEY:
        raise RuntimeError("JWT_SECRET is not configured")

    expiration = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1)
    payload = {
        "username": user["username"],
        "role": user["role"],
        "exp": expiration
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

def verify_token(token):
    """Verify JWT token"""
    if not SECRET_KEY:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except:
        return None

def update_login_timestamp(username):
    """Update the last login timestamp for a user"""
    conn = sqlite3.connect(DB_PATH)
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

def authenticate():
    """Handle user authentication in Streamlit"""
    if "user" not in st.session_state:
        # Display login form
        with st.sidebar:
            st.title("🔑 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                user = login(username, password)
                if user:
                    try:
                        token = create_token(user)
                        st.session_state.user = user
                        st.session_state.token = token
                        st.success("Login successful!")
                    except RuntimeError as e:
                        st.error(str(e))
                else:
                    st.error("Invalid username or password")
        
        st.warning("Please login to access the dashboard")
        st.stop()
