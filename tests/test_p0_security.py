import hashlib
import hmac
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
import types
from unittest.mock import patch

if "jwt" not in sys.modules:
    sys.modules["jwt"] = types.SimpleNamespace(
        encode=lambda payload, key, algorithm=None: "token",
        decode=lambda token, key, algorithms=None: {"username": "mock", "role": "user"},
    )

if "bcrypt" not in sys.modules:
    class _FakeBcrypt:
        @staticmethod
        def gensalt():
            return b"salt"

        @staticmethod
        def hashpw(password, salt):
            return password + b":hashed"

        @staticmethod
        def checkpw(password, hashed):
            return hashed == password + b":hashed"

    sys.modules["bcrypt"] = _FakeBcrypt()

if "flask" not in sys.modules:
    class _FakeFlaskApp:
        def __init__(self, name):
            self.name = name

        def route(self, *args, **kwargs):
            def _decorator(func):
                return func

            return _decorator

        def run(self, *args, **kwargs):
            return None

    class _FakeRequest:
        headers = {}
        data = b""

        @staticmethod
        def get_json(silent=True):
            return {}

    sys.modules["flask"] = types.SimpleNamespace(
        Flask=_FakeFlaskApp,
        jsonify=lambda payload: payload,
        request=_FakeRequest(),
    )

from dashboard.components import auth, database
from utils import finnhub_webhook


class TestP0Security(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "trading_bot.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _user_count(self):
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        conn.close()
        return count

    def _find_user(self, username):
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()
        return row

    def test_no_default_admin_without_bootstrap_env(self):
        with patch.object(database, "DB_PATH", self.db_path), patch.object(
            auth, "DB_PATH", str(self.db_path)
        ), patch.dict(os.environ, {}, clear=False):
            database.ensure_db_exists()
            self.assertEqual(self._user_count(), 0)
            self.assertIsNone(self._find_user("admin"))

    def test_bootstrap_admin_is_created_once(self):
        env = {
            "TRADING_BOT_BOOTSTRAP_ADMIN_USERNAME": "bootstrap_admin",
            "TRADING_BOT_BOOTSTRAP_ADMIN_PASSWORD": "StrongPass!123",
            "TRADING_BOT_BOOTSTRAP_ADMIN_EMAIL": "bootstrap@example.com",
        }
        with patch.object(database, "DB_PATH", self.db_path), patch.object(
            auth, "DB_PATH", str(self.db_path)
        ), patch.dict(os.environ, env, clear=False):
            database.ensure_db_exists()
            database.ensure_db_exists()
            self.assertEqual(self._user_count(), 1)
            self.assertIsNotNone(self._find_user("bootstrap_admin"))

    def test_admin_admin123_is_rejected_when_not_bootstrapped(self):
        with patch.object(database, "DB_PATH", self.db_path), patch.object(
            auth, "DB_PATH", str(self.db_path)
        ), patch.dict(os.environ, {}, clear=False):
            database.ensure_db_exists()
            user = auth.login("admin", "admin123")
            self.assertIsNone(user)

    def test_webhook_signature_fails_closed_when_secret_missing(self):
        with patch.object(finnhub_webhook, "FINNHUB_WEBHOOK_SECRET", ""):
            self.assertFalse(finnhub_webhook.verify_webhook_signature(b"{}", "sig"))

    def test_webhook_signature_validates_when_secret_present(self):
        payload = b'{"type":"news"}'
        secret = "testsecret"
        signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        with patch.object(finnhub_webhook, "FINNHUB_WEBHOOK_SECRET", secret):
            self.assertTrue(
                finnhub_webhook.verify_webhook_signature(payload, signature)
            )

    def test_create_token_requires_jwt_secret(self):
        with patch.object(auth, "SECRET_KEY", ""):
            with self.assertRaises(RuntimeError):
                auth.create_token({"username": "alice", "role": "user"})


if __name__ == "__main__":
    unittest.main()
