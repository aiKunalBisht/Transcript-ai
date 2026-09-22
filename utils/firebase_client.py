# utils/firebase_client.py
# Firebase Firestore client — replaces local SQLite user storage.
# Users persist across HuggingFace redeploys.
# Set FIREBASE_SERVICE_ACCOUNT env var to the full service account JSON string.

import os
import json
import asyncio

_db = None
_initialized = False


def get_db():
    global _db, _initialized
    if _initialized:
        return _db

    _initialized = True
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT", "").strip()

    if not service_account_json:
        print("[FIREBASE] FIREBASE_SERVICE_ACCOUNT not set — user storage disabled.",
              flush=True)
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred_dict = json.loads(service_account_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        print("[FIREBASE] Firestore connected.", flush=True)
        return _db

    except ImportError:
        print("[FIREBASE] firebase-admin not installed. "
              "Run: pip install firebase-admin", flush=True)
        return None
    except Exception as e:
        print(f"[FIREBASE] init failed: {e}", flush=True)
        return None


async def upsert_user_firebase(user_id: str, email: str, name: str) -> None:
    import datetime
    db = get_db()
    if not db:
        return

    now = datetime.datetime.utcnow().isoformat()

    def _write():
        try:
            doc_ref = db.collection("users").document(user_id)
            doc = doc_ref.get()
            if doc.exists:
                doc_ref.update({"last_seen": now, "name": name})
            else:
                doc_ref.set({
                    "user_id":          user_id,
                    "email":            email,
                    "name":             name,
                    "created_at":       now,
                    "last_seen":        now,
                    "total_analyses":   0,
                })
        except Exception as e:
            print(f"[FIREBASE] _write failed: {e}", flush=True)

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        print(f"[FIREBASE] upsert_user failed: {e}", flush=True)