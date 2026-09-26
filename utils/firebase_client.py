# utils/firebase_client.py
# Firebase Firestore client — replaces local SQLite user storage.
# Users persist across HuggingFace redeploys.
# Set FIREBASE_SERVICE_ACCOUNT env var to the full service account JSON string.
#
# v3.3 changes:
#   - upsert_user_firebase now accepts password_hash (optional)
#   - password_hash only written on create or when explicitly passed
#   - Added get_user_by_email_firebase for login verification

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


async def upsert_user_firebase(user_id: str, email: str, name: str,
                               password_hash: str = "") -> None:
    """
    Create or update a user document in Firestore.
    password_hash is only written on create, or on update when explicitly passed.
    Google OAuth users will never have a password_hash — that's intentional.
    """
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
                update_data: dict = {"last_seen": now, "name": name}
                # Only overwrite password_hash if a new one is explicitly provided
                if password_hash:
                    update_data["password_hash"] = password_hash
                doc_ref.update(update_data)
            else:
                new_doc: dict = {
                    "user_id":        user_id,
                    "email":          email,
                    "name":           name,
                    "created_at":     now,
                    "last_seen":      now,
                    "total_analyses": 0,
                }
                if password_hash:
                    new_doc["password_hash"] = password_hash
                doc_ref.set(new_doc)
        except Exception as e:
            print(f"[FIREBASE] _write failed: {e}", flush=True)

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        print(f"[FIREBASE] upsert_user failed: {e}", flush=True)


async def get_user_by_email_firebase(email: str) -> dict | None:
    """
    Look up a user by email address.
    Returns the full user dict (including password_hash if set) or None.
    Used by the /login route to verify credentials.
    """
    db = get_db()
    if not db:
        return None

    def _query():
        try:
            docs = (
                db.collection("users")
                .where("email", "==", email)
                .limit(1)
                .stream()
            )
            for doc in docs:
                return doc.to_dict()
            return None
        except Exception as e:
            print(f"[FIREBASE] get_user_by_email query failed: {e}", flush=True)
            return None

    try:
        return await asyncio.to_thread(_query)
    except Exception as e:
        print(f"[FIREBASE] get_user_by_email_firebase failed: {e}", flush=True)
        return None