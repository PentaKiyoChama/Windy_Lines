#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from typing import Dict, Optional, Tuple
from pathlib import Path

DEFAULT_ENDPOINT = "https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test"
DEFAULT_TIMEOUT = 8
DEFAULT_FAIL_TTL = 600
DEFAULT_OK_TTL = 600
PRODUCT_NAME = "OST_WindyLines"
PLUGIN_VERSION = "1.0.0"
CACHE_SIGNATURE_SALT = "OST_WL_2026_SALT_K9x3"


def djb2_hash(s: str) -> int:
    """DJB2 hash matching C++ SimpleHash32."""
    h = 5381
    for c in s.encode("utf-8"):
        h = ((h << 5) + h + c) & 0xFFFFFFFF
    return h


def compute_cache_signature(authorized_str: str, validated_unix_str: str, machine_id_hash_val: str) -> str:
    """Compute cache signature matching C++ ComputeCacheSignature."""
    payload = f"{authorized_str}|{validated_unix_str}|{machine_id_hash_val}|{CACHE_SIGNATURE_SALT}"
    h1 = djb2_hash(payload)
    pass2 = f"{payload}|{h1}"
    h2 = djb2_hash(pass2)
    return f"{h1:08x}{h2:08x}"


def cache_path() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
                return Path(appdata) / "OshareTelop" / "license_cache_v1.txt"
        userprofile = os.environ.get("USERPROFILE", "C:/")
            return Path(userprofile) / "AppData" / "Roaming" / "OshareTelop" / "license_cache_v1.txt"
    home = Path.home()
        return home / "Library" / "Application Support" / "OshareTelop" / "license_cache_v1.txt"


def machine_id_hash() -> str:
    """Compute machine ID matching C++ GetMachineIdHash (DJB2-based)."""
    import socket
    import struct
    hostname = socket.gethostname()
    ptr_size = struct.calcsize("P")
    raw = f"{hostname}|mac|{ptr_size}"
    h1 = djb2_hash(raw)
    h2 = djb2_hash(raw + "_salt_ost")
    return f"{h1:08x}{h2:08x}"


def post_verify(endpoint: str, license_key: Optional[str], api_key: Optional[str], timeout_sec: int) -> Tuple[int, Dict, str]:
    payload = {
        "action": "verify",
        "machine_id": machine_id_hash(),
        "product": PRODUCT_NAME,
        "plugin_version": PLUGIN_VERSION,
        "platform": "win" if os.name == "nt" else "mac",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if license_key:
        payload["license_key"] = license_key

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, method="POST", data=data)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = {}
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {}
            return int(resp.status), parsed, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        parsed = {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        return int(e.code), parsed, body


def parse_authorized(status_code: int, body_json: Dict) -> Tuple[bool, str, int]:
    if status_code != 200:
        return False, f"http_{status_code}", DEFAULT_FAIL_TTL

    if "authorized" in body_json:
        authorized = bool(body_json.get("authorized", False))
        reason = str(body_json.get("reason", "ok" if authorized else "denied"))
        ttl = int(body_json.get("cache_ttl_sec", DEFAULT_OK_TTL if authorized else DEFAULT_FAIL_TTL))
        return authorized, reason, ttl

    response_obj = body_json.get("response")
    if isinstance(response_obj, dict):
        nested_authorized = response_obj.get("authorized")
        if nested_authorized is not None:
            if isinstance(nested_authorized, bool):
                authorized = nested_authorized
            else:
                normalized = str(nested_authorized).strip().lower()
                authorized = normalized in ("1", "true", "authorized", "ok", "success")
            reason = str(body_json.get("reason", "ok" if authorized else "denied"))
            ttl = int(body_json.get("cache_ttl_sec", DEFAULT_OK_TTL if authorized else DEFAULT_FAIL_TTL))
            return authorized, reason, ttl

        parameter_1 = response_obj.get("Parameter 1")
        if parameter_1 is not None:
            normalized = str(parameter_1).strip().lower()
            if normalized in ("authorized", "true", "ok", "success"):
                return True, "ok", DEFAULT_OK_TTL
            if normalized in ("unauthorized", "false", "denied", "invalid"):
                return False, "denied", DEFAULT_FAIL_TTL

    status = str(body_json.get("status", "")).lower()
    if status == "success":
        return False, "missing_authorized_field", DEFAULT_FAIL_TTL

    return False, "invalid_json_response", DEFAULT_FAIL_TTL


def mask_key(license_key: str) -> str:
    if not license_key:
        return ""
    if len(license_key) <= 4:
        return "*" * len(license_key)
    return "*" * (len(license_key) - 4) + license_key[-4:]


def write_cache_file(path: Path, authorized: bool, reason: str, ttl_sec: int, license_key: str, mid_hash: str) -> None:
    now = int(time.time())
    expire = now + max(1, ttl_sec)
    auth_str = "true" if authorized else "false"
    now_str = str(now)
    sig = compute_cache_signature(auth_str, now_str, mid_hash)

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"authorized={auth_str}",
            f"reason={reason}",
            f"validated_unix={now}",
            f"cache_expire_unix={expire}",
            f"license_key_masked={mask_key(license_key)}",
            f"machine_id_hash={mid_hash}",
            f"cache_signature={sig}",
            "",
        ]
    )

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify license via API and write local auth cache for OST_WindyLines")
    parser.add_argument("--license-key", default="", help="License key string (optional)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Verification endpoint (POST)")
    parser.add_argument("--api-key", default="", help="Optional bearer token")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds")
    args = parser.parse_args()

    print(f"[INFO] endpoint: {args.endpoint}")
    print(f"[INFO] platform: {'win' if os.name == 'nt' else 'mac'}")

    status_code, body_json, raw_body = post_verify(
        endpoint=args.endpoint,
        license_key=(args.license_key or "").strip() or None,
        api_key=args.api_key or None,
        timeout_sec=max(1, args.timeout),
    )

    authorized, reason, ttl_sec = parse_authorized(status_code, body_json)
    mid_hash = machine_id_hash()
    path = cache_path()
    write_cache_file(
        path=path,
        authorized=authorized,
        reason=reason,
        ttl_sec=ttl_sec,
        license_key=(args.license_key or "").strip(),
        mid_hash=mid_hash,
    )

    print(f"[INFO] http_status: {status_code}")
    print(f"[INFO] authorized: {str(authorized).lower()}")
    print(f"[INFO] reason: {reason}")
    print(f"[INFO] cache_ttl_sec: {ttl_sec}")
    print(f"[INFO] cache_file: {path}")

    if not body_json:
        snippet = raw_body[:300].replace("\n", " ")
        print(f"[INFO] response_preview: {snippet}")

    if authorized:
        print("[PASS] license cache updated (authenticated)")
        return 0

    print("[WARN] license cache updated (not authenticated)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
