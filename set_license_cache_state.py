#!/usr/bin/env python3
import argparse
import hashlib
import os
import platform
import tempfile
import time
import uuid
from pathlib import Path


def cache_path() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "OST" / "WindyLines" / "license_cache_v1.txt"
        userprofile = os.environ.get("USERPROFILE", "C:/")
        return Path(userprofile) / "AppData" / "Roaming" / "OST" / "WindyLines" / "license_cache_v1.txt"
    home = Path.home()
    return home / "Library" / "Application Support" / "OST" / "WindyLines" / "license_cache_v1.txt"


def machine_id_hash() -> str:
    node = platform.node() or "unknown-host"
    mac = f"{uuid.getnode():012x}"
    raw = f"{node}|{mac}|{platform.system()}|{platform.machine()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def mask_key(license_key: str) -> str:
    if len(license_key) <= 4:
        return "*" * len(license_key)
    return "*" * (len(license_key) - 4) + license_key[-4:]


def write_cache_file(path: Path, authorized: bool, reason: str, ttl_sec: int, license_key: str) -> None:
    now = int(time.time())
    expire = now + max(1, ttl_sec)

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"authorized={'true' if authorized else 'false'}",
            f"reason={reason}",
            f"validated_unix={now}",
            f"cache_expire_unix={expire}",
            f"license_key_masked={mask_key(license_key)}",
            f"machine_id_hash={machine_id_hash()}",
            "",
        ]
    )

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Force local license cache state for Premiere watermark verification")
    parser.add_argument("--state", required=True, choices=["authorized", "unauthorized"], help="Target local auth state")
    parser.add_argument("--ttl", type=int, default=3600, help="TTL seconds for forced state")
    parser.add_argument("--license-key", default="WL-LOCAL-TEST", help="Masked key source for cache file")
    args = parser.parse_args()

    authorized = args.state == "authorized"
    reason = "manual_override" if authorized else "manual_unauthorized"
    path = cache_path()

    write_cache_file(
        path=path,
        authorized=authorized,
        reason=reason,
        ttl_sec=max(1, args.ttl),
        license_key=args.license_key,
    )

    print(f"[INFO] state: {args.state}")
    print(f"[INFO] cache_file: {path}")
    print("[PASS] cache updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
