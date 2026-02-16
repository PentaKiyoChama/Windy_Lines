#!/usr/bin/env python3
import argparse
import hashlib
import os
import platform
import tempfile
import time
import uuid
from pathlib import Path

CACHE_SIGNATURE_SALT = "OST_WL_2026_SALT_K9x3"


def djb2_hash(s: str) -> int:
    """DJB2 hash matching C++ SimpleHash32."""
    h = 5381
    for c in s.encode("utf-8"):
        h = ((h << 5) + h + c) & 0xFFFFFFFF
    return h


def compute_cache_signature(authorized_str: str, validated_unix_str: str, machine_id_hash: str, expire_unix_str: str) -> str:
    """Compute cache signature matching C++ ComputeCacheSignature."""
    payload = f"{authorized_str}|{validated_unix_str}|{machine_id_hash}|{expire_unix_str}|{CACHE_SIGNATURE_SALT}"
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


def mask_key(license_key: str) -> str:
    if len(license_key) <= 4:
        return "*" * len(license_key)
    return "*" * (len(license_key) - 4) + license_key[-4:]


def write_cache_file(path: Path, authorized: bool, reason: str, ttl_sec: int, license_key: str, validated_ago: int = 0) -> None:
    now = int(time.time())
    validated_time = now - validated_ago
    expire = validated_time + max(1, ttl_sec)
    mid = machine_id_hash()
    auth_str = "true" if authorized else "false"
    now_str = str(validated_time)
    expire_str = str(expire)
    sig = compute_cache_signature(auth_str, now_str, mid, expire_str)

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"authorized={auth_str}",
            f"reason={reason}",
            f"validated_unix={validated_time}",
            f"cache_expire_unix={expire}",
            f"license_key_masked={mask_key(license_key)}",
            f"machine_id_hash={mid}",
            f"cache_signature={sig}",
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
    parser.add_argument("--validated-ago", type=int, default=0, help="Set validated_unix to N seconds in the past (e.g. 7200 = 2h ago)")
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
        validated_ago=max(0, args.validated_ago),
    )

    print(f"[INFO] state: {args.state}")
    print(f"[INFO] cache_file: {path}")
    print("[PASS] cache updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
