#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <url> [api_key|method] [method|json_body] [json_body]"
  echo "Example (GET):  $0 https://example.com/api/license/ping"
  echo "Example (POST): $0 https://example.com/api/license/verify POST '{\"license_key\":\"xxxx\"}'"
  echo "Example (with key): $0 https://example.com/api/license/verify YOUR_API_KEY POST '{\"license_key\":\"xxxx\"}'"
  exit 1
fi

URL="$1"
API_KEY=""
METHOD="GET"
JSON_BODY="{}"

to_upper() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
}

if [[ $# -ge 2 ]]; then
  ARG2_UPPER="$(to_upper "$2")"
  if [[ "$ARG2_UPPER" == "GET" || "$ARG2_UPPER" == "POST" || "$ARG2_UPPER" == "PUT" || "$ARG2_UPPER" == "PATCH" || "$ARG2_UPPER" == "DELETE" || "$ARG2_UPPER" == "HEAD" ]]; then
    METHOD="$ARG2_UPPER"
    JSON_BODY="${3:-$JSON_BODY}"
  else
    API_KEY="$2"
    if [[ $# -ge 3 ]]; then
      ARG3_UPPER="$(to_upper "$3")"
      if [[ "$ARG3_UPPER" == "GET" || "$ARG3_UPPER" == "POST" || "$ARG3_UPPER" == "PUT" || "$ARG3_UPPER" == "PATCH" || "$ARG3_UPPER" == "DELETE" || "$ARG3_UPPER" == "HEAD" ]]; then
        METHOD="$ARG3_UPPER"
      fi
    fi
    if [[ $# -ge 4 ]]; then
      JSON_BODY="$4"
    fi
  fi
fi

TIMEOUT=8

TMP_HEADERS="$(mktemp)"
TMP_BODY="$(mktemp)"
trap 'rm -f "$TMP_HEADERS" "$TMP_BODY"' EXIT

echo "[INFO] URL      : $URL"
echo "[INFO] Method   : $METHOD"
echo "[INFO] Timeout  : ${TIMEOUT}s"

CMD=(curl -sS
  --location
  --connect-timeout "$TIMEOUT"
  --max-time "$TIMEOUT"
  --write-out "%{http_code}"
  --output "$TMP_BODY"
  --dump-header "$TMP_HEADERS")

if [[ -n "$API_KEY" ]]; then
  CMD+=( -H "Authorization: Bearer ${API_KEY}" )
fi

if [[ "$METHOD" != "GET" ]]; then
  CMD+=( -X "$METHOD" -H "Content-Type: application/json" )
  if [[ "$METHOD" == "POST" || "$METHOD" == "PUT" || "$METHOD" == "PATCH" ]]; then
    CMD+=( --data "$JSON_BODY" )
  fi
fi

HTTP_CODE=$("${CMD[@]}" "$URL")

echo "[INFO] HTTP code: $HTTP_CODE"

echo "[INFO] Response headers (excerpt):"
grep -Ei "^(HTTP/|content-type:|server:|date:|x-|cf-|strict-transport-security:)" "$TMP_HEADERS" || true

echo "[INFO] Response body (first 300 chars):"
head -c 300 "$TMP_BODY" || true
echo

if [[ "$HTTP_CODE" == "200" || "$HTTP_CODE" == "204" ]]; then
  echo "[PASS] Connectivity OK (Go candidate)"
  exit 0
fi

if [[ "$HTTP_CODE" == "401" || "$HTTP_CODE" == "403" ]]; then
  echo "[PASS] Connectivity OK but auth failed (endpoint reachable)"
  exit 0
fi

echo "[FAIL] Connectivity NG (No-Go candidate)"
exit 2
