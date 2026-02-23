#pragma once
// ============================================================================
// __TPL_MATCH_NAME___License.h
// Shared license verification interface — single source of truth for CPU + GPU
//
// All license logic (cache read, signature verify, offline grace, background
// refresh, rapid check) lives in __TPL_MATCH_NAME___CPU.cpp.
// GPU render paths simply call these two functions.
//
// License cache is SHARED across all OshareTelop plugins:
//   Mac: ~/Library/Application Support/OshareTelop/license_cache_v1.txt
//   Win: %APPDATA%\OshareTelop\license_cache_v1.txt
//
// One activation from ANY plugin unlocks ALL plugins.
// ============================================================================

/// Call once per render frame (debounced internally, ~200ms cadence).
/// Reads cache, verifies signature, triggers background API refresh if needed.
void RefreshLicenseAuthenticatedState(bool force);

/// Fast atomic read — returns the last known authentication state.
/// Always call RefreshLicenseAuthenticatedState() earlier in the same
/// render cycle so the value stays current.
bool IsLicenseAuthenticated();
