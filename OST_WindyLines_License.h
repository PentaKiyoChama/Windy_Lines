#pragma once
// ============================================================================
// OST_WindyLines_License.h
// Shared license verification interface — single source of truth for CPU + GPU
//
// All license logic (cache read, signature verify, offline grace, background
// refresh, rapid check) lives in OST_WindyLines_CPU.cpp.
// GPU render paths simply call these two functions.
// ============================================================================

/// Call once per render frame (debounced internally, ~200ms cadence).
/// Reads cache, verifies signature, triggers background API refresh if needed.
void RefreshLicenseAuthenticatedState(bool force);

/// Fast atomic read — returns the last known authentication state.
/// Always call RefreshLicenseAuthenticatedState() earlier in the same
/// render cycle so the value stays current.
bool IsLicenseAuthenticated();
