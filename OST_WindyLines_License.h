#pragma once
// ============================================================================
// OST_WindyLines_License.h
// Shared license verification interface — single source of truth for CPU + GPU
//
// All license logic (cache read, signature verify, offline grace, background
// refresh, project token verify) now lives in the shared SDK
// (PremierePro_GPU_Plugin_DevGuide/sdk/license/OSTLicense*). この header は
// 2026-06-05 に確定した「GPU でも GetParam(PROJECT_UUID) で UUID 文字列を読める」
// 仕様を使い、CPU 経路 (UPDATE_PARAMS_UI) で per-UUID 認可を SDK キャッシュに書き、
// GPU はそれを参照する per-project パターンを提供する (SBV2 / TileFlipPro と同仕様)。
//
// License cache (device-local 認証) は SHARED across all OshareTelop plugins:
//   Mac: ~/Library/Application Support/OshareTelop/license_cache_v1.txt
//   Win: %APPDATA%\OshareTelop\license_cache_v1.txt
// One activation from ANY plugin unlocks ALL plugins.
//
// Project token (.prproj 埋め込み永久ライセンス) は per-project な ARB に格納。
// device-local 認証でない別マシンでも、.prproj 内の token が valid なら認可される。
// ============================================================================

/// Call once per render frame (debounced internally, ~200ms cadence).
/// SDK 提供 (OSTLicense.cpp)。
void RefreshLicenseAuthenticatedState(bool force);

/// Fast atomic read — returns the last known device-local authentication state.
/// SDK 提供 (OSTLicense.cpp)。
bool IsLicenseAuthenticated();

// ---------------------------------------------------------------------------
// Per-UUID authorization cache (SBV2 / TileFlipPro と同仕様, 2026-06-05 確定モデル)
// 実装本体は OST_WindyLines_CPU.cpp に置き、GPU TU は header 経由で呼ぶ。
// ---------------------------------------------------------------------------
struct PF_ParamDef_;  // 前方宣言 (AE_Effect.h 不要)

/// CPU UPDATE_PARAMS_UI で呼ぶ。ARB から UUID/token を読出して Path 1/2/3 で
/// 認可判定し、SDK の per-UUID キャッシュ + per-project サイドカーに反映する。
void OST_WindyLines_UpdateAuthCacheFromParams(struct PF_ParamDef_* params[]);

/// CPU SmartRender / Render 用 fallback (UUID context 無し → Path 1 のみで判定)。
bool OST_WindyLines_IsAuthorizedCached();

/// GPU/CPU 共通の per-UUID 判定。GPU は GetParam(PROJECT_UUID) で取った UUID
/// 文字列を渡す。Path 1 active なら uuid 不要で true。未登録 UUID は true (flicker 防止)。
bool OST_WindyLines_IsAuthorizedForUUID(const char* uuid);
