/*******************************************************************/
/*                                                                 */
/*  OST_WindyLines - Particle Line Effect Plugin                   */
/*  for Adobe Premiere Pro                                         */
/*                                                                 */
/*  Copyright (c) 2026 Kiyoto Nakamura. All rights reserved.       */
/*                                                                 */
/*  This plugin was developed using the Adobe Premiere Pro SDK.    */
/*  Portions based on SDK sample code:                             */
/*    Copyright 2012 Adobe Systems Incorporated.                   */
/*    Used in accordance with the Adobe Developer SDK License.     */
/*                                                                 */
/*  This software is not affiliated with or endorsed by Adobe.     */
/*                                                                 */
/*******************************************************************/


#include "OST_WindyLines.h"
#include "OST_WindyLines_ParamNames.h"
#include "OST_WindyLines_Version.h"
#include "OST_WindyLines_WatermarkMask.h"
#include "OST_WindyLines_Common.h"
#include "AE_EffectSuites.h"
#include "PrSDKAESupport.h"
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <ctime>
#include <unordered_map>
#include <vector>
#include <cstdarg>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#include <thread>
#else
#include <pwd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <thread>
#endif

// Debug logging function is now in OST_WindyLines.h

// Debounce for preset button double-fire issue
static std::atomic<uint32_t> sLastPresetClickTime{ 0 };
static const uint32_t kPresetDebounceMs = 200;
static std::atomic<uint32_t> sLastLicenseRefreshTimeMs{ 0 };
static const uint32_t kLicenseRefreshIntervalMs = 200;

static uint32_t GetCurrentTimeMs()
{
#ifdef _WIN32
	return GetTickCount();
#else
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return static_cast<uint32_t>(tv.tv_sec * 1000 + tv.tv_usec / 1000);
#endif
}

static std::string TrimAscii(const std::string& value)
{
	size_t begin = 0;
	size_t end = value.size();
	while (begin < end && (value[begin] == ' ' || value[begin] == '\t' || value[begin] == '\r' || value[begin] == '\n'))
	{
		++begin;
	}
	while (end > begin && (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\r' || value[end - 1] == '\n'))
	{
		--end;
	}
	return value.substr(begin, end - begin);
}

static bool ParseBoolLike(const std::string& value, bool* outValue)
{
	if (!outValue)
	{
		return false;
	}
	if (value == "1" || value == "true" || value == "TRUE" || value == "True")
	{
		*outValue = true;
		return true;
	}
	if (value == "0" || value == "false" || value == "FALSE" || value == "False")
	{
		*outValue = false;
		return true;
	}
	return false;
}

static std::vector<std::string> GetLicenseCachePaths()
{
#ifdef _WIN32
	std::vector<std::string> paths;
	char appData[MAX_PATH] = { 0 };
	DWORD appDataLen = GetEnvironmentVariableA("APPDATA", appData, MAX_PATH);
	if (appDataLen > 0 && appDataLen < MAX_PATH)
	{
		paths.push_back(std::string(appData) + "\\OshareTelop\\license_cache_v1.txt");
	}
	const char* userProfile = std::getenv("USERPROFILE");
	if (userProfile && *userProfile)
	{
		paths.push_back(std::string(userProfile) + "\\AppData\\Roaming\\OshareTelop\\license_cache_v1.txt");
	}
	paths.push_back(std::string("C:\\Temp\\ost_windylines_license_cache_v1.txt"));
	return paths;
#else
	std::vector<std::string> paths;
	const char* home = std::getenv("HOME");
	if (home && *home)
	{
		paths.push_back(std::string(home) + "/Library/Application Support/OshareTelop/license_cache_v1.txt");
	}

	struct passwd* pw = getpwuid(getuid());
	if (pw && pw->pw_dir && *pw->pw_dir)
	{
		const std::string pwHomePath = std::string(pw->pw_dir) + "/Library/Application Support/OshareTelop/license_cache_v1.txt";
		bool exists = false;
		for (const auto& path : paths)
		{
			if (path == pwHomePath)
			{
				exists = true;
				break;
			}
		}
		if (!exists)
		{
			paths.push_back(pwHomePath);
		}
	}

	paths.push_back(std::string("/tmp/ost_windylines_license_cache_v1.txt"));
	return paths;
#endif
}

// Forward declarations (defined later in file)
static std::string GetMachineIdHash();
static std::string ComputeCacheSignature(const std::string& authorizedStr,
                                          const std::string& validatedUnixStr,
                                          const std::string& machineIdHash,
                                          const std::string& expireUnixStr);

// Offline grace: allow authorized state for 1h after last successful verification
// even when cache TTL expires (e.g. network temporarily unavailable)
static const int kOfflineGracePeriodSec = 3600;

static bool LoadLicenseAuthenticatedFromCache(bool* outAuthenticated, bool* outExpired = nullptr)
{
	if (!outAuthenticated)
	{
		return false;
	}
	if (outExpired)
	{
		*outExpired = false;
	}

	const std::vector<std::string> cachePaths = GetLicenseCachePaths();
	std::string cachePath;
	FILE* file = nullptr;
	for (const auto& candidate : cachePaths)
	{
		file = std::fopen(candidate.c_str(), "rb");
		if (file)
		{
			cachePath = candidate;
			break;
		}
	}

	if (!file)
	{
		if (!cachePaths.empty())
		{
			DebugLog("[License] cache file not found. first_path=%s", cachePaths.front().c_str());
		}
		else
		{
			DebugLog("[License] cache file not found. no candidate paths");
		}
		return false;
	}

	bool hasAuthorized = false;
	bool authorized = false;
	bool hasExpire = false;
	long long expireUnix = 0;
	std::string cachedMachineIdHash;
	std::string authorizedRaw;
	std::string validatedUnixRaw;
	std::string expireUnixRaw;
	std::string cachedSignature;

	char line[512];
	while (std::fgets(line, static_cast<int>(sizeof(line)), file) != nullptr)
	{
		std::string rawLine(line);
		const size_t sep = rawLine.find('=');
		if (sep == std::string::npos)
		{
			continue;
		}
		const std::string key = TrimAscii(rawLine.substr(0, sep));
		const std::string value = TrimAscii(rawLine.substr(sep + 1));

		if (key == "authorized")
		{
			authorizedRaw = value;
			bool parsed = false;
			if (ParseBoolLike(value, &parsed))
			{
				authorized = parsed;
				hasAuthorized = true;
			}
		}
		else if (key == "cache_expire_unix")
		{
			expireUnixRaw = value;
			char* endPtr = nullptr;
			const long long parsed = std::strtoll(value.c_str(), &endPtr, 10);
			if (endPtr != value.c_str())
			{
				expireUnix = parsed;
				hasExpire = true;
			}
		}
		else if (key == "validated_unix")
		{
			validatedUnixRaw = value;
		}
		else if (key == "machine_id_hash")
		{
			cachedMachineIdHash = value;
		}
		else if (key == "cache_signature")
		{
			cachedSignature = value;
		}
	}

	std::fclose(file);

	if (!hasAuthorized)
	{
		DebugLog("[License] cache invalid: authorized missing");
		return false;
	}

	if (!hasExpire)
	{
		DebugLog("[License] cache invalid: cache_expire_unix missing");
		return false;
	}

	// --- Machine ID verification (anti-copy) ---
	if (!cachedMachineIdHash.empty())
	{
		const std::string localMid = GetMachineIdHash();
		if (cachedMachineIdHash != localMid)
		{
			DebugLog("[License] machine_id mismatch: cached=%s local=%s",
				cachedMachineIdHash.c_str(), localMid.c_str());
			return false;
		}
	}

	// --- Signature verification (anti-tampering) ---
	if (cachedSignature.empty())
	{
		// No signature = legacy or tampered cache → treat as unauthorized
		DebugLog("[License] cache missing signature — treating as unauthorized");
		*outAuthenticated = false;
		return true;
	}

	const std::string expectedSig = ComputeCacheSignature(
		authorizedRaw, validatedUnixRaw, cachedMachineIdHash, expireUnixRaw);
	if (cachedSignature != expectedSig)
	{
		DebugLog("[License] cache signature mismatch (tampering detected)");
		*outAuthenticated = false;
		return true;
	}

	const long long nowUnix = static_cast<long long>(std::time(nullptr));

	// --- TTL check with offline grace ---
	if (expireUnix <= nowUnix)
	{
		// Cache expired — check offline grace period (signature was valid)
		long long validatedUnix = 0;
		if (!validatedUnixRaw.empty())
		{
			char* endPtr = nullptr;
			validatedUnix = std::strtoll(validatedUnixRaw.c_str(), &endPtr, 10);
		}
		const long long graceCutoff = validatedUnix + kOfflineGracePeriodSec;
		if (authorized && graceCutoff > nowUnix)
		{
			// Within 1h grace: keep authorized state, but caller should try refresh
			DebugLog("[License] cache expired but within offline grace (grace_remain=%llds)",
				graceCutoff - nowUnix);
			*outAuthenticated = true;
			if (outExpired)
			{
				*outExpired = true;  // Signal caller to trigger background refresh
			}
			return true;
		}
		DebugLog("[License] cache expired beyond grace: now=%lld expire=%lld", nowUnix, expireUnix);
		return false;
	}

	*outAuthenticated = authorized;
	DebugLog("[License] cache loaded: authenticated=%s sig=OK path=%s",
		authorized ? "true" : "false", cachePath.c_str());
	return true;
}

static std::atomic<bool> sLicenseAuthenticated{ false };

// === Machine ID hash (simple, no OpenSSL dependency) ===
static uint32_t SimpleHash32(const char* str)
{
	uint32_t hash = 5381;
	while (*str)
	{
		hash = ((hash << 5) + hash) + static_cast<unsigned char>(*str);
		++str;
	}
	return hash;
}

static std::string GetMachineIdHash()
{
	char hostname[256] = {0};
#ifdef _WIN32
	DWORD size = sizeof(hostname);
	GetComputerNameA(hostname, &size);
	std::string raw = std::string(hostname) + "|win|" + std::to_string(sizeof(void*));
#else
	gethostname(hostname, sizeof(hostname) - 1);
	std::string raw = std::string(hostname) + "|mac|" + std::to_string(sizeof(void*));
#endif
	uint32_t h1 = SimpleHash32(raw.c_str());
	uint32_t h2 = SimpleHash32((raw + "_salt_ost").c_str());
	char buf[20];
	std::snprintf(buf, sizeof(buf), "%08x%08x", h1, h2);
	return std::string(buf);
}

// === Activation token generation & persistence ===
static std::string GenerateActivationToken()
{
	uint32_t parts[4];
	parts[0] = static_cast<uint32_t>(std::time(nullptr));
	parts[1] = GetCurrentTimeMs();
#ifdef _WIN32
	parts[2] = GetCurrentProcessId();
	parts[3] = GetTickCount();
#else
	parts[2] = static_cast<uint32_t>(getpid());
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	parts[3] = static_cast<uint32_t>(tv.tv_usec);
#endif
	char buf[40];
	std::snprintf(buf, sizeof(buf), "%08x%08x%08x%08x",
		SimpleHash32(reinterpret_cast<const char*>(&parts[0])),
		SimpleHash32(reinterpret_cast<const char*>(&parts[1])),
		parts[2], parts[3]);
	return std::string(buf);
}

static std::string GetActivationTokenPath()
{
	const std::vector<std::string> paths = GetLicenseCachePaths();
	if (paths.empty()) return "";
	std::string dir = paths.front();
	const size_t lastSep = dir.find_last_of("/\\");
	if (lastSep != std::string::npos)
		dir = dir.substr(0, lastSep);
#ifdef _WIN32
	return dir + "\\activation_token.txt";
#else
	return dir + "/activation_token.txt";
#endif
}

static std::string LoadOrCreateActivationToken()
{
	const std::string path = GetActivationTokenPath();
	if (path.empty()) return GenerateActivationToken();

	FILE* f = std::fopen(path.c_str(), "rb");
	if (f)
	{
		char buf[128] = {0};
		if (std::fgets(buf, sizeof(buf), f))
		{
			std::fclose(f);
			std::string token = TrimAscii(std::string(buf));
			if (!token.empty()) return token;
		}
		else
		{
			std::fclose(f);
		}
	}

	std::string token = GenerateActivationToken();
#ifdef _WIN32
	const size_t lastSep = path.find_last_of('\\');
	if (lastSep != std::string::npos)
	{
		CreateDirectoryA(path.substr(0, lastSep).c_str(), nullptr);
	}
#else
	const size_t lastSep = path.find_last_of('/');
	if (lastSep != std::string::npos)
	{
		std::string mkdirCmd = "/bin/mkdir -p '" + path.substr(0, lastSep) + "'";
		system(mkdirCmd.c_str());
	}
#endif
	FILE* fw = std::fopen(path.c_str(), "wb");
	if (fw)
	{
		std::fputs(token.c_str(), fw);
		std::fputs("\n", fw);
		std::fclose(fw);
	}
	return token;
}

// === Open browser for activation ===
static const char* kActivatePageUrl = "https://penta.bubbleapps.io/version-test/activate";

static void OpenActivationPage()
{
	const std::string mid = GetMachineIdHash();
	const std::string token = LoadOrCreateActivationToken();
	const std::string url = std::string(kActivatePageUrl)
		+ "?token=" + token
		+ "&mid=" + mid
		+ "&product=OST_WindyLines"
		+ "&ver=" OST_WINDYLINES_VERSION_FULL;

	DebugLog("[License] Opening activation page: token=%s mid=%s", token.c_str(), mid.c_str());

#ifdef _WIN32
	ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
	std::string cmd = "open '" + url + "' &";
	system(cmd.c_str());
#endif
}

// === License auto-refresh: background API check when cache expires ===
static const char* kLicenseApiEndpoint = "https://penta.bubbleapps.io/version-test/api/1.1/wf/ppplugin_test";
static const int kAuthorizedCacheTtlSec = 600;     // 10 min TTL (periodic re-verification)
static const int kDeniedCacheTtlSec = 600;          // 10 min TTL when denied
// Salt is XOR-obfuscated so `strings` cannot extract it from the binary.
// To regenerate: python3 -c "salt='OST_WL_2026_SALT_K9x3'; key=0xA7; print(','.join(f'0x{c^key:02X}' for c in salt.encode()))"
static std::string GetCacheSignatureSalt()
{
	static const unsigned char enc[] = {
		0xE8, 0xF4, 0xF3, 0xF8, 0xF0, 0xEB, 0xF8, 0x95, 0x97, 0x95, 0x91,
		0xF8, 0xF4, 0xE6, 0xEB, 0xF3, 0xF8, 0xEC, 0x9E, 0xDF, 0x94
	};
	static const unsigned char k = 0xA7;
	std::string s(sizeof(enc), '\0');
	for (size_t i = 0; i < sizeof(enc); ++i) { s[i] = static_cast<char>(enc[i] ^ k); }
	return s;
}
static std::atomic<bool> sAutoRefreshInProgress{false};
static std::atomic<uint32_t> sLastAutoRefreshAttemptMs{0};
static const uint32_t kMinAutoRefreshIntervalMs = 60000; // min 60s between API calls

// Post-activation rapid check: after user clicks "Activate", check every 5s for 2 minutes
static std::atomic<uint32_t> sActivationTriggeredAtMs{0};
static const uint32_t kPostActivationRapidWindowMs = 120000; // 2 min window
static const uint32_t kPostActivationCheckIntervalMs = 5000; // 5s interval during rapid window

// === Cache signature: DJB2-based HMAC to detect tampering ===
static std::string ComputeCacheSignature(const std::string& authorizedStr,
                                          const std::string& validatedUnixStr,
                                          const std::string& machineIdHash,
                                          const std::string& expireUnixStr)
{
	// Build: "authorized_val|validated_unix|machine_id|expire_unix|salt"
	std::string payload = authorizedStr + "|" + validatedUnixStr + "|" + machineIdHash + "|" + expireUnixStr + "|" + GetCacheSignatureSalt();
	uint32_t h1 = SimpleHash32(payload.c_str());
	// Double-hash for extra diffusion
	std::string pass2 = payload + "|" + std::to_string(h1);
	uint32_t h2 = SimpleHash32(pass2.c_str());
	char buf[20];
	std::snprintf(buf, sizeof(buf), "%08x%08x", h1, h2);
	return std::string(buf);
}

static void TriggerBackgroundCacheRefresh()
{
	bool expected = false;
	if (!sAutoRefreshInProgress.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
	{
		return;
	}

	const uint32_t nowMs = GetCurrentTimeMs();
	const uint32_t lastMs = sLastAutoRefreshAttemptMs.load(std::memory_order_relaxed);

	// Use shorter interval during post-activation rapid check window
	const uint32_t activatedAt = sActivationTriggeredAtMs.load(std::memory_order_relaxed);
	const bool inRapidWindow = (activatedAt != 0) && (nowMs - activatedAt < kPostActivationRapidWindowMs);
	const uint32_t minInterval = inRapidWindow ? kPostActivationCheckIntervalMs : kMinAutoRefreshIntervalMs;

	if (nowMs - lastMs < minInterval)
	{
		sAutoRefreshInProgress.store(false, std::memory_order_release);
		return;
	}
	sLastAutoRefreshAttemptMs.store(nowMs, std::memory_order_relaxed);

	const std::vector<std::string> paths = GetLicenseCachePaths();
	if (paths.empty())
	{
		sAutoRefreshInProgress.store(false, std::memory_order_release);
		return;
	}

	const std::string cachePath = paths.front();
	std::string cacheDir = cachePath;
	const size_t lastSlash = cacheDir.find_last_of("/\\");
	if (lastSlash != std::string::npos)
	{
		cacheDir = cacheDir.substr(0, lastSlash);
	}

#ifndef _WIN32
	const std::string endpoint(kLicenseApiEndpoint);
	const int ttlSecOk = kAuthorizedCacheTtlSec;
	const int ttlSecDenied = kDeniedCacheTtlSec;
	const std::string mid = GetMachineIdHash();

	std::thread([endpoint, ttlSecOk, ttlSecDenied, mid, cachePath, cacheDir]() {
		DebugLog("[License] background cache refresh started (Mac/popen)");

		// Ensure directory exists
		std::string mkdirCmd = "/bin/mkdir -p '" + cacheDir + "'";
		system(mkdirCmd.c_str());

		// Run curl and capture response
		std::string curlCmd =
			"/usr/bin/curl -s -m 10 -X POST "
			"-H 'Content-Type: application/json' "
			"-d '{\"action\":\"verify\",\"product\":\"OST_WindyLines\","
			"\"plugin_version\":\"" OST_WINDYLINES_VERSION_FULL "\","
			"\"platform\":\"mac\",\"machine_id\":\"" + mid + "\"}' "
			"'" + endpoint + "' 2>/dev/null";

		FILE* pipe = popen(curlCmd.c_str(), "r");
		if (!pipe)
		{
			DebugLog("[License] popen failed for curl");
			sAutoRefreshInProgress.store(false, std::memory_order_release);
			return;
		}

		std::string responseBody;
		char buf[512];
		while (std::fgets(buf, sizeof(buf), pipe) != nullptr)
		{
			responseBody += buf;
		}
		pclose(pipe);

		DebugLog("[License] API response: %s", responseBody.c_str());

		// Parse "authorized" field
		bool authorized = false;
		std::string reason = "unknown";
		if (responseBody.find("\"authorized\"") != std::string::npos)
		{
			if (responseBody.find("\"authorized\":true") != std::string::npos ||
				responseBody.find("\"authorized\": true") != std::string::npos ||
				responseBody.find("\"authorized\" : true") != std::string::npos)
			{
				authorized = true;
				reason = "ok";
			}
			else
			{
				authorized = false;
				reason = "denied";
			}
		}
		else
		{
			DebugLog("[License] API response missing 'authorized' field");
			sAutoRefreshInProgress.store(false, std::memory_order_release);
			return;
		}

		// Build cache content with signature
		const long long nowUnix = static_cast<long long>(std::time(nullptr));
		const int ttlSec = authorized ? ttlSecOk : ttlSecDenied;
		const long long expireUnix = nowUnix + ttlSec;
		const std::string authStr = authorized ? "true" : "false";
		const std::string nowStr = std::to_string(nowUnix);
		const std::string expireStr = std::to_string(expireUnix);
		const std::string sig = ComputeCacheSignature(authStr, nowStr, mid, expireStr);

		char content[768];
		snprintf(content, sizeof(content),
			"authorized=%s\nreason=%s\nvalidated_unix=%lld\ncache_expire_unix=%lld\n"
			"license_key_masked=\nmachine_id_hash=%s\ncache_signature=%s\n",
			authStr.c_str(), reason.c_str(), nowUnix, expireUnix,
			mid.c_str(), sig.c_str());

		// Atomic write: temp file then rename
		std::string tmpPath = "/tmp/ost_wl_cache_XXXXXX";
		char tmpBuf[64];
		std::snprintf(tmpBuf, sizeof(tmpBuf), "/tmp/ost_wl_cache_%08x", static_cast<unsigned>(nowUnix));
		std::string tmpFile(tmpBuf);

		FILE* fp = std::fopen(tmpFile.c_str(), "wb");
		if (fp)
		{
			std::fputs(content, fp);
			std::fclose(fp);
			if (std::rename(tmpFile.c_str(), cachePath.c_str()) != 0)
			{
				// rename failed (cross-device), try copy
				fp = std::fopen(cachePath.c_str(), "wb");
				if (fp)
				{
					std::fputs(content, fp);
					std::fclose(fp);
				}
				std::remove(tmpFile.c_str());
			}
			DebugLog("[License] cache updated: authorized=%s sig=%s path=%s",
				authStr.c_str(), sig.c_str(), cachePath.c_str());
		}

		sLicenseAuthenticated.store(authorized, std::memory_order_relaxed);

		// If authorized, clear rapid-check window
		if (authorized)
		{
			sActivationTriggeredAtMs.store(0, std::memory_order_relaxed);
		}

		sAutoRefreshInProgress.store(false, std::memory_order_release);
	}).detach();

	DebugLog("[License] background cache refresh triggered");
#else
	// Windows: WinHTTP-based background API call (runs in detached thread)
	const int ttlSecOk = kAuthorizedCacheTtlSec;
	const int ttlSecDenied = kDeniedCacheTtlSec;
	const std::string mid = GetMachineIdHash();
	std::thread([cachePath, cacheDir, ttlSecOk, ttlSecDenied, mid]() {
		DebugLog("[License] background cache refresh started (WinHTTP)");

		// Ensure directory exists
		CreateDirectoryA(cacheDir.c_str(), nullptr);

		std::string body = "{\"action\":\"verify\",\"product\":\"OST_WindyLines\","
			"\"plugin_version\":\"" OST_WINDYLINES_VERSION_FULL "\",\"platform\":\"win\","
			"\"machine_id\":\"" + mid + "\"}";

		HINTERNET hSession = WinHttpOpen(
			L"OST_WindyLines/1.0",
			WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
			WINHTTP_NO_PROXY_NAME,
			WINHTTP_NO_PROXY_BYPASS, 0);
		if (!hSession)
		{
			DebugLog("[License] WinHttpOpen failed: %lu", GetLastError());
			sAutoRefreshInProgress.store(false, std::memory_order_release);
			return;
		}

		HINTERNET hConnect = WinHttpConnect(hSession,
			L"penta.bubbleapps.io",
			INTERNET_DEFAULT_HTTPS_PORT, 0);
		if (!hConnect)
		{
			DebugLog("[License] WinHttpConnect failed: %lu", GetLastError());
			WinHttpCloseHandle(hSession);
			sAutoRefreshInProgress.store(false, std::memory_order_release);
			return;
		}

		HINTERNET hRequest = WinHttpOpenRequest(hConnect,
			L"POST",
			L"/version-test/api/1.1/wf/ppplugin_test",
			NULL, WINHTTP_NO_REFERER,
			WINHTTP_DEFAULT_ACCEPT_TYPES,
			WINHTTP_FLAG_SECURE);
		if (!hRequest)
		{
			DebugLog("[License] WinHttpOpenRequest failed: %lu", GetLastError());
			WinHttpCloseHandle(hConnect);
			WinHttpCloseHandle(hSession);
			sAutoRefreshInProgress.store(false, std::memory_order_release);
			return;
		}

		DWORD timeout = 10000;
		WinHttpSetOption(hRequest, WINHTTP_OPTION_CONNECT_TIMEOUT, &timeout, sizeof(timeout));
		WinHttpSetOption(hRequest, WINHTTP_OPTION_SEND_TIMEOUT, &timeout, sizeof(timeout));
		WinHttpSetOption(hRequest, WINHTTP_OPTION_RECEIVE_TIMEOUT, &timeout, sizeof(timeout));

		BOOL bResult = WinHttpSendRequest(hRequest,
			L"Content-Type: application/json", -1,
			(LPVOID)body.c_str(), (DWORD)body.size(), (DWORD)body.size(), 0);
		if (!bResult)
		{
			DebugLog("[License] WinHttpSendRequest failed: %lu", GetLastError());
			WinHttpCloseHandle(hRequest);
			WinHttpCloseHandle(hConnect);
			WinHttpCloseHandle(hSession);
			sAutoRefreshInProgress.store(false, std::memory_order_release);
			return;
		}

		bResult = WinHttpReceiveResponse(hRequest, NULL);
		if (!bResult)
		{
			DebugLog("[License] WinHttpReceiveResponse failed: %lu", GetLastError());
			WinHttpCloseHandle(hRequest);
			WinHttpCloseHandle(hConnect);
			WinHttpCloseHandle(hSession);
			sAutoRefreshInProgress.store(false, std::memory_order_release);
			return;
		}

		std::string responseBody;
		DWORD bytesAvailable = 0;
		while (WinHttpQueryDataAvailable(hRequest, &bytesAvailable) && bytesAvailable > 0)
		{
			std::vector<char> buf(bytesAvailable);
			DWORD bytesRead = 0;
			if (WinHttpReadData(hRequest, buf.data(), bytesAvailable, &bytesRead))
			{
				responseBody.append(buf.data(), bytesRead);
			}
			else { break; }
		}

		WinHttpCloseHandle(hRequest);
		WinHttpCloseHandle(hConnect);
		WinHttpCloseHandle(hSession);

		DebugLog("[License] API response: %s", responseBody.c_str());

		bool authorized = false;
		std::string reason = "unknown";
		if (responseBody.find("\"authorized\"") != std::string::npos)
		{
			if (responseBody.find("\"authorized\":true") != std::string::npos ||
				responseBody.find("\"authorized\": true") != std::string::npos ||
				responseBody.find("\"authorized\" : true") != std::string::npos)
			{
				authorized = true;
				reason = "ok";
			}
			else
			{
				authorized = false;
				reason = "denied";
			}
		}
		else
		{
			DebugLog("[License] API response missing 'authorized' field");
			sAutoRefreshInProgress.store(false, std::memory_order_release);
			return;
		}

		const long long nowUnix = static_cast<long long>(std::time(nullptr));
		const int ttlSec = authorized ? ttlSecOk : ttlSecDenied;
		const long long expireUnix = nowUnix + ttlSec;
		const std::string authStr = authorized ? "true" : "false";
		const std::string nowStr = std::to_string(nowUnix);
		const std::string expireStr = std::to_string(expireUnix);
		const std::string sig = ComputeCacheSignature(authStr, nowStr, mid, expireStr);

		char content[768];
		snprintf(content, sizeof(content),
			"authorized=%s\nreason=%s\nvalidated_unix=%lld\ncache_expire_unix=%lld\n"
			"license_key_masked=\nmachine_id_hash=%s\ncache_signature=%s\n",
			authStr.c_str(), reason.c_str(), nowUnix, expireUnix,
			mid.c_str(), sig.c_str());

		DWORD existingAttr = GetFileAttributesA(cachePath.c_str());
		if (existingAttr != INVALID_FILE_ATTRIBUTES && (existingAttr & FILE_ATTRIBUTE_DIRECTORY))
		{
			if (RemoveDirectoryA(cachePath.c_str()))
			{
				DebugLog("[License] removed unexpected directory at cache path: %s", cachePath.c_str());
			}
			else
			{
				DebugLog("[License] failed to remove unexpected directory at cache path: %s err=%lu",
					cachePath.c_str(), GetLastError());
			}
		}

		FILE* fp = std::fopen(cachePath.c_str(), "wb");
		if (fp)
		{
			std::fputs(content, fp);
			std::fclose(fp);
			DebugLog("[License] cache updated: authorized=%s sig=%s path=%s",
				authStr.c_str(), sig.c_str(), cachePath.c_str());
		}

		sLicenseAuthenticated.store(authorized, std::memory_order_relaxed);

		// If authorized, clear rapid-check window (no need to keep polling)
		if (authorized)
		{
			sActivationTriggeredAtMs.store(0, std::memory_order_relaxed);
		}

		sAutoRefreshInProgress.store(false, std::memory_order_release);
	}).detach();
#endif
}

// Shared with GPU — declared in OST_WindyLines_License.h
void RefreshLicenseAuthenticatedState(bool force)
{
	const uint32_t nowMs = GetCurrentTimeMs();
	if (!force)
	{
		const uint32_t lastMs = sLastLicenseRefreshTimeMs.load(std::memory_order_relaxed);
		if (nowMs - lastMs < kLicenseRefreshIntervalMs)
		{
			return;
		}
	}

	sLastLicenseRefreshTimeMs.store(nowMs, std::memory_order_relaxed);

	bool cachedAuthenticated = false;
	bool cacheExpired = false;
	if (LoadLicenseAuthenticatedFromCache(&cachedAuthenticated, &cacheExpired))
	{
		sLicenseAuthenticated.store(cachedAuthenticated, std::memory_order_relaxed);

		// If authorized during rapid check window, clear the window (no more polling needed)
		if (cachedAuthenticated)
		{
			sActivationTriggeredAtMs.store(0, std::memory_order_relaxed);

			// TTL expired but within offline grace — trigger background refresh
			// so subscription cancellation is detected promptly
			if (cacheExpired)
			{
				TriggerBackgroundCacheRefresh();
			}
		}
		else
		{
			// Cache is valid but authorized=false — during rapid check window,
			// force a background refresh to pick up server-side changes immediately
			const uint32_t activatedAt = sActivationTriggeredAtMs.load(std::memory_order_relaxed);
			if (activatedAt != 0 && (nowMs - activatedAt < kPostActivationRapidWindowMs))
			{
				TriggerBackgroundCacheRefresh();
			}
		}
	}
	else
	{
		sLicenseAuthenticated.store(false, std::memory_order_relaxed);

		// Cache expired or missing — always attempt refresh
		TriggerBackgroundCacheRefresh();
	}
}
static int NormalizePopupValue(int value, int maxValue)
{
	// Premiere Pro popup values are 1-based, convert to 0-based
	if (value >= 1 && value <= maxValue)
	{
		return value - 1;
	}
	// Already 0-based or out of range
	if (value >= 0 && value < maxValue)
	{
		return value;
	}
	return 0;
}

// Define shared static variables for SharedClipData
std::unordered_map<csSDK_int64, csSDK_int64> SharedClipData::clipStartMap;
std::unordered_map<csSDK_int64, ElementBounds> SharedClipData::elementBoundsMap;
std::mutex SharedClipData::mapMutex;

// ========================================================================
// Phase 3-1: Easing Look-Up Table (LUT)
// ========================================================================

#define EASING_LUT_SIZE 256
#define EASING_COUNT 28

// LUT storage: [easing_type][sample_index]
static float sEasingLUT[EASING_COUNT][EASING_LUT_SIZE];
static bool sEasingLUTInitialized = false;

// Forward declaration
static float ApplyEasing(float t, int easing);

/**
 * Initialize easing LUT by pre-computing all 28 easing functions
 * at 256 sample points (0.0 to 1.0)
 */
static void InitializeEasingLUT()
{
	if (sEasingLUTInitialized) return;
	
	for (int easingType = 0; easingType < EASING_COUNT; ++easingType)
	{
		for (int i = 0; i < EASING_LUT_SIZE; ++i)
		{
			const float t = static_cast<float>(i) / static_cast<float>(EASING_LUT_SIZE - 1);
			sEasingLUT[easingType][i] = ApplyEasing(t, easingType);
		}
	}
	
	sEasingLUTInitialized = true;
}

/**
 * Fast easing lookup using pre-computed LUT
 * @param t Input value [0.0, 1.0]
 * @param easingType Easing type [0-27]
 * @return Eased value
 */
static inline float ApplyEasingLUT(float t, int easingType)
{
	// Clamp input
	if (t <= 0.0f) return 0.0f;
	if (t >= 1.0f) return 1.0f;
	
	// Bounds check
	if (easingType < 0 || easingType >= EASING_COUNT) {
		return t; // Fallback to linear
	}
	
	// Map t to LUT index with linear interpolation
	const float fidx = t * static_cast<float>(EASING_LUT_SIZE - 1);
	const int idx = static_cast<int>(fidx);
	const float frac = fidx - static_cast<float>(idx);
	
	if (idx >= EASING_LUT_SIZE - 1) {
		return sEasingLUT[easingType][EASING_LUT_SIZE - 1];
	}
	
	// Linear interpolation between samples
	const float v0 = sEasingLUT[easingType][idx];
	const float v1 = sEasingLUT[easingType][idx + 1];
	return v0 + (v1 - v0) * frac;
}

// ========================================================================
// Phase 3-2: Trigonometric Function Look-Up Table (LUT)
// ========================================================================

#define TRIG_LUT_SIZE 256

// LUT for sine: covers [0, 2π]
static float sSinLUT[TRIG_LUT_SIZE];
static bool sTrigLUTInitialized = false;

/**
 * Initialize trigonometric LUT
 * Pre-computes sin values for [0, 2π] range
 */
static void InitializeTrigLUT()
{
	if (sTrigLUTInitialized) return;
	
	for (int i = 0; i < TRIG_LUT_SIZE; ++i)
	{
		const float angle = (2.0f * static_cast<float>(M_PI) * static_cast<float>(i)) / static_cast<float>(TRIG_LUT_SIZE);
		sSinLUT[i] = sinf(angle);
	}
	
	sTrigLUTInitialized = true;
}

/**
 * Fast sine lookup using pre-computed LUT
 * @param angle Angle in radians
 * @return sin(angle)
 */
static inline float FastSin(float angle)
{
	// Normalize angle to [0, 2π]
	const float twoPi = 2.0f * static_cast<float>(M_PI);
	float normalized = fmodf(angle, twoPi);
	if (normalized < 0.0f) normalized += twoPi;
	
	// Map to LUT index with linear interpolation
	const float fidx = (normalized / twoPi) * static_cast<float>(TRIG_LUT_SIZE - 1);
	const int idx = static_cast<int>(fidx);
	const float frac = fidx - static_cast<float>(idx);
	
	if (idx >= TRIG_LUT_SIZE - 1) {
		return sSinLUT[0];  // Wrap around
	}
	
	// Linear interpolation
	const float v0 = sSinLUT[idx];
	const float v1 = sSinLUT[idx + 1];
	return v0 + (v1 - v0) * frac;
}

/**
 * Fast cosine lookup using pre-computed LUT
 * Uses identity: cos(x) = sin(x + π/2)
 * @param angle Angle in radians
 * @return cos(angle)
 */
static inline float FastCos(float angle)
{
	return FastSin(angle + static_cast<float>(M_PI) * 0.5f);
}

/*
**
*/
static PF_Err GlobalSetup(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	RefreshLicenseAuthenticatedState(true);

	out_data->my_version	= PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);

	if (in_data->appl_id == 'PrMr')
	{
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
	}

	out_data->out_flags |= PF_OutFlag_USE_OUTPUT_EXTENT;
	out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
	out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
	out_data->out_flags |= PF_OutFlag_SEND_UPDATE_PARAMS_UI;
	out_data->out_flags2 |= PF_OutFlag2_PRESERVES_FULLY_OPAQUE_PIXELS;
	// Tell Premiere this effect uses timecode/sequence position to help with cache invalidation.
	out_data->out_flags2 |= PF_OutFlag2_I_USE_TIMECODE;

	// Initialize LUTs
	InitializeEasingLUT();
	InitializeTrigLUT();

	return PF_Err_NONE;
}
/*
**
*/
static PF_Err GlobalSetdown(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	return PF_Err_NONE;
}

struct LineParams
{
	float startFrame;
	float posX;
	float posY;
	float baseLen;
	float baseThick;
	float angle;
	float depthScale;
	float depthValue;  // Depth value (0-1) for blend mode
};

// LineDerived: Pre-computed values for each line
// Layout optimized for cache efficiency - frequently accessed fields first
// Total size: 60 bytes (fits in one 64-byte cache line)
struct alignas(64) LineDerived
{
	// Hot path: Early skip check (offset 0-7)
	float halfThick;   // Used for tiny line skip and bounding box
	float halfLen;     // Used for bounding box check
	
	// Hot path: Coordinate transform (offset 8-27)
	float centerX;
	float centerY;
	float cosA;
	float sinA;
	float segCenterX;
	
	// Hot path: Rendering (offset 28-35)
	float depthAlpha;      // Pre-computed depth fade alpha
	float invDenom;        // Pre-computed 1 / (2.0f * halfLen) for tail fade
	
	// Medium frequency: Color and effects (offset 36-47)
	float depth;           // Depth value for blend mode
	float focusAlpha;      // Alpha multiplier for focus blur
	float appearAlpha;     // Alpha multiplier for appear/disappear fade
	
	// Lower frequency (offset 48-55)
	float lineVelocity;    // Instantaneous velocity for motion blur
	int colorIndex;        // Palette color index (0-7)
	
	// Padding to align to cache line boundary (offset 56-63)
	int _padding;
};

struct LineInstanceState
{
	std::vector<LineParams> lineParams;
	std::vector<LineDerived> lineDerived;
	std::vector<int> tileOffsets;
	std::vector<int> tileCounts;
	std::vector<int> tileIndices;
	std::vector<char> lineActive;
	int lineCount = 0;
	int lineSeed = 0;
	float lineDepthStrength = 0.0f;
	int lineInterval = 0;
};

struct ClipTimeState
{
	A_long startTime = 0;
	A_long lastTime = 0;
	bool valid = false;
};

struct InstanceState
{
	std::mutex mutex;
	LineInstanceState lineState;
	ClipTimeState clipTime;
	bool allowMidPlayCached = false;
};

static std::unordered_map<const void*, std::shared_ptr<InstanceState>> sInstanceStates;
static std::mutex sInstanceStatesMutex;

// Shared with GPU — declared in OST_WindyLines_License.h
bool IsLicenseAuthenticated()
{
	return sLicenseAuthenticated.load(std::memory_order_relaxed);
}

static const void* GetInstanceKey(const PF_InData* in_data)
{
	if (in_data && in_data->effect_ref)
	{
		return in_data->effect_ref;
	}
	if (in_data && in_data->sequence_data)
	{
		return in_data->sequence_data;
	}
	return in_data;
}

static void SyncLineColorParams(PF_ParamDef* params[])
{
	if (!params || !params[OST_WINDYLINES_LINE_COLOR] ||
		!params[OST_WINDYLINES_LINE_COLOR_R] ||
		!params[OST_WINDYLINES_LINE_COLOR_G] ||
		!params[OST_WINDYLINES_LINE_COLOR_B])
	{
		return;
	}
	const PF_Pixel color = params[OST_WINDYLINES_LINE_COLOR]->u.cd.value;
	params[OST_WINDYLINES_LINE_COLOR_R]->u.fs_d.value = color.red / 255.0f;
	params[OST_WINDYLINES_LINE_COLOR_G]->u.fs_d.value = color.green / 255.0f;
	params[OST_WINDYLINES_LINE_COLOR_B]->u.fs_d.value = color.blue / 255.0f;
}

static void HideLineColorParams(PF_InData* in_data)
{
	if (!in_data)
	{
		return;
	}
	PF_ParamDef def;
	AEFX_CLR_STRUCT(def);
	def.ui_flags = PF_PUI_INVISIBLE;
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
	paramUtils->PF_UpdateParamUI(in_data->effect_ref, OST_WINDYLINES_LINE_COLOR_R, &def);
	paramUtils->PF_UpdateParamUI(in_data->effect_ref, OST_WINDYLINES_LINE_COLOR_G, &def);
	paramUtils->PF_UpdateParamUI(in_data->effect_ref, OST_WINDYLINES_LINE_COLOR_B, &def);
}

// Hide/Show Alpha Threshold based on Spawn Source selection
// UI visibility update is disabled because PF_UpdateParamUI breaks slider range
// The alpha threshold value is forced to 1.0 in Render when spawnSource == FULL_FRAME
static void UpdateAlphaThresholdVisibility(PF_InData* in_data, PF_ParamDef* params[])
{
	if (!in_data || !params)
	{
		return;
	}
}

static void ApplyRectColorUi(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[])
{
	(void)in_data;
	(void)out_data;
	(void)params;
}

// Update visibility of mode-dependent parameters (no checkbox groups)
static void UpdatePseudoGroupVisibility(
	PF_InData* in_data,
	PF_ParamDef* params[])
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
	if (!paramUtils.get())
	{
		return;
	}

	// Helper lambda to set visibility
	auto setVisible = [&](int paramId, bool visible)
	{
		PF_ParamDef paramCopy = *params[paramId];
		if (visible)
		{
			paramCopy.ui_flags &= ~PF_PUI_INVISIBLE;
		}
		else
		{
			paramCopy.ui_flags |= PF_PUI_INVISIBLE;
		}
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, &paramCopy);
	};

	// ========================================
	// Flat parameters (always visible):
	// - Basic: LINE_COUNT, LINE_LIFETIME, LINE_TRAVEL, LINE_SEED
	// - Color: COLOR_PRESET (unified dropdown)
	// - Appearance: LINE_THICKNESS, LINE_LENGTH, LINE_CAP, LINE_ANGLE, LINE_AA, LINE_TAIL_FADE
	// - Position: LINE_ORIGIN_MODE, LINE_INTERVAL, SPAWN_SCALE_X/Y, SPAWN_ROTATION, 
	//             SHOW_SPAWN_AREA, SPAWN_AREA_COLOR, ORIGIN_OFFSET_X/Y
	// - Animation: ANIM_PATTERN, CENTER_GAP, LINE_EASING, LINE_START_TIME, LINE_DURATION
	// ========================================

	// Color visibility control based on unified preset selection
	// Structure: 単色|(-|カスタム|(-|preset1|preset2|...
	// Note: Separators (-|) ARE included in menu numbering as items 2 and 4
	// 1-based UI values: 1=単色, 2=Sep, 3=カスタム, 4=Sep, 5=Rainbow, 6=Pastel, ...
	const int unifiedPresetValue = params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value;  // 1-based
	
	DebugLog("[Visibility] unifiedPresetValue = %d", unifiedPresetValue);
	
	// Single color: visible only when unified preset = 1 (単色)
	const bool showSingleColor = (unifiedPresetValue == 1);
	setVisible(OST_WINDYLINES_LINE_COLOR, showSingleColor);
	
	DebugLog("[Visibility] showSingleColor = %s", showSingleColor ? "true" : "false");
	
	// Custom Colors 1-8: visible only when unified preset = 3 (カスタム)
	const bool showCustomColors = (unifiedPresetValue == 3);
	setVisible(OST_WINDYLINES_CUSTOM_COLOR_1, showCustomColors);
	setVisible(OST_WINDYLINES_CUSTOM_COLOR_2, showCustomColors);
	setVisible(OST_WINDYLINES_CUSTOM_COLOR_3, showCustomColors);
	setVisible(OST_WINDYLINES_CUSTOM_COLOR_4, showCustomColors);
	setVisible(OST_WINDYLINES_CUSTOM_COLOR_5, showCustomColors);
	setVisible(OST_WINDYLINES_CUSTOM_COLOR_6, showCustomColors);
	setVisible(OST_WINDYLINES_CUSTOM_COLOR_7, showCustomColors);
	setVisible(OST_WINDYLINES_CUSTOM_COLOR_8, showCustomColors);
	
	DebugLog("[Visibility] showCustomColors = %s", showCustomColors ? "true" : "false");

	// ========================================
	// Linkage Parameters Visibility Control
	// ========================================
	
	// Travel Distance Linkage: show/hide rate and value based on mode
	const int travelLinkage = params[OST_WINDYLINES_TRAVEL_LINKAGE]->u.pd.value;
	const bool travelLinkageOff = (travelLinkage == 1); // 1 = Off (1-based popup)
	setVisible(OST_WINDYLINES_TRAVEL_LINKAGE_RATE, !travelLinkageOff); // Show rate when linkage is ON
	setVisible(OST_WINDYLINES_LINE_TRAVEL, travelLinkageOff);          // Show value when linkage is OFF
	
	// Thickness Linkage: show/hide rate and value based on mode
	const int thicknessLinkage = params[OST_WINDYLINES_THICKNESS_LINKAGE]->u.pd.value;
	const bool thicknessLinkageOff = (thicknessLinkage == 1); // 1 = Off (1-based popup)
	setVisible(OST_WINDYLINES_THICKNESS_LINKAGE_RATE, !thicknessLinkageOff); // Show rate when linkage is ON
	setVisible(OST_WINDYLINES_LINE_THICKNESS, thicknessLinkageOff);           // Show value when linkage is OFF
	
	// Length Linkage: show/hide rate and value based on mode
	const int lengthLinkage = params[OST_WINDYLINES_LENGTH_LINKAGE]->u.pd.value;
	const bool lengthLinkageOff = (lengthLinkage == 1); // 1 = Off (1-based popup)
	setVisible(OST_WINDYLINES_LENGTH_LINKAGE_RATE, !lengthLinkageOff); // Show rate when linkage is ON
	setVisible(OST_WINDYLINES_LINE_LENGTH, lengthLinkageOff);           // Show value when linkage is OFF

	// Color Preset: always visible
	setVisible(OST_WINDYLINES_COLOR_PRESET, true);

	// Shadow / Advanced / Focus params are always visible (no checkbox groups)
}

static void ApplyEffectPreset(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	int presetIndex)
{
	if (presetIndex < 0 || presetIndex >= kEffectPresetCount)
	{
		return;
	}

	const EffectPreset& preset = kEffectPresets[presetIndex];
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
	if (!paramUtils.get())
	{
		return;
	}

	auto updateFloat = [&](int paramId, float value)
	{
		params[paramId]->u.fs_d.value = value;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	auto updatePopup = [&](int paramId, int value)
	{
		params[paramId]->u.pd.value = value;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	auto updateAngle = [&](int paramId, float degrees)
	{
		const A_long fixedAngle = FLOAT2FIX(degrees);
		params[paramId]->u.ad.value = fixedAngle;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	// Basic settings
	updateFloat(OST_WINDYLINES_LINE_COUNT, static_cast<float>(preset.count));
	updateFloat(OST_WINDYLINES_LINE_LIFETIME, preset.lifetime);
	updateFloat(OST_WINDYLINES_LINE_TRAVEL, preset.travel);
	
	// Appearance
	updateFloat(OST_WINDYLINES_LINE_THICKNESS, preset.thickness);
	updateFloat(OST_WINDYLINES_LINE_LENGTH, preset.length);
	updateAngle(OST_WINDYLINES_LINE_ANGLE, preset.angle);
	updateFloat(OST_WINDYLINES_LINE_TAIL_FADE, preset.tailFade);
	updateFloat(OST_WINDYLINES_LINE_AA, preset.aa);
	
	// Position & Spawn
	updatePopup(OST_WINDYLINES_LINE_ORIGIN_MODE, preset.originMode);
	updateFloat(OST_WINDYLINES_LINE_SPAWN_SCALE_X, preset.spawnScaleX);
	updateFloat(OST_WINDYLINES_LINE_SPAWN_SCALE_Y, preset.spawnScaleY);
	updateFloat(OST_WINDYLINES_ORIGIN_OFFSET_X, preset.originOffsetX);
	updateFloat(OST_WINDYLINES_ORIGIN_OFFSET_Y, preset.originOffsetY);
	updateFloat(OST_WINDYLINES_LINE_INTERVAL, preset.interval);
	
	// Animation
	updatePopup(OST_WINDYLINES_ANIM_PATTERN, preset.animPattern);
	updateFloat(OST_WINDYLINES_CENTER_GAP, preset.centerGap);
	updatePopup(OST_WINDYLINES_LINE_EASING, preset.easing + 1);
	updateFloat(OST_WINDYLINES_LINE_START_TIME, preset.startTime);
	updateFloat(OST_WINDYLINES_LINE_DURATION, preset.duration);
	
	// Advanced
	updatePopup(OST_WINDYLINES_BLEND_MODE, preset.blendMode);
	updateFloat(OST_WINDYLINES_LINE_DEPTH_STRENGTH, preset.depthStrength);
	
	// New parameters
	updatePopup(OST_WINDYLINES_LINE_CAP, preset.lineCap);
	// Use unified preset index (colorMode hidden, only COLOR_PRESET used)
	updatePopup(OST_WINDYLINES_COLOR_PRESET, preset.unifiedPresetIndex + 1);  // Convert to 1-based
	updatePopup(OST_WINDYLINES_SPAWN_SOURCE, preset.spawnSource);
	
	auto updateCheckbox = [&](int paramId, bool value)
	{
		params[paramId]->u.bd.value = value ? 1 : 0;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};
	updateCheckbox(OST_WINDYLINES_HIDE_ELEMENT, preset.hideElement);

	// Force UI refresh and re-render
	out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
	out_data->out_flags |= PF_OutFlag_REFRESH_UI;
}

static void ApplyDefaultEffectParams(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[])
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
	if (!paramUtils.get())
	{
		return;
	}

	auto updateFloat = [&](int paramId, float value)
	{
		params[paramId]->u.fs_d.value = value;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	auto updatePopup = [&](int paramId, int value)
	{
		params[paramId]->u.pd.value = value;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	auto updateAngle = [&](int paramId, float degrees)
	{
		const A_long fixedAngle = FLOAT2FIX(degrees);
		params[paramId]->u.ad.value = fixedAngle;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};

	// Basic settings
	updateFloat(OST_WINDYLINES_LINE_COUNT, static_cast<float>(LINE_COUNT_DFLT));
	updateFloat(OST_WINDYLINES_LINE_LIFETIME, static_cast<float>(LINE_LIFETIME_DFLT));
	updateFloat(OST_WINDYLINES_LINE_TRAVEL, static_cast<float>(LINE_TRAVEL_DFLT));
	
	// Appearance
	updateFloat(OST_WINDYLINES_LINE_THICKNESS, static_cast<float>(LINE_THICKNESS_DFLT));
	updateFloat(OST_WINDYLINES_LINE_LENGTH, static_cast<float>(LINE_LENGTH_DFLT));
	updateAngle(OST_WINDYLINES_LINE_ANGLE, 0.0f);
	updateFloat(OST_WINDYLINES_LINE_TAIL_FADE, static_cast<float>(LINE_TAIL_FADE_DFLT));
	updateFloat(OST_WINDYLINES_LINE_AA, static_cast<float>(LINE_AA_DFLT));
	
	// Position & Spawn
	updatePopup(OST_WINDYLINES_LINE_ORIGIN_MODE, LINE_ORIGIN_MODE_DFLT);
	updateFloat(OST_WINDYLINES_LINE_SPAWN_SCALE_X, static_cast<float>(LINE_SPAWN_SCALE_X_DFLT));
	updateFloat(OST_WINDYLINES_LINE_SPAWN_SCALE_Y, static_cast<float>(LINE_SPAWN_SCALE_Y_DFLT));
	updateFloat(OST_WINDYLINES_ORIGIN_OFFSET_X, static_cast<float>(ORIGIN_OFFSET_X_DFLT));
	updateFloat(OST_WINDYLINES_ORIGIN_OFFSET_Y, static_cast<float>(ORIGIN_OFFSET_Y_DFLT));
	updateFloat(OST_WINDYLINES_LINE_INTERVAL, static_cast<float>(LINE_INTERVAL_DFLT));
	
	// Animation
	updatePopup(OST_WINDYLINES_ANIM_PATTERN, ANIM_PATTERN_DFLT);
	updateFloat(OST_WINDYLINES_CENTER_GAP, static_cast<float>(CENTER_GAP_DFLT));
	updatePopup(OST_WINDYLINES_LINE_EASING, LINE_EASING_DFLT);
	updateFloat(OST_WINDYLINES_LINE_START_TIME, static_cast<float>(LINE_START_TIME_DFLT));
	updateFloat(OST_WINDYLINES_LINE_DURATION, static_cast<float>(LINE_DURATION_DFLT));
	
	// Advanced
	updatePopup(OST_WINDYLINES_BLEND_MODE, BLEND_MODE_DFLT);
	updateFloat(OST_WINDYLINES_LINE_DEPTH_STRENGTH, static_cast<float>(LINE_DEPTH_DFLT));
	
	// New parameters (use defaults from .h file)
	updatePopup(OST_WINDYLINES_LINE_CAP, LINE_CAP_DFLT);
	updatePopup(OST_WINDYLINES_COLOR_MODE, COLOR_MODE_DFLT);
	updatePopup(OST_WINDYLINES_COLOR_PRESET, COLOR_PRESET_DFLT);
	updatePopup(OST_WINDYLINES_SPAWN_SOURCE, SPAWN_SOURCE_DFLT);
	
	auto updateCheckbox = [&](int paramId, bool value)
	{
		params[paramId]->u.bd.value = value ? 1 : 0;
		params[paramId]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
		paramUtils->PF_UpdateParamUI(in_data->effect_ref, paramId, params[paramId]);
	};
	updateCheckbox(OST_WINDYLINES_HIDE_ELEMENT, HIDE_ELEMENT_DFLT);

	out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
	out_data->out_flags |= PF_OutFlag_REFRESH_UI;
}

/*
**
*/
// UI labels are the first string in each PF_ADD_* call below.
// Change those strings to rename parameters in the effect UI.
static PF_Err ParamsSetup(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_ParamDef	def;

	// ============================================================
	// Effect Preset (top-level, no group)
	// ============================================================
	AEFX_CLR_STRUCT(def);
	std::string presetLabels = "デフォルト|";
	for (int i = 0; i < kEffectPresetCount; ++i)
	{
		presetLabels += kEffectPresets[i].name;
		if (i < kEffectPresetCount - 1)
		{
			presetLabels += "|";
		}
	}
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_EFFECT_PRESET,
		1 + kEffectPresetCount,
		1,
		presetLabels.c_str(),
		OST_WINDYLINES_EFFECT_PRESET);

	// Random Seed (moved to top, after preset)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SEED,
		LINE_SEED_MIN_VALUE,
		LINE_SEED_MAX_VALUE,
		LINE_SEED_MIN_SLIDER,
		LINE_SEED_MAX_SLIDER,
		LINE_SEED_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_LINE_SEED);

	// ============================================================
	// Basic Settings
	// ============================================================

	// Line Count
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_LINE_COUNT,
		LINE_COUNT_MIN_VALUE,
		LINE_COUNT_MAX_VALUE,
		LINE_COUNT_MIN_SLIDER,
		LINE_COUNT_MAX_SLIDER,
		LINE_COUNT_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_LINE_COUNT);

	// Line Lifetime (frames)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_LIFETIME,
		LINE_LIFETIME_MIN_VALUE,
		LINE_LIFETIME_MAX_VALUE,
		LINE_LIFETIME_MIN_SLIDER,
		LINE_LIFETIME_MAX_SLIDER,
		LINE_LIFETIME_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_LINE_LIFETIME);

	// Spawn Interval (moved from Position group)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_INTERVAL,
		LINE_INTERVAL_MIN_VALUE,
		LINE_INTERVAL_MAX_VALUE,
		LINE_INTERVAL_MIN_SLIDER,
		LINE_INTERVAL_MAX_SLIDER,
		LINE_INTERVAL_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_LINE_INTERVAL);

	// ============================================================
	// Color Settings
	// ============================================================

	// Color Mode (kept for backward compatibility but hidden)
	// Now integrated into Color Preset
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;  // Hide from UI
	PF_ADD_POPUP(
		P_COLOR_MODE,
		3,
		COLOR_MODE_DFLT,
		PM_COLOR_MODE,
		OST_WINDYLINES_COLOR_MODE);

	// Color Preset (Unified: Single|Custom|-)|Preset1|Preset2|...)
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_COLOR_PRESET,
		kUnifiedPresetCount,
		COLOR_PRESET_DFLT,
		GetUnifiedPresetMenuString(),
		OST_WINDYLINES_COLOR_PRESET);

	// Single Color
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_COLOR(
		P_COLOR,
		LINE_COLOR_DFLT_R8,
		LINE_COLOR_DFLT_G8,
		LINE_COLOR_DFLT_B8,
		OST_WINDYLINES_LINE_COLOR);

	// Custom Colors 1-8
	AEFX_CLR_STRUCT(def);
	def.ui_flags = PF_PUI_ECW_SEPARATOR;
	PF_ADD_COLOR(P_CUSTOM_1, 255, 0, 0, OST_WINDYLINES_CUSTOM_COLOR_1);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_2, 255, 128, 0, OST_WINDYLINES_CUSTOM_COLOR_2);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_3, 255, 255, 0, OST_WINDYLINES_CUSTOM_COLOR_3);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_4, 0, 255, 0, OST_WINDYLINES_CUSTOM_COLOR_4);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_5, 0, 255, 255, OST_WINDYLINES_CUSTOM_COLOR_5);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_6, 0, 0, 255, OST_WINDYLINES_CUSTOM_COLOR_6);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_7, 128, 0, 255, OST_WINDYLINES_CUSTOM_COLOR_7);
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_CUSTOM_8, 255, 0, 255, OST_WINDYLINES_CUSTOM_COLOR_8);

	// Easing (before Travel Distance Linkage)
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_EASING,
		28,
		LINE_EASING_DFLT,
		PM_EASING,
		OST_WINDYLINES_LINE_EASING);

	// Travel Distance Linkage (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_TRAVEL_LINKAGE,
		3,
		LINKAGE_MODE_DFLT,
		PM_LINKAGE_MODE,
		OST_WINDYLINES_TRAVEL_LINKAGE);

	// Travel Distance Linkage Rate (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_TRAVEL_LINKAGE_RATE,
		LINKAGE_RATE_MIN_VALUE,
		LINKAGE_RATE_MAX_VALUE,
		LINKAGE_RATE_MIN_SLIDER,
		LINKAGE_RATE_MAX_SLIDER,
		LINKAGE_RATE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_TRAVEL_LINKAGE_RATE);

	// Travel Distance (moved from Basic Settings)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_TRAVEL,
		LINE_TRAVEL_MIN_VALUE,
		LINE_TRAVEL_MAX_VALUE,
		LINE_TRAVEL_MIN_SLIDER,
		LINE_TRAVEL_MAX_SLIDER,
		LINE_TRAVEL_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_TRAVEL);

	// ============================================================
	// Appearance
	// ============================================================

	// Thickness Linkage (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_THICKNESS_LINKAGE,
		3,
		LINKAGE_MODE_DFLT,
		PM_LINKAGE_MODE,
		OST_WINDYLINES_THICKNESS_LINKAGE);

	// Thickness Linkage Rate (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_THICKNESS_LINKAGE_RATE,
		LINKAGE_RATE_MIN_VALUE,
		LINKAGE_RATE_MAX_VALUE,
		LINKAGE_RATE_MIN_SLIDER,
		LINKAGE_RATE_MAX_SLIDER,
		LINKAGE_RATE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_THICKNESS_LINKAGE_RATE);

	// Line Thickness
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_THICKNESS,
		LINE_THICKNESS_MIN_VALUE,
		LINE_THICKNESS_MAX_VALUE,
		LINE_THICKNESS_MIN_SLIDER,
		LINE_THICKNESS_MAX_SLIDER,
		LINE_THICKNESS_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_THICKNESS);

	// Length Linkage (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_LENGTH_LINKAGE,
		3,
		LINKAGE_MODE_DFLT,
		PM_LINKAGE_MODE,
		OST_WINDYLINES_LENGTH_LINKAGE);

	// Length Linkage Rate (NEW - moved from old Linkage group)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_LENGTH_LINKAGE_RATE,
		LINKAGE_RATE_MIN_VALUE,
		LINKAGE_RATE_MAX_VALUE,
		LINKAGE_RATE_MIN_SLIDER,
		LINKAGE_RATE_MAX_SLIDER,
		LINKAGE_RATE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_LENGTH_LINKAGE_RATE);

	// Line Length
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_LENGTH,
		LINE_LENGTH_MIN_VALUE,
		LINE_LENGTH_MAX_VALUE,
		LINE_LENGTH_MIN_SLIDER,
		LINE_LENGTH_MAX_SLIDER,
		LINE_LENGTH_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_LENGTH);

	// Line Angle
	AEFX_CLR_STRUCT(def);
	PF_ADD_ANGLE(
		P_ANGLE,
		0,
		OST_WINDYLINES_LINE_ANGLE);

	// Line Cap
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_LINE_CAP,
		2,
		LINE_CAP_DFLT,
		PM_LINE_CAP,
		OST_WINDYLINES_LINE_CAP);

	// Tail Fade
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_TAIL_FADE,
		LINE_TAIL_FADE_MIN_VALUE,
		LINE_TAIL_FADE_MAX_VALUE,
		LINE_TAIL_FADE_MIN_SLIDER,
		LINE_TAIL_FADE_MAX_SLIDER,
		LINE_TAIL_FADE_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_TAIL_FADE);

	// ============================================================
	// Line Origin (線�E起点)
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_POSITION_HEADER, OST_WINDYLINES_POSITION_HEADER);

	// 1. Spawn Source
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_SPAWN_SOURCE,
		2,
		SPAWN_SOURCE_DFLT,
		P_SPAWN_SOURCE_CHOICES,
		OST_WINDYLINES_SPAWN_SOURCE);

	// 2. Alpha Threshold
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_ALPHA_THRESH,
		LINE_ALPHA_THRESH_MIN_VALUE,
		LINE_ALPHA_THRESH_MAX_VALUE,
		LINE_ALPHA_THRESH_MIN_SLIDER,
		LINE_ALPHA_THRESH_MAX_SLIDER,
		LINE_ALPHA_THRESH_DFLT,
		PF_Precision_THOUSANDTHS,
		0,
		0,
		OST_WINDYLINES_LINE_ALPHA_THRESH);

	// 3. Wind Origin Mode (moved before Animation Pattern)
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_ORIGIN_MODE,
		3,
		LINE_ORIGIN_MODE_DFLT,
		PM_ORIGIN_MODE,
		OST_WINDYLINES_LINE_ORIGIN_MODE);

	// 4. Animation Pattern (Direction)
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_ANIM_PATTERN,
		3,
		ANIM_PATTERN_DFLT,
		PM_ANIM_PATTERN,
		OST_WINDYLINES_ANIM_PATTERN);

	// 5. Start Time
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_START_TIME,
		LINE_START_TIME_MIN_VALUE,
		LINE_START_TIME_MAX_VALUE,
		LINE_START_TIME_MIN_SLIDER,
		LINE_START_TIME_MAX_SLIDER,
		LINE_START_TIME_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_LINE_START_TIME);

	// 6. Duration
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_DURATION,
		LINE_DURATION_MIN_VALUE,
		LINE_DURATION_MAX_VALUE,
		LINE_DURATION_MIN_SLIDER,
		LINE_DURATION_MAX_SLIDER,
		LINE_DURATION_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_LINE_DURATION);

	// 7. Depth Strength (moved from Advanced)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_DEPTH_STRENGTH,
		LINE_DEPTH_MIN_VALUE,
		LINE_DEPTH_MAX_VALUE,
		LINE_DEPTH_MIN_SLIDER,
		LINE_DEPTH_MAX_SLIDER,
		LINE_DEPTH_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_DEPTH_STRENGTH);

	// 8. Center Gap (moved before Offset X)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_CENTER_GAP,
		CENTER_GAP_MIN_VALUE,
		CENTER_GAP_MAX_VALUE,
		CENTER_GAP_MIN_SLIDER,
		CENTER_GAP_MAX_SLIDER,
		CENTER_GAP_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		OST_WINDYLINES_CENTER_GAP);

	// 9. Origin Offset X
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_OFFSET_X,
		ORIGIN_OFFSET_X_MIN_VALUE,
		ORIGIN_OFFSET_X_MAX_VALUE,
		ORIGIN_OFFSET_X_MIN_SLIDER,
		ORIGIN_OFFSET_X_MAX_SLIDER,
		ORIGIN_OFFSET_X_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_ORIGIN_OFFSET_X);

	// 10. Origin Offset Y
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_OFFSET_Y,
		ORIGIN_OFFSET_Y_MIN_VALUE,
		ORIGIN_OFFSET_Y_MAX_VALUE,
		ORIGIN_OFFSET_Y_MIN_SLIDER,
		ORIGIN_OFFSET_Y_MAX_SLIDER,
		ORIGIN_OFFSET_Y_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_ORIGIN_OFFSET_Y);

	// 11. Spawn Scale X
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SPAWN_SCALE_X,
		LINE_SPAWN_SCALE_X_MIN_VALUE,
		LINE_SPAWN_SCALE_X_MAX_VALUE,
		LINE_SPAWN_SCALE_X_MIN_SLIDER,
		LINE_SPAWN_SCALE_X_MAX_SLIDER,
		LINE_SPAWN_SCALE_X_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_SPAWN_SCALE_X);

	// 12. Spawn Scale Y
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SPAWN_SCALE_Y,
		LINE_SPAWN_SCALE_Y_MIN_VALUE,
		LINE_SPAWN_SCALE_Y_MAX_VALUE,
		LINE_SPAWN_SCALE_Y_MIN_SLIDER,
		LINE_SPAWN_SCALE_Y_MAX_SLIDER,
		LINE_SPAWN_SCALE_Y_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_SPAWN_SCALE_Y);

	// 13. Spawn Rotation
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SPAWN_ROTATION,
		LINE_SPAWN_ROTATION_MIN_VALUE,
		LINE_SPAWN_ROTATION_MAX_VALUE,
		LINE_SPAWN_ROTATION_MIN_SLIDER,
		LINE_SPAWN_ROTATION_MAX_SLIDER,
		LINE_SPAWN_ROTATION_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_SPAWN_ROTATION);

	// 14. Show Spawn Area
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(P_SHOW_SPAWN, "", SHOW_SPAWN_AREA_DFLT, 0, OST_WINDYLINES_LINE_SHOW_SPAWN_AREA);

	// 16. Spawn Area Color
	AEFX_CLR_STRUCT(def);
	PF_ADD_COLOR(P_SPAWN_COLOR, 128, 128, 255, OST_WINDYLINES_LINE_SPAWN_AREA_COLOR);  // Light blue default

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(OST_WINDYLINES_POSITION_TOPIC_END);

	// ============================================================
	// Shadow
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_SHADOW, OST_WINDYLINES_SHADOW_HEADER);

	// Shadow Enable
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(P_SHADOW_ENABLE, "", SHADOW_ENABLE_DFLT, 0, OST_WINDYLINES_SHADOW_ENABLE);

	// Shadow Color
	AEFX_CLR_STRUCT(def);
	def.u.cd.value.red = static_cast<A_u_short>(SHADOW_COLOR_R_DFLT * 65535);
	def.u.cd.value.green = static_cast<A_u_short>(SHADOW_COLOR_G_DFLT * 65535);
	def.u.cd.value.blue = static_cast<A_u_short>(SHADOW_COLOR_B_DFLT * 65535);
	def.u.cd.value.alpha = 65535;
	def.u.cd.dephault = def.u.cd.value;
	PF_ADD_COLOR(P_SHADOW_COLOR, def.u.cd.value.red, def.u.cd.value.green, def.u.cd.value.blue, OST_WINDYLINES_SHADOW_COLOR);

	// Shadow Offset X
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SHADOW_OFFSET_X,
		SHADOW_OFFSET_X_MIN,
		SHADOW_OFFSET_X_MAX,
		SHADOW_OFFSET_X_MIN,
		SHADOW_OFFSET_X_MAX,
		SHADOW_OFFSET_X_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_SHADOW_OFFSET_X);

	// Shadow Offset Y
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SHADOW_OFFSET_Y,
		SHADOW_OFFSET_Y_MIN,
		SHADOW_OFFSET_Y_MAX,
		SHADOW_OFFSET_Y_MIN,
		SHADOW_OFFSET_Y_MAX,
		SHADOW_OFFSET_Y_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_SHADOW_OFFSET_Y);

	// Shadow Opacity
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_SHADOW_OPACITY,
		SHADOW_OPACITY_MIN,
		SHADOW_OPACITY_MAX,
		SHADOW_OPACITY_MIN,
		SHADOW_OPACITY_MAX,
		SHADOW_OPACITY_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		OST_WINDYLINES_SHADOW_OPACITY);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(OST_WINDYLINES_SHADOW_TOPIC_END);

	// ============================================================
	// Motion Blur
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_MOTION_BLUR, OST_WINDYLINES_MOTION_BLUR_HEADER);

	// Motion Blur Enable
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(P_MOTION_BLUR, "", MOTION_BLUR_ENABLE_DFLT, 0, OST_WINDYLINES_MOTION_BLUR_ENABLE);

	// Motion Blur Samples (Quality)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_BLUR_SAMPLES,
		MOTION_BLUR_SAMPLES_MIN_VALUE,
		MOTION_BLUR_SAMPLES_MAX_VALUE,
		MOTION_BLUR_SAMPLES_MIN_SLIDER,
		MOTION_BLUR_SAMPLES_MAX_SLIDER,
		MOTION_BLUR_SAMPLES_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_MOTION_BLUR_SAMPLES);

	// Motion Blur Shutter Angle (0-360°)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_BLUR_ANGLE,
		MOTION_BLUR_ANGLE_MIN_VALUE,
		MOTION_BLUR_ANGLE_MAX_VALUE,
		MOTION_BLUR_ANGLE_MIN_SLIDER,
		MOTION_BLUR_ANGLE_MAX_SLIDER,
		MOTION_BLUR_ANGLE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		OST_WINDYLINES_MOTION_BLUR_STRENGTH);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(OST_WINDYLINES_MOTION_BLUR_TOPIC_END);

	// ============================================================
	// Advanced
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_ADVANCED_HEADER, OST_WINDYLINES_ADVANCED_HEADER);

	// Anti-Aliasing (moved to Advanced)
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(
		P_AA,
		LINE_AA_MIN_VALUE,
		LINE_AA_MAX_VALUE,
		LINE_AA_MIN_SLIDER,
		LINE_AA_MAX_SLIDER,
		LINE_AA_DFLT,
		PF_Precision_TENTHS,
		0,
		0,
		OST_WINDYLINES_LINE_AA);

	// Hide Element (lines only)
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX(P_HIDE_ELEMENT, "", HIDE_ELEMENT_DFLT, 0, OST_WINDYLINES_HIDE_ELEMENT);

	// Blend Mode
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(
		P_BLEND_MODE,
		4,
		BLEND_MODE_DFLT,
		PM_BLEND_MODE,
		OST_WINDYLINES_BLEND_MODE);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(OST_WINDYLINES_ADVANCED_TOPIC_END);

	// ============================================================
	// Hidden Parameters (for backwards compatibility)
	// ============================================================
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;
	PF_ADD_CHECKBOX("", "", LINE_ALLOW_MIDPLAY_DFLT, 0, OST_WINDYLINES_LINE_ALLOW_MIDPLAY);

	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;
	PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
		LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
		PF_Precision_TENTHS, 0, 0, OST_WINDYLINES_LINE_COLOR_R);

	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;
	PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
		LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
		PF_Precision_TENTHS, 0, 0, OST_WINDYLINES_LINE_COLOR_G);

	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
	def.ui_flags = PF_PUI_INVISIBLE;
	PF_ADD_FLOAT_SLIDERX("", LINE_COLOR_CH_MIN_VALUE, LINE_COLOR_CH_MAX_VALUE,
		LINE_COLOR_CH_MIN_SLIDER, LINE_COLOR_CH_MAX_SLIDER, LINE_COLOR_CH_DFLT,
		PF_Precision_TENTHS, 0, 0, OST_WINDYLINES_LINE_COLOR_B);

	// ============================================================
	// License Section
	// ============================================================
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(P_LICENSE_HEADER, OST_WINDYLINES_LICENSE_HEADER);

	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP(
		P_LICENSE_STATUS,
		2,
		1,
		PM_LICENSE_STATUS,
		OST_WINDYLINES_LICENSE_STATUS);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(OST_WINDYLINES_LICENSE_TOPIC_END);

	out_data->num_params = OST_WINDYLINES_NUM_PARAMS;
	return PF_Err_NONE;
}

/*
**
*/
static PF_Err Render(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	RefreshLicenseAuthenticatedState(false);

	auto normalizePopup = [](int value, int maxValue) {
		if (value >= 1 && value <= maxValue)
		{
			return value - 1;
		}
		if (value >= 0 && value < maxValue)
		{
			return value;
		}
		return 0;
	};

	if (in_data->appl_id == 'PrMr')
	{
		PF_LayerDef* src = &params[0]->u.ld;
		PF_LayerDef* dest = output;

		const char* srcData = (const char*)src->data;
		char* destData = (char*)dest->data;

		// Color Mode and Preset (unified index)
		// Note: Separators (-|) ARE included in menu numbering
		// Menu display: 単色|(-|カスタム|(-|Rainbow|Pastel|Forest|...
		// UI 1-based:   1=単色, 2=Sep, 3=カスタム, 4=Sep, 5=Rainbow, 6=Pastel, 7=Forest, ...
		// After normalization (0-based): 0=単色, 1=Sep, 2=カスタム, 3=Sep, 4=Rainbow, 5=Pastel, 6=Forest, ...
		const int unifiedPresetIndex = normalizePopup(params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value, kUnifiedPresetCount);
		
		// Convert unified index to colorMode and presetIndex
		int colorMode, presetIndex;
		UnifiedIndexToColorModeAndPreset(unifiedPresetIndex, colorMode, presetIndex);
		
		// Debug logging for color preset selection
		DebugLog("[CPU ColorPreset] UI value: %d (1-based) → Normalized: %d (0-based) → colorMode: %d, presetIndex: %d",
			params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value, unifiedPresetIndex, colorMode, presetIndex);
		
		// Build color palette (8 colors, RGB normalized)
		float colorPalette[8][3];
		
		if (colorMode == 0)  // Single (0-based)
		{
			// Single color mode: all 8 slots have the same color
			const PF_Pixel singleColor = params[OST_WINDYLINES_LINE_COLOR]->u.cd.value;
			const float singleR = singleColor.red / 255.0f;
			const float singleG = singleColor.green / 255.0f;
			const float singleB = singleColor.blue / 255.0f;
			for (int i = 0; i < 8; ++i)
			{
				colorPalette[i][0] = singleR;
				colorPalette[i][1] = singleG;
				colorPalette[i][2] = singleB;
			}
			DebugLog("[ColorPreset] Single color mode: R=%.2f G=%.2f B=%.2f", singleR, singleG, singleB);
		}
		else if (colorMode == 1)  // Custom (0-based)
		{
			// Custom mode: load from custom color parameters
			const int customColorParams[8] = {
				OST_WINDYLINES_CUSTOM_COLOR_1, OST_WINDYLINES_CUSTOM_COLOR_2,
				OST_WINDYLINES_CUSTOM_COLOR_3, OST_WINDYLINES_CUSTOM_COLOR_4,
				OST_WINDYLINES_CUSTOM_COLOR_5, OST_WINDYLINES_CUSTOM_COLOR_6,
				OST_WINDYLINES_CUSTOM_COLOR_7, OST_WINDYLINES_CUSTOM_COLOR_8
			};
			for (int i = 0; i < 8; ++i)
			{
				const PF_Pixel customColor = params[customColorParams[i]]->u.cd.value;
				colorPalette[i][0] = customColor.red / 255.0f;
				colorPalette[i][1] = customColor.green / 255.0f;
				colorPalette[i][2] = customColor.blue / 255.0f;
			}
			DebugLog("[ColorPreset] Custom mode: Loaded 8 custom colors, Color[0]: R=%.2f G=%.2f B=%.2f", 
				colorPalette[0][0], colorPalette[0][1], colorPalette[0][2]);
		}
		else  // Preset (colorMode == 2, 0-based)
		{
			BuildPresetPalette(colorPalette, presetIndex);
			DebugLog("[ColorPreset] Loaded 8 colors, Color[0]: R=%.2f G=%.2f B=%.2f", 
				colorPalette[0][0], colorPalette[0][1], colorPalette[0][2]);
		}
		
		// Convert color palette to VUYA format for rendering
		float paletteV[8], paletteU[8], paletteY[8];
		for (int i = 0; i < 8; ++i)
		{
			const float r = colorPalette[i][0];
			const float g = colorPalette[i][1];
			const float b = colorPalette[i][2];
			paletteY[i] = r * 0.299f + g * 0.587f + b * 0.114f;
			paletteU[i] = r * -0.168736f + g * -0.331264f + b * 0.5f;
			paletteV[i] = r * 0.5f + g * -0.418688f + b * -0.081312f;
		}
		
		// Default line color (for compatibility, use first palette color)
		const float lineR = colorPalette[0][0];
		const float lineG = colorPalette[0][1];
		const float lineB = colorPalette[0][2];
		const float lineYVal = paletteY[0];
		const float lineUVal = paletteU[0];
		const float lineVVal = paletteV[0];
		(void)lineR; (void)lineG; (void)lineB;
		(void)lineYVal; (void)lineUVal; (void)lineVVal;

		const float lineThickness = (float)params[OST_WINDYLINES_LINE_THICKNESS]->u.fs_d.value;
		const float lineLength = (float)params[OST_WINDYLINES_LINE_LENGTH]->u.fs_d.value;
		const int lineCap = normalizePopup(params[OST_WINDYLINES_LINE_CAP]->u.pd.value, 2);
		const float lineAngle = (float)FIX_2_FLOAT(params[OST_WINDYLINES_LINE_ANGLE]->u.ad.value);
		const float lineAA = (float)params[OST_WINDYLINES_LINE_AA]->u.fs_d.value;
		
		// Spawn Source: if "Full Frame" selected, ignore alpha threshold
		const int spawnSource = normalizePopup(params[OST_WINDYLINES_SPAWN_SOURCE]->u.pd.value, 2);
		float lineAlphaThreshold = (float)params[OST_WINDYLINES_LINE_ALPHA_THRESH]->u.fs_d.value;
		if (spawnSource == SPAWN_SOURCE_FULL_FRAME) {
			lineAlphaThreshold = 1.0f;  // Full frame: ignore alpha, spawn everywhere
		}
		const int lineOriginMode = normalizePopup(params[OST_WINDYLINES_LINE_ORIGIN_MODE]->u.pd.value, 3);
		const float dsx = (in_data->downsample_x.den != 0) ? ((float)in_data->downsample_x.num / (float)in_data->downsample_x.den) : 1.0f;
		const float dsy = (in_data->downsample_y.den != 0) ? ((float)in_data->downsample_y.num / (float)in_data->downsample_y.den) : 1.0f;
		const float dsMax = dsx > dsy ? dsx : dsy;
		const float dsScale = dsMax >= 1.0f ? (1.0f / dsMax) : (dsMax > 0.0f ? dsMax : 1.0f);
		// Note: lineThicknessScaled/lineLengthScaled/lineTravelScaled are calculated
		// after alphaBounds + linkage application below
		const float lineAAScaled = lineAA * dsScale;
		const float effectiveAA = lineAAScaled > 0.0f ? lineAAScaled : 1.0f;
		// Center is now controlled by Origin Offset X/Y only
		const int lineCount = (int)params[OST_WINDYLINES_LINE_COUNT]->u.fs_d.value;
		const float lineLifetime = (float)params[OST_WINDYLINES_LINE_LIFETIME]->u.fs_d.value;
		const float lineInterval = (float)params[OST_WINDYLINES_LINE_INTERVAL]->u.fs_d.value;
		const int lineSeed = (int)params[OST_WINDYLINES_LINE_SEED]->u.fs_d.value;
		const int lineEasing = normalizePopup(params[OST_WINDYLINES_LINE_EASING]->u.pd.value, 28);
		const float lineTravel = (float)params[OST_WINDYLINES_LINE_TRAVEL]->u.fs_d.value;
		// Note: lineTravelScaled will be recalculated after linkage application below
		const float lineTailFade = (float)params[OST_WINDYLINES_LINE_TAIL_FADE]->u.fs_d.value;
		const float lineDepthStrength = (float)params[OST_WINDYLINES_LINE_DEPTH_STRENGTH]->u.fs_d.value / 10.0f; // Normalize 0-10 to 0-1
		// allowMidPlay is now replaced by negative Start Time - kept for backward compatibility but ignored
		// const bool allowMidPlay = params[OST_WINDYLINES_LINE_ALLOW_MIDPLAY]->u.bd.value != 0;
		const bool hideElement = params[OST_WINDYLINES_HIDE_ELEMENT]->u.bd.value != 0;
		const int blendMode = NormalizePopupValue((int)params[OST_WINDYLINES_BLEND_MODE]->u.pd.value, 4);
		
		// Shadow parameters
		const bool shadowEnable = params[OST_WINDYLINES_SHADOW_ENABLE]->u.bd.value != 0;
		const float shadowColorR = (float)params[OST_WINDYLINES_SHADOW_COLOR]->u.cd.value.red / 255.0f;
		const float shadowColorG = (float)params[OST_WINDYLINES_SHADOW_COLOR]->u.cd.value.green / 255.0f;
		const float shadowColorB = (float)params[OST_WINDYLINES_SHADOW_COLOR]->u.cd.value.blue / 255.0f;
		// Convert shadow color to YUV
		const float shadowY = shadowColorR * 0.299f + shadowColorG * 0.587f + shadowColorB * 0.114f;
		const float shadowU = shadowColorR * -0.168736f + shadowColorG * -0.331264f + shadowColorB * 0.5f;
		const float shadowV = shadowColorR * 0.5f + shadowColorG * -0.418688f + shadowColorB * -0.081312f;
		const float shadowOffsetX = (float)params[OST_WINDYLINES_SHADOW_OFFSET_X]->u.fs_d.value * dsScale;
		const float shadowOffsetY = (float)params[OST_WINDYLINES_SHADOW_OFFSET_Y]->u.fs_d.value * dsScale;
		const float shadowOpacity = (float)params[OST_WINDYLINES_SHADOW_OPACITY]->u.fs_d.value;
		
		// Motion Blur parameters
		const bool motionBlurEnable = params[OST_WINDYLINES_MOTION_BLUR_ENABLE]->u.bd.value != 0;
		const int motionBlurSamples = (int)params[OST_WINDYLINES_MOTION_BLUR_SAMPLES]->u.fs_d.value;
		const float motionBlurStrength = (float)params[OST_WINDYLINES_MOTION_BLUR_STRENGTH]->u.fs_d.value;
		
		// Focus (Depth of Field) parameters
		// Focus parameters removed
		const float spawnScaleX = (float)params[OST_WINDYLINES_LINE_SPAWN_SCALE_X]->u.fs_d.value / 100.0f;
		const float spawnScaleY = (float)params[OST_WINDYLINES_LINE_SPAWN_SCALE_Y]->u.fs_d.value / 100.0f;
		const float spawnRotationDeg = (float)params[OST_WINDYLINES_LINE_SPAWN_ROTATION]->u.fs_d.value;
		const float spawnRotationRad = spawnRotationDeg * 3.14159265f / 180.0f;
		const float spawnCos = FastCos(spawnRotationRad);
		const float spawnSin = FastSin(spawnRotationRad);
		const bool showSpawnArea = params[OST_WINDYLINES_LINE_SHOW_SPAWN_AREA]->u.bd.value != 0;
		const PF_Pixel spawnAreaColorPx = params[OST_WINDYLINES_LINE_SPAWN_AREA_COLOR]->u.cd.value;
		const float spawnAreaColorR = spawnAreaColorPx.red / 255.0f;
		const float spawnAreaColorG = spawnAreaColorPx.green / 255.0f;
		const float spawnAreaColorB = spawnAreaColorPx.blue / 255.0f;
		// Convert spawn area color to YUV
		const float spawnAreaY = spawnAreaColorR * 0.299f + spawnAreaColorG * 0.587f + spawnAreaColorB * 0.114f;
		const float spawnAreaU = spawnAreaColorR * -0.168736f + spawnAreaColorG * -0.331264f + spawnAreaColorB * 0.5f;
		const float spawnAreaV = spawnAreaColorR * 0.5f + spawnAreaColorG * -0.418688f + spawnAreaColorB * -0.081312f;
		// CPU rendering uses current_time which is clip-relative time.
		// This ensures cache consistency - same clip frame = same result.
		const A_long clipTime = in_data->current_time; // Clip-relative time
		const A_long frameIndex = (in_data->time_step != 0) ? (clipTime / in_data->time_step) : 0;

		const float angleRadians = (float)(M_PI / 180) * lineAngle;
		const float lineCos = FastCos(angleRadians);
		const float lineSin = FastSin(angleRadians);
		// Optimized branchless saturate using fminf/fmaxf
		auto saturate = [](float v) { return fminf(fmaxf(v, 0.0f), 1.0f); };

		// Use local state for stateless rendering (no global caching)
		LineInstanceState localLineState;
		LineInstanceState* lineState = &localLineState;

		// Branchless clamp for line count
		const int clampedLineCount = (int)fminf(fmaxf((float)lineCount, 1.0f), 5000.0f);
		const bool isLicensedThisFrame = IsLicenseAuthenticated();
		const int intervalFrames = lineInterval < 0.5f ? 0 : (int)(lineInterval + 0.5f);
		
		// Generate line params locally each frame for stateless rendering
		lineState->lineCount = clampedLineCount;
		lineState->lineSeed = lineSeed;
		lineState->lineDepthStrength = lineDepthStrength;
		lineState->lineInterval = intervalFrames;
		lineState->lineParams.assign(clampedLineCount, {});
		for (int i = 0; i < clampedLineCount; ++i)
		{
			const csSDK_uint32 base = (csSDK_uint32)(lineSeed * 1315423911u) + (csSDK_uint32)i * 2654435761u;
			const float rx = Rand01(base + 1);
			const float ry = Rand01(base + 2);
			const float rstart = Rand01(base + 5);
			const float rdepth = Rand01(base + 6);
			const float depthScale = DepthScale(rdepth, lineDepthStrength);

			LineParams lp;
			lp.posX = rx;
			lp.posY = ry;
			// Store raw depth-scaled values as multipliers; actual scaled values
			// will be applied after linkage calculation (which needs alphaBounds first)
			lp.baseLen = depthScale;   // Temporary: store depthScale, will be recalculated
			lp.baseThick = depthScale; // Temporary: store depthScale, will be recalculated
			lp.angle = lineAngle;
			const float life = lineLifetime > 1.0f ? lineLifetime : 1.0f;
			const float interval = intervalFrames > 0 ? (float)intervalFrames : 0.0f;
			const float period = life + interval;
			float startFrame = rstart * period; // Sequence-time based, no offset
			lp.startFrame = startFrame;
			lp.depthScale = depthScale;
			lp.depthValue = rdepth;
			lineState->lineParams[i] = lp;
		}

		lineState->lineDerived.assign(lineState->lineCount, {});
		lineState->lineActive.assign(lineState->lineCount, 0);

		const float life = lineLifetime > 1.0f ? lineLifetime : 1.0f;
		const float interval = intervalFrames > 0 ? (float)intervalFrames : 0.0f;
		const float period = life + interval;
		// No center offset - use Origin Offset X/Y instead
		const float centerOffsetX = 0.0f;
		const float centerOffsetY = 0.0f;
		const float alphaThreshold = lineAlphaThreshold;
		const int alphaStride = 4;
		int alphaMinX = output->width;
		int alphaMinY = output->height;
		int alphaMaxX = -1;
		int alphaMaxY = -1;
		for (int y = 0; y < output->height; y += alphaStride)
		{
			const float* row = (const float*)(srcData + y * src->rowbytes);
			for (int x = 0; x < output->width; x += alphaStride)
			{
				const float aSample = row[x * 4 + 3];
				if (aSample > alphaThreshold)
				{
					if (x < alphaMinX) alphaMinX = x;
					if (y < alphaMinY) alphaMinY = y;
					if (x > alphaMaxX) alphaMaxX = x;
					if (y > alphaMaxY) alphaMaxY = y;
				}

			}
		}
		if (alphaMaxX < alphaMinX || alphaMaxY < alphaMinY)
		{
			alphaMinX = 0;
			alphaMinY = 0;
			alphaMaxX = output->width > 0 ? (output->width - 1) : 0;
			alphaMaxY = output->height > 0 ? (output->height - 1) : 0;
		}
		const float alphaBoundsMinX = (float)alphaMinX + centerOffsetX;
		const float alphaBoundsMinY = (float)alphaMinY + centerOffsetY;
		const float alphaBoundsWidth = (float)(alphaMaxX - alphaMinX + 1);
		const float alphaBoundsHeight = (float)(alphaMaxY - alphaMinY + 1);
		const float alphaBoundsWidthSafe = alphaBoundsWidth > 0.0f ? alphaBoundsWidth : (float)output->width;
		const float alphaBoundsHeightSafe = alphaBoundsHeight > 0.0f ? alphaBoundsHeight : (float)output->height;
		
		// Apply linkage using spawn area bounds (matches GPU behavior)
		// CPU popup values are 1-based, normalize to 0-based LINKAGE_MODE_* constants
		const int lengthLinkage = normalizePopup(params[OST_WINDYLINES_LENGTH_LINKAGE]->u.pd.value, 3);
		const float lengthLinkageRate = (float)params[OST_WINDYLINES_LENGTH_LINKAGE_RATE]->u.fs_d.value / 100.0f;
		const int thicknessLinkage = normalizePopup(params[OST_WINDYLINES_THICKNESS_LINKAGE]->u.pd.value, 3);
		const float thicknessLinkageRate = (float)params[OST_WINDYLINES_THICKNESS_LINKAGE_RATE]->u.fs_d.value / 100.0f;
		const int travelLinkage = normalizePopup(params[OST_WINDYLINES_TRAVEL_LINKAGE]->u.pd.value, 3);
		const float travelLinkageRate = (float)params[OST_WINDYLINES_TRAVEL_LINKAGE_RATE]->u.fs_d.value / 100.0f;
		
		const float lineLengthScaledFinal = ApplyLinkageValue(
			lineLength,
			lengthLinkage,
			lengthLinkageRate,
			alphaBoundsWidthSafe,
			alphaBoundsHeightSafe,
			dsScale);
		const float lineThicknessScaledFinal_temp = ApplyLinkageValue(
			lineThickness,
			thicknessLinkage,
			thicknessLinkageRate,
			alphaBoundsWidthSafe,
			alphaBoundsHeightSafe,
			dsScale);
		const float lineThicknessScaledFinal = lineThicknessScaledFinal_temp < 1.0f ? 1.0f : lineThicknessScaledFinal_temp;
		const float lineTravelScaled = ApplyLinkageValue(
			lineTravel,
			travelLinkage,
			travelLinkageRate,
			alphaBoundsWidthSafe,
			alphaBoundsHeightSafe,
			dsScale);
		
		// Now recalculate lineParams baseLen/baseThick with final linkage-applied values
		for (int i = 0; i < lineState->lineCount; ++i)
		{
			LineParams& lp = lineState->lineParams[i];
			const float depthScale = lp.depthScale; // Was stored during initial loop
			const float baseLenTemp = lineLengthScaledFinal * depthScale;
			const float baseThickTemp = lineThicknessScaledFinal * depthScale;
			lp.baseLen = baseLenTemp < 1.0f ? 1.0f : baseLenTemp;
			lp.baseThick = baseThickTemp < 1.0f ? 1.0f : baseThickTemp;
		}
	
	// Start Time + Duration: control when lines spawn
	const float lineStartTime = (float)params[OST_WINDYLINES_LINE_START_TIME]->u.fs_d.value;
	const float lineDuration = (float)params[OST_WINDYLINES_LINE_DURATION]->u.fs_d.value;
	// Calculate effective end time (0 duration = infinite)
	const float lineEndTime = (lineDuration > 0.0f) ? (lineStartTime + lineDuration) : 0.0f;
	
	// Use frameIndex directly for sequence-time based rendering.
	const float timeFramesBase = (float)frameIndex;
	
	// DEBUG: Log CPU time values (first 5 frames only)
	{
		static int sCpuDebugCount = 0;
		if (sCpuDebugCount < 5)
		{
			sCpuDebugCount++;
			DebugLog("CPU TIME DEBUG: current_time=%ld time_step=%ld frameIndex=%ld lineStartTime=%.1f timeFramesBase=%.1f",
				(long)in_data->current_time, (long)in_data->time_step, (long)frameIndex, lineStartTime, timeFramesBase);
		}
	}
	
	// Origin Offset X/Y (px) - 線の起点のオフセット
	// Apply downsample scale to origin offsets (user inputs in full-resolution pixels)
	const float userOriginOffsetX = (float)params[OST_WINDYLINES_ORIGIN_OFFSET_X]->u.fs_d.value * dsScale;
	const float userOriginOffsetY = (float)params[OST_WINDYLINES_ORIGIN_OFFSET_Y]->u.fs_d.value * dsScale;
	
	// Animation Pattern (1=Simple, 2=Half Reverse, 3=Split)
	const int animPattern = params[OST_WINDYLINES_ANIM_PATTERN]->u.pd.value;
	const float centerGap = (float)params[OST_WINDYLINES_CENTER_GAP]->u.fs_d.value;
	
	for (int i = 0; lineState && i < lineState->lineCount; ++i)
		{
			const LineParams& lp = lineState->lineParams[i];
			const float timeFrames = timeFramesBase;
			// Note: allowMidPlay functionality is now handled by negative Start Time
			// Start Time < 0 allows lines to appear mid-animation at clip start
			float age = fmodf(timeFrames - lp.startFrame, period);
			if (age < 0.0f)
			{
				age += period;
			}
			
			// Start Time + End Time support: skip cycles outside the active time range
			{
				// Calculate when this cycle started
				const float cycleStartFrame = timeFrames - age;
				// Skip if this cycle started before startTime
				if (cycleStartFrame < lineStartTime)
				{
					continue;
				}
				// Skip if endTime is set and this cycle started after endTime
				if (lineDuration > 0.0f && cycleStartFrame >= lineEndTime)
				{
					continue;
				}
			}
			
			if (age > life)
			{
				continue;
			}
		const float t = age / life;
		const float tMove = ApplyEasingLUT(t, lineEasing);
		(void)tMove;  // Unused, kept for compatibility
		const float maxLen = lp.baseLen;
		const float travelRange = lineTravelScaled * lp.depthScale;
		
		// "Head extends from tail, then tail retracts" animation (matches GPU logic)????
		// Total travel distance includes line length for proper appearance/disappearance
		const float totalTravelDist = travelRange + maxLen;  // Total distance for full animation
		const float tailStartPos = -0.5f * travelRange - maxLen;  // Start hidden on left
		
		const float travelT = ApplyEasingLUT(t, lineEasing);
		const float currentTravelPos = tailStartPos + totalTravelDist * travelT;
		
		float headPosX, tailPosX, currentLength;
		
		if (t <= 0.5f)
		{
			// First half: tail at current travel position, head extends from it
			const float extendT = ApplyEasingLUT(t * 2.0f, lineEasing);
			tailPosX = currentTravelPos;
			headPosX = tailPosX + maxLen * extendT;
			currentLength = maxLen * extendT;
		}
		else
		{
			// Second half: head at current travel position + maxLen, tail retracts toward it
			const float retractT = ApplyEasingLUT((t - 0.5f) * 2.0f, lineEasing);
			headPosX = currentTravelPos + maxLen;
			tailPosX = headPosX - maxLen * (1.0f - retractT);
			currentLength = maxLen * (1.0f - retractT);
		}
		
		// For rendering: center = midpoint between head and tail
		const float segCenterX = (headPosX + tailPosX) * 0.5f;
		
		// Appear/Disappear scale + fade: smooth fade-in at start, fade-out at end
		// Uses easeOutCubic for appear (fast start, very slow end - more natural)
		// Uses easeInCubic for disappear (very slow start, fast end - more natural)
		const float appearDuration = 0.10f;   // 10% of lifetime for appear
		const float disappearDuration = 0.10f; // 10% of lifetime for disappear
		float appearScale = 1.0f;
		float appearAlpha = 1.0f;
		if (t < appearDuration)
		{
			// Appear: easeOutCubic (1 - (1-t)^3) - starts fast, ends very smoothly
			const float at = t / appearDuration;
			const float inv = 1.0f - at;
			const float eased = 1.0f - inv * inv * inv;
			appearScale = eased;
			appearAlpha = eased;  // Fade in with scale
		}
		else if (t > (1.0f - disappearDuration))
		{
			// Disappear: easeInCubic (t^3) - starts very smoothly, ends fast
			const float dt = (1.0f - t) / disappearDuration;
			const float eased = dt * dt * dt;
			appearScale = eased;
			appearAlpha = eased;  // Fade out with scale
		}
		
		const float halfLen = currentLength * 0.5f * appearScale;

		// Skip only if line has zero length (GPU doesn't skip on thickness)
		const bool isTiny = (halfLen < 0.01f && appearScale < 0.01f);
		lineState->lineActive[i] = isTiny ? 0 : 1;
		if (isTiny)
		{
			continue;
		}

		// Wind Origin: adjust spawn area position (overall atmosphere, not per-line animation)
		// Apply offset in the direction of line angle (both X and Y components)
		// Use maxLen*0.5 (max possible halfLen) for conservative compensation
		// Note: some minor protrusion is inherent in head/tail animation
		const float maxHalfLen = maxLen * 0.5f;
		float originOffset = 0.0f;
		if (lineOriginMode == 1)  // Forward
		{
			originOffset = 0.5f * travelRange + maxHalfLen;
		}
		else if (lineOriginMode == 2)  // Backward
		{
			originOffset = -(0.5f * travelRange + maxHalfLen);
		}

		// Animation Pattern adjustments
		// Pattern 1: Simple - all same direction
		// Pattern 2: Half Reverse - every other line reversed
		// Pattern 3: Split - sides go opposite directions (angle-linked)
		// Center Gap applies to all patterns when > 0
		
		float adjustedPosX = lp.posX;
		float adjustedPosY = lp.posY;
		float adjustedAngle = lineAngle;
		
		// Calculate perpendicular axis for center gap and Split pattern (aspect-corrected)
		const float invW = alphaBoundsWidth > 0.0f ? (1.0f / alphaBoundsWidth) : 1.0f;
		const float invH = alphaBoundsHeight > 0.0f ? (1.0f / alphaBoundsHeight) : 1.0f;
		const float dirX = lineCos * invW;
		const float dirY = lineSin * invH;
		float perpX = -dirY;  // Perpendicular to movement direction
		float perpY = dirX;
		const float perpLen = sqrtf(perpX * perpX + perpY * perpY);
		if (perpLen > 0.00001f)
		{
			perpX /= perpLen;
			perpY /= perpLen;
		}
		const float sideValue = (lp.posX - 0.5f) * perpX + (lp.posY - 0.5f) * perpY;
		
		// Apply center gap (hide lines in center zone)
		if (centerGap > 0.0f && sideValue > -centerGap && sideValue < centerGap)
		{
			// Center zone - hide line
			adjustedPosX = -10.0f;
			adjustedPosY = -10.0f;
		}
		else
		{
			// Pattern-specific direction adjustments
			if (animPattern == 2)  // Half Reverse: 50% of lines go opposite direction
			{
				if (i % 2 == 1)
				{
					adjustedAngle = lineAngle + 180.0f;
				}
			}
			else if (animPattern == 3)  // Split: sides go opposite directions
			{
				if (sideValue < 0.0f)
				{
					adjustedAngle = lineAngle + 180.0f;  // Negative side flows opposite
				}
				// Positive side keeps original angle
			}
			// animPattern == 1 (Simple): no direction adjustment
		}
		
		const float adjustedCos = FastCos(adjustedAngle * 3.14159265f / 180.0f);
		const float adjustedSin = FastSin(adjustedAngle * 3.14159265f / 180.0f);

		LineDerived ld;
		const float alphaCenterX = alphaBoundsMinX + alphaBoundsWidth * 0.5f;
		const float alphaCenterY = alphaBoundsMinY + alphaBoundsHeight * 0.5f;
		// Apply spawn rotation to the spawn position offset
		const float spawnOffsetX = (adjustedPosX - 0.5f) * alphaBoundsWidth * spawnScaleX;
		const float spawnOffsetY = (adjustedPosY - 0.5f) * alphaBoundsHeight * spawnScaleY;
		const float rotatedSpawnX = spawnOffsetX * spawnCos - spawnOffsetY * spawnSin;
		const float rotatedSpawnY = spawnOffsetX * spawnSin + spawnOffsetY * spawnCos;
		ld.centerX = alphaCenterX + rotatedSpawnX + originOffset * adjustedCos + userOriginOffsetX;
		ld.centerY = alphaCenterY + rotatedSpawnY + originOffset * adjustedSin + userOriginOffsetY;
		ld.cosA = adjustedCos;
		ld.sinA = adjustedSin;
		ld.halfLen = halfLen;
		ld.segCenterX = segCenterX;
		ld.depth = lp.depthValue;  // Store depth value for consistent blend mode

		// Focus (Depth of Field) disabled
		// Don't multiply by appearScale - GPU handles appear/disappear via alpha only,
		// not by physical size reduction. This prevents thin lines from disappearing.
		ld.halfThick = lp.baseThick * 0.5f;
		ld.focusAlpha = 1.0f;
		ld.appearAlpha = appearAlpha;  // Appear/disappear fade alpha
		ld.lineVelocity = ApplyEasingDerivative(t, lineEasing);  // Motion blur velocity
		
		// Select color from palette: Single mode uses 0, Preset/Custom uses random based on seed
		if (colorMode == 0)  // Single mode
		{
			ld.colorIndex = 0;
		}
		else  // Preset or Custom mode
		{
			// Use existing seed + line index for random color selection
			const csSDK_uint32 colorBase = (csSDK_uint32)(lineSeed * 1315423911u) + (csSDK_uint32)i * 2654435761u + 12345u;
			ld.colorIndex = (int)(Rand01(colorBase) * 8.0f);
			if (ld.colorIndex > 7) ld.colorIndex = 7;
		}
		
		// Pre-compute expensive per-pixel calculations once per line
		const float depthScale = DepthScale(ld.depth, lineDepthStrength);
		const float depthFadeT = saturate((depthScale - 0.2f) / (0.6f - 0.2f));
		ld.depthAlpha = 0.05f + 0.95f * depthFadeT;
		const float denom = (2.0f * ld.halfLen) > 0.0001f ? (2.0f * ld.halfLen) : 0.0001f;
		ld.invDenom = 1.0f / denom;
		ld._padding = 0;  // Initialize padding for clean memory
		
		lineState->lineDerived[i] = ld;
	}

		const int tileSize = 32;
		const int tileCountX = (output->width + tileSize - 1) / tileSize;
		const int tileCountY = (output->height + tileSize - 1) / tileSize;
		const int tileCount = tileCountX * tileCountY;

		if (lineState)
		{
			lineState->tileCounts.assign(tileCount, 0);
		}
		for (int i = 0; lineState && i < lineState->lineCount; ++i)
		{
			if (!lineState->lineActive[i])
			{
				continue;
			}
			const LineDerived& ld = lineState->lineDerived[i];
			const float radius = fabsf(ld.segCenterX) + ld.halfLen + ld.halfThick + lineAAScaled;
			int minX = (int)((ld.centerX - radius) / tileSize);
			int maxX = (int)((ld.centerX + radius) / tileSize);
			int minY = (int)((ld.centerY - radius) / tileSize);
			int maxY = (int)((ld.centerY + radius) / tileSize);
			if (minX < 0) minX = 0;
			if (minY < 0) minY = 0;
			if (maxX >= tileCountX) maxX = tileCountX - 1;
			if (maxY >= tileCountY) maxY = tileCountY - 1;
			for (int ty = minY; ty <= maxY; ++ty)
			{
				for (int tx = minX; tx <= maxX; ++tx)
				{
					lineState->tileCounts[ty * tileCountX + tx] += 1;
				}
			}
		}

		if (lineState)
		{
			lineState->tileOffsets.assign(tileCount + 1, 0);
			for (int i = 0; i < tileCount; ++i)
			{
				lineState->tileOffsets[i + 1] = lineState->tileOffsets[i] + lineState->tileCounts[i];
			}
		}
		if (lineState)
		{
			lineState->tileIndices.assign(lineState->tileOffsets[tileCount], 0);
			std::vector<int> tileCursor = lineState->tileOffsets;
			for (int i = 0; i < lineState->lineCount; ++i)
			{
				if (!lineState->lineActive[i])
				{
					continue;
				}
				const LineDerived& ld = lineState->lineDerived[i];
				const float radius = fabsf(ld.segCenterX) + ld.halfLen + ld.halfThick + lineAAScaled;
				int minX = (int)((ld.centerX - radius) / tileSize);
				int maxX = (int)((ld.centerX + radius) / tileSize);
				int minY = (int)((ld.centerY - radius) / tileSize);
				int maxY = (int)((ld.centerY + radius) / tileSize);
				if (minX < 0) minX = 0;
				if (minY < 0) minY = 0;
				if (maxX >= tileCountX) maxX = tileCountX - 1;
				if (maxY >= tileCountY) maxY = tileCountY - 1;
				for (int ty = minY; ty <= maxY; ++ty)
				{
					for (int tx = minX; tx <= maxX; ++tx)
					{
						const int idx = ty * tileCountX + tx;
						lineState->tileIndices[tileCursor[idx]++] = i;
					}
				}
			}
		}

		// Pre-compute motion blur frame constants (avoid recalculating per-pixel per-line)
		const float mbShutterFraction = motionBlurStrength / 360.0f;
		const float mbPixelsPerFrame = (lineLifetime > 0.0f) ? (lineTravel / lineLifetime) : 0.0f;

		// Pre-compute watermark frame constants (avoid recalculating per-pixel)
		const int wmTextWidthPx = FreeModeWatermark::TextWidthPx();
		const int wmTextHeightPx = FreeModeWatermark::TextHeightPx();
		const int wmMarginX = FreeModeWatermark::kMarginX;
		const int wmMarginY = FreeModeWatermark::kMarginY;
		const float wmScale = dsScale > 0.0f ? dsScale : 1.0f;
		const int wmScaledMarginX = std::max(0, static_cast<int>(std::lround(static_cast<float>(wmMarginX) * wmScale)));
		const int wmScaledMarginY = std::max(0, static_cast<int>(std::lround(static_cast<float>(wmMarginY) * wmScale)));
		const int wmScaledWidthPx = std::max(1, static_cast<int>(std::lround(static_cast<float>(wmTextWidthPx) * wmScale)));
		const int wmScaledHeightPx = std::max(1, static_cast<int>(std::lround(static_cast<float>(wmTextHeightPx) * wmScale)));
		const int wmStartX = std::max(0, output->width - wmScaledMarginX - wmScaledWidthPx);

		// Pre-compute spawn area preview constants (avoid recalculating per-pixel)
		const float saAlphaCenterX = alphaBoundsMinX + alphaBoundsWidth * 0.5f;
		const float saAlphaCenterY = alphaBoundsMinY + alphaBoundsHeight * 0.5f;
		const float saHalfW = alphaBoundsWidth * spawnScaleX * 0.5f;
		const float saHalfH = alphaBoundsHeight * spawnScaleY * 0.5f;

		// Main render loop - compiler optimization hints
#if defined(__clang__)
#pragma clang loop unroll_count(4)
#endif
		for (int y = 0; y < output->height; ++y, srcData += src->rowbytes, destData += dest->rowbytes)
		{
			for (int x = 0; x < output->width; ++x)
			{
				float v, u, luma, a;
				if (hideElement)
				{
					// Hide original element - start with transparent black
					v = 0.0f;
					u = 0.0f;
					luma = 0.0f;
					a = 0.0f;
				}
				else
				{
					v = ((const float*)srcData)[x * 4 + 0];
					u = ((const float*)srcData)[x * 4 + 1];
					luma = ((const float*)srcData)[x * 4 + 2];
					a = ((const float*)srcData)[x * 4 + 3];
				}

				float outV = v;
				float outU = u;
				float outY = luma;
				const float originalAlpha = a;  // Save original alpha for blend modes
				const float originalV = v;  // Save original color for Alpha XOR mode
				const float originalU = u;
				const float originalY = luma;
				// Accumulate front lines separately for blend mode 2
				float frontV = 0.0f;
				float frontU = 0.0f;
				float frontY = 0.0f;
				float frontA = 0.0f;
				float frontAppearAlpha = 1.0f;
				
				// Track line-only alpha for Alpha XOR mode (blend mode 3)
				float lineOnlyAlpha = 0.0f;

				const int tileX = x / tileSize;
				const int tileY = y / tileSize;
				const int tileIndex = tileY * tileCountX + tileX;
				const int start = lineState ? lineState->tileOffsets[tileIndex] : 0;
				const int count = lineState ? lineState->tileCounts[tileIndex] : 0;
				// Main line pass (with shadow drawn first for each line)
				for (int i = 0; i < count; ++i)
				{
					const LineDerived& ld = lineState->lineDerived[lineState->tileIndices[start + i]];
					const float depthAlpha = ld.depthAlpha;  // Use pre-computed

					// === STEP 1-4: Skip extremely tiny lines ===
					// Use very low threshold - GPU/CUDA kernel has no thickness skip
					if (ld.halfThick < 0.01f)
					{
						continue;  // Invisible line - skip all processing
					}

					// === STEP 2: Early skip optimization ===
					// Check if pixel is outside line bounding box (with shadow offset margin)
					const float skipDx = (x + 0.5f) - ld.centerX;
					const float skipDy = (y + 0.5f) - ld.centerY;
					const float shadowMargin = shadowEnable ? fmaxf(fabsf(shadowOffsetX), fabsf(shadowOffsetY)) : 0.0f;
					const float margin = ld.halfThick + lineAAScaled + shadowMargin;
					const float skipPx = skipDx * ld.cosA + skipDy * ld.sinA - ld.segCenterX;
					const float skipPy = -skipDx * ld.sinA + skipDy * ld.cosA;
					
					if (fabsf(skipPx) > ld.halfLen + margin && fabsf(skipPy) > margin)
					{
						continue;  // Skip this line - pixel is too far away
					}
					// === END STEP 2 ===
					
				// Draw shadow first (before the line) with motion blur
				if (shadowEnable)
				{
					const float sdx = (x + 0.5f) - (ld.centerX + shadowOffsetX);
					const float sdy = (y + 0.5f) - (ld.centerY + shadowOffsetY);
					float scoverage = 0.0f;

					// Motion blur for shadow (matching OpenCL/Metal implementation)
					if (motionBlurEnable && motionBlurSamples > 1)
					{
						const int samples = motionBlurSamples;
						const float effectiveVelocity = mbPixelsPerFrame * ld.lineVelocity;
						const float blurRange = effectiveVelocity * mbShutterFraction;
						if (blurRange > 0.5f)
						{
							float saccumA = 0.0f;

							for (int s = 0; s < samples; ++s)
							{
								const float t = (float)s / fmaxf((float)(samples - 1), 1.0f);
								const float sampleOffset = blurRange * t;

								float spxSample = sdx * ld.cosA + sdy * ld.sinA;
								const float spySample = -sdx * ld.sinA + sdy * ld.cosA;
								spxSample -= (ld.segCenterX + sampleOffset);

								float sdistSample = (lineCap == 0)
									? SDFBox(spxSample, spySample, ld.halfLen, ld.halfThick)
									: SDFCapsule(spxSample, spySample, ld.halfLen, ld.halfThick);

								const float stailT = saturate((spxSample + ld.halfLen) * ld.invDenom);  // Use pre-computed
								const float stailFade = 1.0f + (stailT - 1.0f) * lineTailFade;
								const float saa = effectiveAA;
								float sampleCoverage = 0.0f;
								if (saa > 0.0f)
								{
									const float tt = saturate((sdistSample - saa) / (0.0f - saa));
									sampleCoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade;
								}

								saccumA += sampleCoverage;
							}

							scoverage = (saccumA / (float)samples) * shadowOpacity * depthAlpha;
						}
						else
						{
							// Blur too small, use single sample
							float spx = sdx * ld.cosA + sdy * ld.sinA;
							const float spy = -sdx * ld.sinA + sdy * ld.cosA;
							spx -= ld.segCenterX;

							float sdist = (lineCap == 0)
								? SDFBox(spx, spy, ld.halfLen, ld.halfThick)
								: SDFCapsule(spx, spy, ld.halfLen, ld.halfThick);

							const float stailT = saturate((spx + ld.halfLen) * ld.invDenom);  // Use pre-computed
							const float stailFade = 1.0f + (stailT - 1.0f) * lineTailFade;
							const float saa = effectiveAA;
							if (saa > 0.0f)
							{
								const float tt = saturate((sdist - saa) / (0.0f - saa));
								scoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * shadowOpacity * depthAlpha;
							}
						}
					}
					else
					{
						// No motion blur - single sample
						float spx = sdx * ld.cosA + sdy * ld.sinA;
						const float spy = -sdx * ld.sinA + sdy * ld.cosA;
						spx -= ld.segCenterX;

						float sdist = (lineCap == 0)
							? SDFBox(spx, spy, ld.halfLen, ld.halfThick)
							: SDFCapsule(spx, spy, ld.halfLen, ld.halfThick);

						const float stailT = saturate((spx + ld.halfLen) * ld.invDenom);  // Use pre-computed
						const float stailFade = 1.0f + (stailT - 1.0f) * lineTailFade;
						const float saa = effectiveAA;
						if (saa > 0.0f)
						{
							const float tt = saturate((sdist - saa) / (0.0f - saa));
							scoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * shadowOpacity * depthAlpha;
						}
					}
						if (scoverage > 0.001f)
						{
							float shadowBlend = scoverage;
							if (blendMode == 0)
							{
								// Back: keep shadow behind the original element
								shadowBlend = scoverage * (1.0f - originalAlpha);
							}
							else if (blendMode == 2 && ld.depth < 0.5f)
							{
								// Back portion of "Back and Front": keep shadow behind the original element
								shadowBlend = scoverage * (1.0f - originalAlpha);
							}

							// Shadow: premultiplied alpha compositing (v62 fix)
							float invShadow = 1.0f - shadowBlend;
							float outAlpha = shadowBlend + a * invShadow;
							if (outAlpha > 0.0f) {
								outY = (shadowY * shadowBlend + outY * a * invShadow) / outAlpha;
								outU = (shadowU * shadowBlend + outU * a * invShadow) / outAlpha;
								outV = (shadowV * shadowBlend + outV * a * invShadow) / outAlpha;
							}
							a = outAlpha;
						}
					}
					
					// Draw the main line with motion blur
					const float dx = (x + 0.5f) - ld.centerX;
					const float dy = (y + 0.5f) - ld.centerY;
					float coverage = 0.0f;

					// Motion blur sampling
					if (motionBlurEnable && motionBlurSamples > 1)
					{
						const int samples = motionBlurSamples;
						const float effectiveVelocity = mbPixelsPerFrame * ld.lineVelocity;
						const float blurRange = effectiveVelocity * mbShutterFraction;
	
						if (blurRange > 0.5f)
						{
							// Multi-sample motion blur with uniform averaging
							float accumA = 0.0f;

							for (int s = 0; s < samples; ++s)
							{
								const float t = (float)s / fmaxf((float)(samples - 1), 1.0f);
								const float sampleOffset = blurRange * t;

								float pxSample = dx * ld.cosA + dy * ld.sinA;
								const float pySample = -dx * ld.sinA + dy * ld.cosA;
								pxSample -= (ld.segCenterX + sampleOffset);

								float distSample = (lineCap == 0)
									? SDFBox(pxSample, pySample, ld.halfLen, ld.halfThick)
									: SDFCapsule(pxSample, pySample, ld.halfLen, ld.halfThick);

								const float tailT = saturate((pxSample + ld.halfLen) * ld.invDenom);  // Use pre-computed
								const float tailFade = 1.0f + (tailT - 1.0f) * lineTailFade;
								const float aa = effectiveAA;
								float sampleCoverage = 0.0f;
								if (aa > 0.0f)
								{
									const float tt = saturate((distSample - aa) / (0.0f - aa));
									sampleCoverage = tt * tt * (3.0f - 2.0f * tt) * tailFade;
								}

								accumA += sampleCoverage;
							}

							coverage = (accumA / (float)samples) * ld.focusAlpha * depthAlpha;
						}
						else
						{
							// Blur too small, use single sample
							float px = dx * ld.cosA + dy * ld.sinA;
							const float py = -dx * ld.sinA + dy * ld.cosA;
							px -= ld.segCenterX;

							float dist = (lineCap == 0)
								? SDFBox(px, py, ld.halfLen, ld.halfThick)
								: SDFCapsule(px, py, ld.halfLen, ld.halfThick);

							const float aa = effectiveAA;
									const float tailT = saturate((px + ld.halfLen) * ld.invDenom);  // Use pre-computed
							const float tailFade = 1.0f + (tailT - 1.0f) * lineTailFade;
							if (aa > 0.0f)
							{
								const float tt = saturate((dist - aa) / (0.0f - aa));
								coverage = tt * tt * (3.0f - 2.0f * tt) * tailFade * ld.focusAlpha * depthAlpha;
							}
						}
					}
					else
					{
						// No motion blur (single sample)
						float px = dx * ld.cosA + dy * ld.sinA;
						const float py = -dx * ld.sinA + dy * ld.cosA;
						px -= ld.segCenterX;

						float dist = (lineCap == 0)
							? SDFBox(px, py, ld.halfLen, ld.halfThick)
							: SDFCapsule(px, py, ld.halfLen, ld.halfThick);

						const float aa = effectiveAA;
							const float tailT = saturate((px + ld.halfLen) * ld.invDenom);  // Use pre-computed
						const float tailFade = 1.0f + (tailT - 1.0f) * lineTailFade;
						if (aa > 0.0f)
						{
							const float tt = saturate((dist - aa) / (0.0f - aa));
							coverage = tt * tt * (3.0f - 2.0f * tt) * tailFade * ld.focusAlpha * depthAlpha;
						}
					}
					if (coverage > 0.001f)
					{
						const int ci = ld.colorIndex >= 0 && ld.colorIndex < 8 ? ld.colorIndex : 0;
						
						// Save alpha before this line's contribution
						const float prevAlpha = a;
						(void)prevAlpha;  // Unused, kept for future use
						
						// Apply blend mode
						if (blendMode == 0)  // Back (behind element)
						{
							float srcAlpha = coverage * (1.0f - originalAlpha);
							BlendPremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
							                   outY, outU, outV, a);
						}
						else if (blendMode == 1)  // Front (in front of element)
						{
							float srcAlpha = coverage;
							BlendPremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
							                   outY, outU, outV, a);
						}
						else if (blendMode == 2)  // Back and Front (split by per-line depth)
						{
							// Use stored depth value from line data (consistent across frames)
							if (ld.depth < 0.5f)
							{
								// Back mode (full)
								float srcAlpha = coverage * (1.0f - originalAlpha);
								BlendPremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
								                   outY, outU, outV, a);
							}
							else
							{
								// Front mode (full) -> accumulate separately, apply after loop
								// Use un-premultiplied accumulation (same as CUDA/OpenCL)
								float srcAlpha = coverage;
								BlendUnpremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
								                     frontY, frontU, frontV, frontA);
								frontAppearAlpha = std::min(frontAppearAlpha, ld.appearAlpha);
							}
						}
						else if (blendMode == 3)  // Alpha (XOR with original element only)
						{
							// Line-to-line blending: premultiplied compositing
							float srcAlpha = coverage;
							BlendPremultiplied(paletteY[ci], paletteU[ci], paletteV[ci], srcAlpha,
							                   outY, outU, outV, a);
							// Track line-only alpha
							lineOnlyAlpha = std::max(lineOnlyAlpha, coverage * ld.appearAlpha);
						}
					}
				}

				// Apply XOR with original element AFTER all lines are drawn (blend mode 3)
				if (blendMode == 3 && originalAlpha > 0.0f)
				{
					// Alpha XOR mode: 
					// - Where lines exist: XOR alpha with original, show original color
					// - Where only original exists (no lines): show original element as-is
					
					if (lineOnlyAlpha > 0.0f)
					{
						// XOR alpha calculation: where overlap exists, alpha is reduced
						const float xorAlpha = saturate(originalAlpha + lineOnlyAlpha - (originalAlpha * lineOnlyAlpha * 2.0f));
						
						// For color: where original element exists, show original color
						outV = outV * (1.0f - originalAlpha) + originalV * originalAlpha;
						outU = outU * (1.0f - originalAlpha) + originalU * originalAlpha;
						outY = outY * (1.0f - originalAlpha) + originalY * originalAlpha;
						a = xorAlpha;
					}
					else
					{
						// No lines here, but original element exists - show original element
						outV = originalV;
						outU = originalU;
						outY = originalY;
						a = originalAlpha;
					}
				}

				// Apply front lines after back lines (blend mode 2)
				if (blendMode == 2 && frontA > 0.0f)
				{
				float prevAlpha = a;
				float invFrontA = 1.0f - frontA;
				float newAlpha = frontA + a * invFrontA;
				if (newAlpha > 0.0f) {
					outV = (frontV * frontA + outV * a * invFrontA) / newAlpha;
					outU = (frontU * frontA + outU * a * invFrontA) / newAlpha;
					outY = (frontY * frontA + outY * a * invFrontA) / newAlpha;
				}
				a = prevAlpha + (newAlpha - prevAlpha) * frontAppearAlpha;
			}

			// Draw spawn area preview (filled with inverted colors)
			if (showSpawnArea)
				{
					// Transform pixel position to rotated spawn space
					const float relX = (x + 0.5f) - saAlphaCenterX - userOriginOffsetX;
					const float relY = (y + 0.5f) - saAlphaCenterY - userOriginOffsetY;
					// Inverse rotate to check bounds
					const float localX = relX * spawnCos + relY * spawnSin;
					const float localY = -relX * spawnSin + relY * spawnCos;
					
					// Check if inside the spawn area (filled)
					if (fabsf(localX) <= saHalfW && fabsf(localY) <= saHalfH)
					{
						// Blend with spawn area color at 50%
						const float blendAlpha = 0.5f;
						const float baseV = (a <= 0.0f) ? spawnAreaV : outV;
						const float baseU = (a <= 0.0f) ? spawnAreaU : outU;
						const float baseY = (a <= 0.0f) ? spawnAreaY : outY;
						float blendedV = baseV + (spawnAreaV - baseV) * blendAlpha;
						float blendedU = baseU + (spawnAreaU - baseU) * blendAlpha;
						float blendedY = baseY + (spawnAreaY - baseY) * blendAlpha;
						outV = blendedV;
						outU = blendedU;
						outY = blendedY;
						a = std::max(a, blendAlpha);
					}
				}

				// License watermark (non-authenticated mode):
				// top-right with margin
				if (!isLicensedThisFrame)
				{
					if (x >= wmStartX - 1 && x < wmStartX + wmScaledWidthPx + 1 &&
						y >= wmScaledMarginY - 1 && y < wmScaledMarginY + wmScaledHeightPx + 1)
					{
						const int localX = x - wmStartX;
						const int localY = y - wmScaledMarginY;
						const int sampleX = wmMarginX + static_cast<int>(static_cast<float>(localX) / wmScale);
						const int sampleY = wmMarginY + static_cast<int>(static_cast<float>(localY) / wmScale);
						const float fillAlpha = static_cast<float>(FreeModeWatermark::FillAlphaAt(sampleX, sampleY)) / 255.0f;
						const float outlineAlpha = static_cast<float>(FreeModeWatermark::OutlineAlphaAt(sampleX, sampleY)) / 255.0f;
						const bool fill = fillAlpha > 0.0f;
						const bool outline = (!fill) && (outlineAlpha > 0.0f);

						if (fill || outline)
						{
							const float targetY = fill ? 1.0f : 0.0f;
							const float targetU = 0.0f;
							const float targetV = 0.0f;
							const float baseAlpha = fill ? FreeModeWatermark::kFillOpacity : FreeModeWatermark::kOutlineOpacity;
							const float overlayAlpha = baseAlpha * (fill ? fillAlpha : outlineAlpha);

							outY = outY + (targetY - outY) * overlayAlpha;
							outU = outU + (targetU - outU) * overlayAlpha;
							outV = outV + (targetV - outV) * overlayAlpha;
							a = std::max(a, overlayAlpha);
						}
					}
				}
			
			// Note: Premultiplied alpha compositing already produces correctly weighted colors
			// No additional premultiplication needed - output is straight alpha format
			// Premiere Pro handles the compositing correctly with straight alpha

			((float*)destData)[x * 4 + 0] = outV;
			((float*)destData)[x * 4 + 1] = outU;
			((float*)destData)[x * 4 + 2] = outY;
			((float*)destData)[x * 4 + 3] = a;

#if ENABLE_DEBUG_RENDER_MARKERS
			// DEBUG: Draw "CP" text in top-left corner (5x7 bitmap font, 4x scale)
			// Orange text on current background to indicate CPU rendering
			{
				const int fontC[7] = {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
				const int fontP[7] = {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
				const int scale = 4;
				const int baseX = 5, baseY = 5;
				int px = (x - baseX) / scale;
				int py = (y - baseY) / scale;
				if (py >= 0 && py < 7)
				{
					int charBits = 0;
					int localPx = -1;
					if (px >= 0 && px < 5) { charBits = fontC[py]; localPx = px; }
					else if (px >= 6 && px < 11) { charBits = fontP[py]; localPx = px - 6; }
					if (localPx >= 0 && ((charBits >> (4 - localPx)) & 1))
					{
						// Orange in YUV: Y=0.6, U=-0.1, V=0.25
						((float*)destData)[x * 4 + 0] = 0.25f;  // V
						((float*)destData)[x * 4 + 1] = -0.1f;  // U
						((float*)destData)[x * 4 + 2] = 0.6f;   // Y
						((float*)destData)[x * 4 + 3] = 1.0f;   // A
					}
				}
			}
#endif
		}
	}
}	return PF_Err_NONE;
}

/*
**
*/
#if _WIN32 || defined(MSWindows)
#define DllExport   __declspec( dllexport )
#else
#define DllExport	__attribute__((visibility("default")))
#endif
extern "C" DllExport PF_Err EffectMain(
	PF_Cmd inCmd,
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* inOutput,
	void* extra)
{
	PF_Err err = PF_Err_NONE;
	switch (inCmd)
	{
	case PF_Cmd_GLOBAL_SETUP:
		err = GlobalSetup(in_data, out_data, params, inOutput);
		break;
	case PF_Cmd_GLOBAL_SETDOWN:
		err = GlobalSetdown(in_data, out_data, params, inOutput);
		break;
	case PF_Cmd_PARAMS_SETUP:
		err = ParamsSetup(in_data, out_data, params, inOutput);
		break;
	case PF_Cmd_USER_CHANGED_PARAM:
	{
		PF_UserChangedParamExtra* changedExtra = reinterpret_cast<PF_UserChangedParamExtra*>(extra);
		
		DebugLog("USER_CHANGED_PARAM: changedExtra=%p, param_index=%d", 
			changedExtra, changedExtra ? changedExtra->param_index : -1);
		
		// Effect Preset: apply preset parameters or defaults
		if (changedExtra && changedExtra->param_index == OST_WINDYLINES_EFFECT_PRESET)
		{
			const int presetValue = params[OST_WINDYLINES_EFFECT_PRESET]->u.pd.value;
			DebugLog("Effect Preset changed: presetValue=%d (1=Default, 2+=Preset[n-2])", presetValue);
			
			if (presetValue == 1)
			{
				DebugLog("Applying default effect params...");
				ApplyDefaultEffectParams(in_data, out_data, params);
				DebugLog("Default effect params applied.");
			}
			else if (presetValue > 1)
			{
				// Debounce: ignore double-fire within 200ms
				const uint32_t currentTime = GetCurrentTimeMs();
				const uint32_t lastTime = sLastPresetClickTime.load();
				DebugLog("Debounce check: currentTime=%u, lastTime=%u, diff=%u, threshold=%u",
					currentTime, lastTime, currentTime - lastTime, kPresetDebounceMs);
				if (currentTime - lastTime < kPresetDebounceMs)
				{
					DebugLog("DEBOUNCE: Ignoring duplicate event");
					break;  // Ignore duplicate event
				}
				sLastPresetClickTime.store(currentTime);
				DebugLog("Applying effect preset index=%d...", presetValue - 2);
				ApplyEffectPreset(in_data, out_data, params, presetValue - 2);
				DebugLog("Effect preset applied.");
			}
			else
			{
				DebugLog("presetValue <= 0, no action taken");
			}
		}
		
		// Single Color changed: auto-switch unified preset to "単色" (value=1)
		if (changedExtra && changedExtra->param_index == OST_WINDYLINES_LINE_COLOR)
		{
			const int currentPreset = params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value;
			if (currentPreset != 1)  // 1 = 単色 (1-based)
			{
				params[OST_WINDYLINES_COLOR_PRESET]->u.pd.value = 1;
				params[OST_WINDYLINES_COLOR_PRESET]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
				AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
				if (paramUtils.get())
				{
					paramUtils->PF_UpdateParamUI(in_data->effect_ref, OST_WINDYLINES_COLOR_PRESET, params[OST_WINDYLINES_COLOR_PRESET]);
				}
				out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
				DebugLog("Single color changed: Auto-switched unified preset to 単色 (value=1)");
			}
		}
		
		// Spawn Source: enable/disable Alpha Threshold based on selection
		if (changedExtra && changedExtra->param_index == OST_WINDYLINES_SPAWN_SOURCE)
		{
			// Update Alpha Threshold visibility
			UpdateAlphaThresholdVisibility(in_data, params);
			out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
		}
		
		// Linkage parameters: update visibility when linkage mode changes
		if (changedExtra && (
			changedExtra->param_index == OST_WINDYLINES_TRAVEL_LINKAGE ||
			changedExtra->param_index == OST_WINDYLINES_THICKNESS_LINKAGE ||
			changedExtra->param_index == OST_WINDYLINES_LENGTH_LINKAGE))
		{
			DebugLog("Linkage parameter changed: param_index=%d", changedExtra->param_index);
			UpdatePseudoGroupVisibility(in_data, params);
			out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
		}
		
		// Color Mode: update custom colors visibility
		if (changedExtra && changedExtra->param_index == OST_WINDYLINES_COLOR_MODE)
		{
			DebugLog("Color Mode parameter changed");
			UpdatePseudoGroupVisibility(in_data, params);
			out_data->out_flags |= PF_OutFlag_FORCE_RERENDER | PF_OutFlag_REFRESH_UI;
		}
		
		// License Status popup: "Activate" selected (value 2)
		if (changedExtra && changedExtra->param_index == OST_WINDYLINES_LICENSE_STATUS)
		{
			const int licenseValue = params[OST_WINDYLINES_LICENSE_STATUS]->u.pd.value;
			DebugLog("[License] License status popup changed: value=%d", licenseValue);
			if (licenseValue == 2) // "アクティベート..."
			{
				// Open browser for activation
				OpenActivationPage();

				// Start post-activation rapid check mode (5s interval for 2 minutes)
				sActivationTriggeredAtMs.store(GetCurrentTimeMs(), std::memory_order_relaxed);
				sLastAutoRefreshAttemptMs.store(0, std::memory_order_relaxed); // allow immediate refresh
				RefreshLicenseAuthenticatedState(true); // force immediate check
				
				// Reset popup back to "選択してください" (value 1)
				params[OST_WINDYLINES_LICENSE_STATUS]->u.pd.value = 1;
				params[OST_WINDYLINES_LICENSE_STATUS]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
				AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
				if (paramUtils.get())
				{
					paramUtils->PF_UpdateParamUI(in_data->effect_ref, OST_WINDYLINES_LICENSE_STATUS, params[OST_WINDYLINES_LICENSE_STATUS]);
				}
			}
		}
		
		ApplyRectColorUi(in_data, out_data, params);
		SyncLineColorParams(params);
		HideLineColorParams(in_data);
		UpdateAlphaThresholdVisibility(in_data, params);
		UpdatePseudoGroupVisibility(in_data, params);
	}
		break;
	case PF_Cmd_UPDATE_PARAMS_UI:
	{
		// Update UI state for all parameters
		ApplyRectColorUi(in_data, out_data, params);
		SyncLineColorParams(params);
		HideLineColorParams(in_data);
		UpdateAlphaThresholdVisibility(in_data, params);
		UpdatePseudoGroupVisibility(in_data, params);
	}
		break;
	case PF_Cmd_RENDER:
		SyncLineColorParams(params);
		HideLineColorParams(in_data);
		err = Render(in_data, out_data, params, inOutput);
		break;
	}
	return err;
}
