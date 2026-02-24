/*******************************************************************/
/*                                                                 */
/*  __TPL_MATCH_NAME__ - CPU Renderer (Fallback)                   */
/*  GPU が使えない環境用のCPUレンダリングパス                       */
/*                                                                 */
/*  Copyright (c) __TPL_YEAR__ __TPL_AUTHOR__. All rights reserved.*/
/*                                                                 */
/*******************************************************************/

#include "__TPL_MATCH_NAME__.h"
#include "__TPL_MATCH_NAME___Version.h"
#include "__TPL_MATCH_NAME___ParamNames.h"
#include "__TPL_MATCH_NAME___Common.h"
#include "__TPL_MATCH_NAME___WatermarkMask.h"

#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "Param_Utils.h"
#include "PrSDKAESupport.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <math.h>
#include <mutex>
#include <string>
#include <string.h>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#else
#include <pwd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#endif


/*******************************************************************/
/*  静的データの実体化                                              */
/*******************************************************************/
#ifdef __cplusplus
std::unordered_map<csSDK_int64, csSDK_int64> SharedClipData::clipStartMap;
std::mutex SharedClipData::mapMutex;
#endif


// ================================================================
// ================ LICENSE VERIFICATION SYSTEM ===================
// ================================================================
// Shared across all OshareTelop plugins via common cache file.
// One activation from any plugin unlocks all plugins.
// Cache: Mac ~/Library/Application Support/OshareTelop/license_cache_v1.txt
//        Win %APPDATA%\OshareTelop\license_cache_v1.txt
// ================================================================

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
        ++begin;
    while (end > begin && (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\r' || value[end - 1] == '\n'))
        --end;
    return value.substr(begin, end - begin);
}

static bool ParseBoolLike(const std::string& value, bool* outValue)
{
    if (!outValue) return false;
    if (value == "1" || value == "true" || value == "TRUE" || value == "True")
    { *outValue = true; return true; }
    if (value == "0" || value == "false" || value == "FALSE" || value == "False")
    { *outValue = false; return true; }
    return false;
}

// --- Cache file paths (shared across all OshareTelop plugins) ---
static std::vector<std::string> GetLicenseCachePaths()
{
#ifdef _WIN32
    std::vector<std::string> paths;
    char appData[MAX_PATH] = { 0 };
    DWORD appDataLen = GetEnvironmentVariableA("APPDATA", appData, MAX_PATH);
    if (appDataLen > 0 && appDataLen < MAX_PATH)
        paths.push_back(std::string(appData) + "\\OshareTelop\\license_cache_v1.txt");
    const char* userProfile = std::getenv("USERPROFILE");
    if (userProfile && *userProfile)
        paths.push_back(std::string(userProfile) + "\\AppData\\Roaming\\OshareTelop\\license_cache_v1.txt");
    paths.push_back(std::string("C:\\Temp\\ost_license_cache_v1.txt"));
    return paths;
#else
    std::vector<std::string> paths;
    const char* home = std::getenv("HOME");
    if (home && *home)
        paths.push_back(std::string(home) + "/Library/Application Support/OshareTelop/license_cache_v1.txt");
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_dir && *pw->pw_dir)
    {
        const std::string pwHomePath = std::string(pw->pw_dir) + "/Library/Application Support/OshareTelop/license_cache_v1.txt";
        bool exists = false;
        for (const auto& path : paths) { if (path == pwHomePath) { exists = true; break; } }
        if (!exists) paths.push_back(pwHomePath);
    }
    paths.push_back(std::string("/tmp/ost_license_cache_v1.txt"));
    return paths;
#endif
}

// --- DJB2 hash ---
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

// --- Machine ID: hostname + platform + pointer size (shared across all plugins) ---
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

// --- Cache signature salt (XOR-obfuscated, shared with Windy_Lines) ---
// Salt: OST_WL_2026_SALT_K9x3
// Regenerate: python3 -c "salt='OST_WL_2026_SALT_K9x3'; key=0xA7; print(','.join(f'0x{c^key:02X}' for c in salt.encode()))"
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

// --- Cache signature: DJB2-based HMAC to detect tampering ---
static std::string ComputeCacheSignature(const std::string& authorizedStr,
                                          const std::string& validatedUnixStr,
                                          const std::string& machineIdHash,
                                          const std::string& expireUnixStr)
{
    // payload = "authorized|validated_unix|machine_id|expire_unix|salt"
    std::string payload = authorizedStr + "|" + validatedUnixStr + "|" + machineIdHash + "|" + expireUnixStr + "|" + GetCacheSignatureSalt();
    uint32_t h1 = SimpleHash32(payload.c_str());
    std::string pass2 = payload + "|" + std::to_string(h1);
    uint32_t h2 = SimpleHash32(pass2.c_str());
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%08x%08x", h1, h2);
    return std::string(buf);
}

// Offline grace: 1 hour after last successful verification
static const int kOfflineGracePeriodSec = 3600;

static bool LoadLicenseAuthenticatedFromCache(bool* outAuthenticated, bool* outExpired = nullptr)
{
    if (!outAuthenticated) return false;
    if (outExpired) *outExpired = false;

    const std::vector<std::string> cachePaths = GetLicenseCachePaths();
    std::string cachePath;
    FILE* file = nullptr;
    for (const auto& candidate : cachePaths)
    {
        file = std::fopen(candidate.c_str(), "rb");
        if (file) { cachePath = candidate; break; }
    }
    if (!file)
    {
        DebugLog("[License] cache file not found");
        return false;
    }

    bool hasAuthorized = false, authorized = false;
    bool hasExpire = false;
    long long expireUnix = 0;
    std::string cachedMachineIdHash, authorizedRaw, validatedUnixRaw, expireUnixRaw, cachedSignature;

    char line[512];
    while (std::fgets(line, static_cast<int>(sizeof(line)), file) != nullptr)
    {
        std::string rawLine(line);
        const size_t sep = rawLine.find('=');
        if (sep == std::string::npos) continue;
        const std::string key = TrimAscii(rawLine.substr(0, sep));
        const std::string value = TrimAscii(rawLine.substr(sep + 1));

        if (key == "authorized")
        {
            authorizedRaw = value;
            bool parsed = false;
            if (ParseBoolLike(value, &parsed))
            { authorized = parsed; hasAuthorized = true; }
        }
        else if (key == "cache_expire_unix")
        {
            expireUnixRaw = value;
            char* endPtr = nullptr;
            const long long parsed = std::strtoll(value.c_str(), &endPtr, 10);
            if (endPtr != value.c_str()) { expireUnix = parsed; hasExpire = true; }
        }
        else if (key == "validated_unix") { validatedUnixRaw = value; }
        else if (key == "machine_id_hash") { cachedMachineIdHash = value; }
        else if (key == "cache_signature") { cachedSignature = value; }
    }
    std::fclose(file);

    if (!hasAuthorized || !hasExpire)
    {
        DebugLog("[License] cache invalid: missing fields");
        return false;
    }

    // Machine ID verification (anti-copy)
    if (!cachedMachineIdHash.empty())
    {
        const std::string localMid = GetMachineIdHash();
        if (cachedMachineIdHash != localMid)
        {
            DebugLog("[License] machine_id mismatch");
            return false;
        }
    }

    // Signature verification (anti-tampering)
    if (cachedSignature.empty())
    {
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

    // TTL check with offline grace
    if (expireUnix <= nowUnix)
    {
        long long validatedUnix = 0;
        if (!validatedUnixRaw.empty())
        {
            char* endPtr = nullptr;
            validatedUnix = std::strtoll(validatedUnixRaw.c_str(), &endPtr, 10);
        }
        const long long graceCutoff = validatedUnix + kOfflineGracePeriodSec;
        if (authorized && graceCutoff > nowUnix)
        {
            DebugLog("[License] cache expired but within offline grace");
            *outAuthenticated = true;
            if (outExpired) *outExpired = true;
            return true;
        }
        DebugLog("[License] cache expired beyond grace");
        return false;
    }

    *outAuthenticated = authorized;
    DebugLog("[License] cache loaded: authenticated=%s path=%s",
        authorized ? "true" : "false", cachePath.c_str());
    return true;
}

static std::atomic<bool> sLicenseAuthenticated{ false };

// --- Activation token ---
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
    if (lastSep != std::string::npos) dir = dir.substr(0, lastSep);
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
        else { std::fclose(f); }
    }
    std::string token = GenerateActivationToken();
#ifdef _WIN32
    const size_t lastSep = path.find_last_of('\\');
    if (lastSep != std::string::npos)
        CreateDirectoryA(path.substr(0, lastSep).c_str(), nullptr);
#else
    const size_t lastSep = path.find_last_of('/');
    if (lastSep != std::string::npos)
    {
        std::string mkdirCmd = "/bin/mkdir -p '" + path.substr(0, lastSep) + "'";
        system(mkdirCmd.c_str());
    }
#endif
    FILE* fw = std::fopen(path.c_str(), "wb");
    if (fw) { std::fputs(token.c_str(), fw); std::fputs("\n", fw); std::fclose(fw); }
    return token;
}

// --- Bubble endpoints ---
#if defined(__TPL_UPPER_PREFIX___FORCE_TEST_ENDPOINT)
static const char* kBubbleBaseUrl = "https://penta.bubbleapps.io/version-test";
static const wchar_t* kBubbleApiHostW = L"penta.bubbleapps.io";
static const wchar_t* kBubbleApiPathW = L"/version-test/api/1.1/wf/ppplugin_test";
#elif defined(NDEBUG)
static const char* kBubbleBaseUrl = "https://penta.bubbleapps.io";
static const wchar_t* kBubbleApiHostW = L"penta.bubbleapps.io";
static const wchar_t* kBubbleApiPathW = L"/api/1.1/wf/ppplugin_test";
#else
static const char* kBubbleBaseUrl = "https://penta.bubbleapps.io/version-test";
static const wchar_t* kBubbleApiHostW = L"penta.bubbleapps.io";
static const wchar_t* kBubbleApiPathW = L"/version-test/api/1.1/wf/ppplugin_test";
#endif

static const std::string kActivatePageUrl = std::string(kBubbleBaseUrl) + "/activate";

static void OpenActivationPage()
{
    const std::string mid = GetMachineIdHash();
    const std::string token = LoadOrCreateActivationToken();
    const char* platform =
#ifdef _WIN32
        "win";
#else
        "mac";
#endif
    const std::string url = kActivatePageUrl
        + "?token=" + token
        + "&mid=" + mid
        + "&platform=" + std::string(platform)
        + "&product=__TPL_MATCH_NAME__"
        + "&ver=" __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_FULL;
    DebugLog("[License] Opening activation page: token=%s mid=%s", token.c_str(), mid.c_str());
#ifdef _WIN32
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    std::string cmd = "open '" + url + "' &";
    system(cmd.c_str());
#endif
}

// --- Background auto-refresh ---
static const std::string kLicenseApiEndpoint = std::string(kBubbleBaseUrl) + "/api/1.1/wf/ppplugin_test";
static const int kAuthorizedCacheTtlSec = 600;
static const int kDeniedCacheTtlSec = 600;
static std::atomic<bool> sAutoRefreshInProgress{false};
static std::atomic<uint32_t> sLastAutoRefreshAttemptMs{0};
static const uint32_t kMinAutoRefreshIntervalMs = 60000;
static std::atomic<uint32_t> sActivationTriggeredAtMs{0};
static const uint32_t kPostActivationRapidWindowMs = 120000;
static const uint32_t kPostActivationCheckIntervalMs = 5000;

static void TriggerBackgroundCacheRefresh()
{
    bool expected = false;
    if (!sAutoRefreshInProgress.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return;

    const uint32_t nowMs = GetCurrentTimeMs();
    const uint32_t lastMs = sLastAutoRefreshAttemptMs.load(std::memory_order_relaxed);
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
    if (lastSlash != std::string::npos) cacheDir = cacheDir.substr(0, lastSlash);

#ifndef _WIN32
    // Mac: curl-based background refresh
    const std::string endpoint(kLicenseApiEndpoint);
    const int ttlSecOk = kAuthorizedCacheTtlSec;
    const int ttlSecDenied = kDeniedCacheTtlSec;
    const std::string mid = GetMachineIdHash();

    std::thread([endpoint, ttlSecOk, ttlSecDenied, mid, cachePath, cacheDir]() {
        DebugLog("[License] background cache refresh started (Mac/popen)");
        std::string mkdirCmd = "/bin/mkdir -p '" + cacheDir + "'";
        system(mkdirCmd.c_str());
        std::string curlCmd =
            "/usr/bin/curl -s -m 10 -X POST "
            "-H 'Content-Type: application/json' "
            "-d '{\"action\":\"verify\",\"product\":\"__TPL_MATCH_NAME__\","
            "\"plugin_version\":\"" __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_FULL "\","
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
        while (std::fgets(buf, sizeof(buf), pipe) != nullptr) responseBody += buf;
        pclose(pipe);
        DebugLog("[License] API response: %s", responseBody.c_str());

        bool authorized = false;
        std::string reason = "unknown";
        if (responseBody.find("\"authorized\"") != std::string::npos)
        {
            if (responseBody.find("\"authorized\":true") != std::string::npos ||
                responseBody.find("\"authorized\": true") != std::string::npos ||
                responseBody.find("\"authorized\" : true") != std::string::npos)
            { authorized = true; reason = "ok"; }
            else { authorized = false; reason = "denied"; }
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
            "machine_id_hash=%s\ncache_signature=%s\n",
            authStr.c_str(), reason.c_str(), nowUnix, expireUnix,
            mid.c_str(), sig.c_str());

        char tmpBuf[64];
        std::snprintf(tmpBuf, sizeof(tmpBuf), "/tmp/ost_cache_%08x", static_cast<unsigned>(nowUnix));
        std::string tmpFile(tmpBuf);
        FILE* fp = std::fopen(tmpFile.c_str(), "wb");
        if (fp)
        {
            std::fputs(content, fp);
            std::fclose(fp);
            if (std::rename(tmpFile.c_str(), cachePath.c_str()) != 0)
            {
                fp = std::fopen(cachePath.c_str(), "wb");
                if (fp) { std::fputs(content, fp); std::fclose(fp); }
                std::remove(tmpFile.c_str());
            }
            DebugLog("[License] cache updated: authorized=%s", authStr.c_str());
        }
        sLicenseAuthenticated.store(authorized, std::memory_order_relaxed);
        if (authorized) sActivationTriggeredAtMs.store(0, std::memory_order_relaxed);
        sAutoRefreshInProgress.store(false, std::memory_order_release);
    }).detach();

    DebugLog("[License] background cache refresh triggered");
#else
    // Windows: WinHTTP-based background API call
    const int ttlSecOk = kAuthorizedCacheTtlSec;
    const int ttlSecDenied = kDeniedCacheTtlSec;
    const std::string mid = GetMachineIdHash();
    std::thread([cachePath, cacheDir, ttlSecOk, ttlSecDenied, mid]() {
        DebugLog("[License] background cache refresh started (WinHTTP)");
        CreateDirectoryA(cacheDir.c_str(), nullptr);
        std::string body = "{\"action\":\"verify\",\"product\":\"__TPL_MATCH_NAME__\","
            "\"plugin_version\":\"" __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_FULL "\",\"platform\":\"win\","
            "\"machine_id\":\"" + mid + "\"}";

        HINTERNET hSession = WinHttpOpen(
            L"__TPL_MATCH_NAME__/1.0",
            WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
            WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (!hSession) { sAutoRefreshInProgress.store(false, std::memory_order_release); return; }
        HINTERNET hConnect = WinHttpConnect(hSession, kBubbleApiHostW, INTERNET_DEFAULT_HTTPS_PORT, 0);
        if (!hConnect) { WinHttpCloseHandle(hSession); sAutoRefreshInProgress.store(false, std::memory_order_release); return; }
        HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", kBubbleApiPathW,
            NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
        if (!hRequest) { WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); sAutoRefreshInProgress.store(false, std::memory_order_release); return; }
        DWORD timeout = 10000;
        WinHttpSetOption(hRequest, WINHTTP_OPTION_CONNECT_TIMEOUT, &timeout, sizeof(timeout));
        WinHttpSetOption(hRequest, WINHTTP_OPTION_SEND_TIMEOUT, &timeout, sizeof(timeout));
        WinHttpSetOption(hRequest, WINHTTP_OPTION_RECEIVE_TIMEOUT, &timeout, sizeof(timeout));
        BOOL bResult = WinHttpSendRequest(hRequest, L"Content-Type: application/json", -1,
            (LPVOID)body.c_str(), (DWORD)body.size(), (DWORD)body.size(), 0);
        if (!bResult) { WinHttpCloseHandle(hRequest); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); sAutoRefreshInProgress.store(false, std::memory_order_release); return; }
        bResult = WinHttpReceiveResponse(hRequest, NULL);
        if (!bResult) { WinHttpCloseHandle(hRequest); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); sAutoRefreshInProgress.store(false, std::memory_order_release); return; }
        std::string responseBody;
        DWORD bytesAvailable = 0;
        while (WinHttpQueryDataAvailable(hRequest, &bytesAvailable) && bytesAvailable > 0)
        {
            std::vector<char> buf(bytesAvailable);
            DWORD bytesRead = 0;
            if (WinHttpReadData(hRequest, buf.data(), bytesAvailable, &bytesRead))
                responseBody.append(buf.data(), bytesRead);
            else break;
        }
        WinHttpCloseHandle(hRequest); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession);
        DebugLog("[License] API response: %s", responseBody.c_str());

        bool authorized = false;
        std::string reason = "unknown";
        if (responseBody.find("\"authorized\"") != std::string::npos)
        {
            if (responseBody.find("\"authorized\":true") != std::string::npos ||
                responseBody.find("\"authorized\": true") != std::string::npos ||
                responseBody.find("\"authorized\" : true") != std::string::npos)
            { authorized = true; reason = "ok"; }
            else { authorized = false; reason = "denied"; }
        }
        else { sAutoRefreshInProgress.store(false, std::memory_order_release); return; }

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
            "machine_id_hash=%s\ncache_signature=%s\n",
            authStr.c_str(), reason.c_str(), nowUnix, expireUnix, mid.c_str(), sig.c_str());
        FILE* fp = std::fopen(cachePath.c_str(), "wb");
        if (fp) { std::fputs(content, fp); std::fclose(fp); }
        sLicenseAuthenticated.store(authorized, std::memory_order_relaxed);
        if (authorized) sActivationTriggeredAtMs.store(0, std::memory_order_relaxed);
        sAutoRefreshInProgress.store(false, std::memory_order_release);
    }).detach();
#endif
}

// --- Public API (declared in _License.h) ---
void RefreshLicenseAuthenticatedState(bool force)
{
    const uint32_t nowMs = GetCurrentTimeMs();
    if (!force)
    {
        const uint32_t lastMs = sLastLicenseRefreshTimeMs.load(std::memory_order_relaxed);
        if (nowMs - lastMs < kLicenseRefreshIntervalMs) return;
    }
    sLastLicenseRefreshTimeMs.store(nowMs, std::memory_order_relaxed);

    bool cachedAuthenticated = false;
    bool cacheExpired = false;
    if (LoadLicenseAuthenticatedFromCache(&cachedAuthenticated, &cacheExpired))
    {
        sLicenseAuthenticated.store(cachedAuthenticated, std::memory_order_relaxed);
        if (cachedAuthenticated)
        {
            sActivationTriggeredAtMs.store(0, std::memory_order_relaxed);
            if (cacheExpired) TriggerBackgroundCacheRefresh();
        }
        else
        {
            const uint32_t activatedAt = sActivationTriggeredAtMs.load(std::memory_order_relaxed);
            if (activatedAt != 0 && (nowMs - activatedAt < kPostActivationRapidWindowMs))
                TriggerBackgroundCacheRefresh();
        }
    }
    else
    {
        sLicenseAuthenticated.store(false, std::memory_order_relaxed);
        TriggerBackgroundCacheRefresh();
    }
}

bool IsLicenseAuthenticated()
{
    return sLicenseAuthenticated.load(std::memory_order_relaxed);
}

// ================================================================
// ================ END LICENSE VERIFICATION ======================
// ================================================================


/*******************************************************************/
/*  エントリーポイント                                              */
/*******************************************************************/
static PF_Err About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_SPRINTF(out_data->return_msg,
        "%s %s\n%s",
        "__TPL_EFFECT_NAME_JP__",
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___VERSION_FULL,
        "(c) __TPL_YEAR__ __TPL_AUTHOR__");
    return PF_Err_NONE;
}


static PF_Err GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION,
                                       STAGE_VERSION, BUILD_VERSION);

    out_data->out_flags =
        PF_OutFlag_PIX_INDEPENDENT |
        PF_OutFlag_SEND_UPDATE_PARAMS_UI;

    out_data->out_flags2 =
        PF_OutFlag2_SUPPORTS_SMART_RENDER |
        PF_OutFlag2_FLOAT_COLOR_AWARE |
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

#if ENABLE_GPU_RENDERING
    out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
#endif

    if (in_data->appl_id == 'PrMr')
    {
        AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite(
            in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);
        (*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
        (*pixelFormatSuite->AddSupportedPixelFormat)(
            in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
    }

    return err;
}


static PF_Err ParamsSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    // ---- パラメータ1: エフェクト量（スライダー） ----
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDER(
        P_AMOUNT,                      // パラメータ名
        PARAM_AMOUNT_MIN_VALUE,        // 有効最小値
        PARAM_AMOUNT_MAX_VALUE,        // 有効最大値
        PARAM_AMOUNT_MIN_SLIDER,       // スライダー最小値
        PARAM_AMOUNT_MAX_SLIDER,       // スライダー最大値
        0,                             // カーブ許容値
        PARAM_AMOUNT_DFLT,             // デフォルト値
        1,                             // 精度
        0,                             // 表示精度
        0,                             // フラグ
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_AMOUNT);

    // ---- パラメータ2: カラー ----
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(
        P_COLOR,
        255, 255, 255,                 // デフォルト RGB
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_COLOR);

    // ---- パラメータ3: ポップアップ ----
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        P_MODE,
        3,                             // 選択肢の数
        PARAM_POPUP_DFLT,              // デフォルト値
        PM_MODE,                       // メニュー文字列（"|"区切り）
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_POPUP);

    out_data->num_params = __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___NUM_PARAMS;

    return err;
}


/*******************************************************************/
/*  CPUレンダリング本体                                             */
/*  NOTE: GPUが使えない場合のフォールバック。                        */
/*  SmartRender 経由で呼ばれる。                                    */
/*******************************************************************/
static PF_Err SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    PF_RenderRequest req = extra->input->output_request;
    PF_CheckoutResult result;

    ERR(extra->cb->checkout_layer(
        in_data->effect_ref,
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___INPUT,
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &result));

    UnionLRect(&result.result_rect, &extra->output->result_rect);
    UnionLRect(&result.max_result_rect, &extra->output->max_result_rect);

    return err;
}


static PF_Err SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    // 入力レイヤーチェックアウト
    PF_EffectWorld* input_worldP = nullptr;
    ERR(extra->cb->checkout_layer_pixels(
        in_data->effect_ref,
        __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___INPUT,
        &input_worldP));

    PF_EffectWorld* output_worldP = nullptr;
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

    if (!err && input_worldP && output_worldP)
    {
        // パラメータ取得
        PF_ParamDef paramAmount, paramColor, paramPopup;
        AEFX_CLR_STRUCT(paramAmount);
        AEFX_CLR_STRUCT(paramColor);
        AEFX_CLR_STRUCT(paramPopup);

        ERR(PF_CHECKOUT_PARAM(in_data,
            __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_AMOUNT,
            in_data->current_time, in_data->time_step, in_data->time_scale,
            &paramAmount));
        ERR(PF_CHECKOUT_PARAM(in_data,
            __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_COLOR,
            in_data->current_time, in_data->time_step, in_data->time_scale,
            &paramColor));
        ERR(PF_CHECKOUT_PARAM(in_data,
            __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___PARAM_POPUP,
            in_data->current_time, in_data->time_step, in_data->time_scale,
            &paramPopup));

        float amount = (float)paramAmount.u.fs_d.value;
        int mode = paramPopup.u.pd.value;

        // License check (done once per frame, debounced)
        RefreshLicenseAuthenticatedState(false);
        const bool isLicensedThisFrame = IsLicenseAuthenticated();

        // ========================================
        // TODO: ここにCPUレンダリングロジックを実装
        // input_worldP → output_worldP にピクセルを書き込む
        // ========================================

        // サンプル: 入力をそのまま出力にコピー
        if (!err)
        {
            PF_Pixel8* inRow;
            PF_Pixel8* outRow;
            int width = output_worldP->width;
            int height = output_worldP->height;

            for (int y = 0; y < height; y++)
            {
                inRow  = (PF_Pixel8*)((char*)input_worldP->data  + y * input_worldP->rowbytes);
                outRow = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                memcpy(outRow, inRow, width * sizeof(PF_Pixel8));
            }

            // --- Watermark overlay (free/unlicensed mode) ---
            if (!isLicensedThisFrame)
            {
                const int wmTextWidthPx = FreeModeWatermark::TextWidthPx();
                const int wmTextHeightPx = FreeModeWatermark::TextHeightPx();
                const int wmMarginX = FreeModeWatermark::kMarginX;
                const int wmMarginY = FreeModeWatermark::kMarginY;
                const float wmScale = (width > 1920) ? 2.0f : 1.0f;
                const int wmScaledWidthPx = static_cast<int>(wmTextWidthPx * wmScale);
                const int wmScaledHeightPx = static_cast<int>(wmTextHeightPx * wmScale);
                const int wmScaledMarginY = static_cast<int>(wmMarginY * wmScale);
                const int wmStartX = width - wmScaledWidthPx - static_cast<int>(wmMarginX * wmScale);

                for (int wy = wmScaledMarginY; wy < wmScaledMarginY + wmScaledHeightPx && wy < height; wy++)
                {
                    outRow = (PF_Pixel8*)((char*)output_worldP->data + wy * output_worldP->rowbytes);
                    for (int wx = (wmStartX > 0 ? wmStartX : 0); wx < wmStartX + wmScaledWidthPx && wx < width; wx++)
                    {
                        const int localX = wx - wmStartX;
                        const int localY = wy - wmScaledMarginY;
                        const int sampleX = wmMarginX + static_cast<int>(static_cast<float>(localX) / wmScale);
                        const int sampleY = wmMarginY + static_cast<int>(static_cast<float>(localY) / wmScale);
                        const float fillAlpha = static_cast<float>(FreeModeWatermark::FillAlphaAt(sampleX, sampleY)) / 255.0f;
                        const float outlineAlpha = static_cast<float>(FreeModeWatermark::OutlineAlphaAt(sampleX, sampleY)) / 255.0f;
                        const bool fill = fillAlpha > 0.0f;
                        const bool outline = (!fill) && (outlineAlpha > 0.0f);
                        if (fill || outline)
                        {
                            const float baseAlpha = fill ? FreeModeWatermark::kFillOpacity : FreeModeWatermark::kOutlineOpacity;
                            const float overlayAlpha = baseAlpha * (fill ? fillAlpha : outlineAlpha);
                            // Simple alpha blend on 8-bit ARGB
                            float r = outRow[wx].red   / 255.0f;
                            float g = outRow[wx].green / 255.0f;
                            float b = outRow[wx].blue  / 255.0f;
                            float targetVal = fill ? 1.0f : 0.0f;
                            r = r + (targetVal - r) * overlayAlpha;
                            g = g + (targetVal - g) * overlayAlpha;
                            b = b + (targetVal - b) * overlayAlpha;
                            outRow[wx].red   = static_cast<A_u_char>(r * 255.0f);
                            outRow[wx].green = static_cast<A_u_char>(g * 255.0f);
                            outRow[wx].blue  = static_cast<A_u_char>(b * 255.0f);
                            if (outRow[wx].alpha < static_cast<A_u_char>(overlayAlpha * 255.0f))
                                outRow[wx].alpha = static_cast<A_u_char>(overlayAlpha * 255.0f);
                        }
                    }
                }
            }
        }

        // パラメータチェックイン
        ERR(PF_CHECKIN_PARAM(in_data, &paramAmount));
        ERR(PF_CHECKIN_PARAM(in_data, &paramColor));
        ERR(PF_CHECKIN_PARAM(in_data, &paramPopup));
    }

    return err;
}


/*******************************************************************/
/*  メインディスパッチャー                                          */
/*******************************************************************/
extern "C" DllExport
PF_Err EffectMain(
    PF_Cmd cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err err = PF_Err_NONE;

    try
    {
        switch (cmd)
        {
            case PF_Cmd_ABOUT:
                err = About(in_data, out_data, params, output);
                break;

            case PF_Cmd_GLOBAL_SETUP:
                err = GlobalSetup(in_data, out_data, params, output);
                break;

            case PF_Cmd_PARAMS_SETUP:
                err = ParamsSetup(in_data, out_data, params, output);
                break;

            case PF_Cmd_SMART_PRE_RENDER:
                err = SmartPreRender(in_data, out_data,
                    reinterpret_cast<PF_PreRenderExtra*>(extra));
                break;

            case PF_Cmd_SMART_RENDER:
                err = SmartRender(in_data, out_data,
                    reinterpret_cast<PF_SmartRenderExtra*>(extra));
                break;

            default:
                break;
        }
    }
    catch (PF_Err& thrown_err)
    {
        err = thrown_err;
    }

    return err;
}
