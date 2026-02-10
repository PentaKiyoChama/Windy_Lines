# デバッグログトラブルシューティングメモ

## 🎯 目的
Mac環境でデバッグログが出力されない問題の調査と解決方法

---

## 📍 現在の状態

### 実装済みの内容
- SDK_ProcAmp.h にクロスプラットフォーム対応の WriteLog 関数を実装
- SDK_ProcAmp_CPU.cpp に色プリセット選択時のデバッグログを追加
- Windows/Mac 両環境で動作するはずのコード

### 問題
Mac環境でログファイルが作成されない

---

## 🔍 トラブルシューティング手順

### 1. `_DEBUG` マクロの確認

**問題**: DebugLog は `#ifdef _DEBUG` でラップされているため、リリースビルドでは何も出力されません。

**確認方法**:
```bash
# ビルド時に _DEBUG を明示的に定義
cd SDK_ProcAmp
make clean
make CFLAGS="-D_DEBUG" CXXFLAGS="-D_DEBUG"

# または Xcode の場合
xcodebuild -configuration Debug
```

**検証**:
```bash
# シンボルテーブルを確認
nm -g SDK_ProcAmp.plugin/Contents/MacOS/SDK_ProcAmp | grep -i debug
```

---

### 2. ファイル書き込み権限の確認

**テスト 1: `/tmp` への書き込み**:
```bash
touch /tmp/test.txt && echo "success" > /tmp/test.txt && cat /tmp/test.txt && rm /tmp/test.txt
```

**テスト 2: `~/Desktop` への書き込み**:
```bash
touch ~/Desktop/test.txt && echo "success" > ~/Desktop/test.txt && cat ~/Desktop/test.txt && rm ~/Desktop/test.txt
```

**テスト 3: After Effects のサンドボックス確認**:
```bash
# After Effects のプラグインプロセスの権限を確認
# システム環境設定 > セキュリティとプライバシー > プライバシー > フルディスクアクセス
# Adobe After Effects を追加
```

---

### 3. HOME 環境変数の確認

**確認方法**:
```bash
echo $HOME
# 期待される出力: /Users/username
```

**After Effects 内での確認**:
プラグインから環境変数を読み取れるか確認する必要があります。After Effects のサンドボックス内では環境変数が異なる可能性があります。

---

### 4. 簡易テストコードの追加

SDK_ProcAmp_CPU.cpp の `Render` 関数の最初に以下のテストコードを追加：

```cpp
// ===== デバッグログテスト開始 =====
// _DEBUG マクロ不要の常時ログ出力テスト
{
    const char* testPaths[] = {
        "/tmp/SDK_ProcAmp_Test.txt",
        "/Users/Shared/SDK_ProcAmp_Test.txt"  // より権限の緩い場所
    };
    
    for (int i = 0; i < 2; ++i) {
        FILE* fp = fopen(testPaths[i], "a");
        if (fp) {
            time_t now = time(NULL);
            fprintf(fp, "[%s] Render called - Test path %d worked!\n", 
                    ctime(&now), i);
            fclose(fp);
            break;  // 成功したら終了
        }
    }
}
// ===== デバッグログテスト終了 =====
```

**確認**:
```bash
# プラグインを実行後、ログファイルを確認
cat /tmp/SDK_ProcAmp_Test.txt
cat /Users/Shared/SDK_ProcAmp_Test.txt
```

---

### 5. WriteLog 関数の詳細確認

**現在の実装** (SDK_ProcAmp.h):
```cpp
static void WriteLog(const char* format, ...)
{
    std::lock_guard<std::mutex> lock(sLogMutex);
    
    #ifdef _WIN32
        const char* paths[] = {
            "C:\\Temp\\SDK_ProcAmp_Log.txt",
            "C:\\Users\\Owner\\Desktop\\SDK_ProcAmp_Log.txt"
        };
    #else
        // Mac/Unix paths
        const char* pathTemplates[] = {
            "/tmp/SDK_ProcAmp_Log.txt",
            "~/Desktop/SDK_ProcAmp_Log.txt"
        };
        char expandedPath[512] = "";
        const char* paths[2];
        paths[0] = pathTemplates[0];
        
        if (pathTemplates[1][0] == '~') {
            const char* home = getenv("HOME");
            if (home) {
                snprintf(expandedPath, sizeof(expandedPath), "%s%s", 
                        home, pathTemplates[1] + 1);
                paths[1] = expandedPath;
            } else {
                paths[1] = pathTemplates[1];
            }
        } else {
            paths[1] = pathTemplates[1];
        }
    #endif
    
    // ファイルオープンを試行
    FILE* fp = NULL;
    for (int i = 0; i < 2 && fp == NULL; ++i) {
        fp = fopen(paths[i], "a");
    }
    
    if (fp) {
        // タイムスタンプ付きでログ出力
        time_t now = time(NULL);
        struct tm* t = localtime(&now);
        fprintf(fp, "[%02d:%02d:%02d] ", t->tm_hour, t->tm_min, t->tm_sec);
        
        va_list args;
        va_start(args, format);
        vfprintf(fp, format, args);
        va_end(args);
        
        fprintf(fp, "\n");
        fclose(fp);
    }
}

#ifdef _DEBUG
    #define DebugLog WriteLog
#else
    #define DebugLog(...)  // No-op in release builds
#endif
```

**潜在的な問題**:
1. `_DEBUG` マクロが定義されていない
2. After Effects のサンドボックスによるファイルアクセス制限
3. `HOME` 環境変数が未設定または異なる値
4. パスの権限問題

---

### 6. 代替ログ出力場所

より権限の緩い場所を試す：

```cpp
#else
    // Mac/Unix paths - より多くの選択肢
    const char* pathTemplates[] = {
        "/tmp/SDK_ProcAmp_Log.txt",                    // 優先度1
        "/Users/Shared/SDK_ProcAmp_Log.txt",           // 優先度2（共有フォルダ）
        "~/Desktop/SDK_ProcAmp_Log.txt",               // 優先度3
        "/var/tmp/SDK_ProcAmp_Log.txt"                 // 優先度4
    };
    
    char expandedPath[512] = "";
    const char* paths[4];
    
    // パス展開ロジック
    for (int i = 0; i < 4; ++i) {
        if (pathTemplates[i][0] == '~') {
            const char* home = getenv("HOME");
            if (home) {
                snprintf(expandedPath, sizeof(expandedPath), "%s%s", 
                        home, pathTemplates[i] + 1);
                paths[i] = expandedPath;
            } else {
                paths[i] = pathTemplates[i];
            }
        } else {
            paths[i] = pathTemplates[i];
        }
    }
#endif
```

---

### 7. ログ出力の確認コマンド

```bash
# すべての可能性のある場所を確認
ls -la /tmp/SDK_ProcAmp_*.txt
ls -la ~/Desktop/SDK_ProcAmp_*.txt
ls -la /Users/Shared/SDK_ProcAmp_*.txt
ls -la /var/tmp/SDK_ProcAmp_*.txt

# リアルタイム監視
tail -f /tmp/SDK_ProcAmp_Log.txt

# 最近作成されたファイルを検索
find /tmp ~/Desktop /Users/Shared /var/tmp -name "*SDK_ProcAmp*" -type f -mmin -10 2>/dev/null
```

---

### 8. Console.app でのログ確認

Mac の Console.app を使用してシステムログを確認：

```bash
# ターミナルから Console.app を起動
open -a Console

# または、ログストリームを直接確認
log stream --predicate 'process == "After Effects"' --level debug
```

プラグインが出力するログやエラーメッセージが表示される可能性があります。

---

## 🧪 段階的テスト計画

### Phase 1: 基本的なファイル出力テスト
1. `_DEBUG` なしの常時ログ出力テストコードを追加
2. プラグインをビルド
3. After Effects で実行
4. ログファイルの存在を確認

### Phase 2: デバッグビルドの確認
1. `make clean && make CFLAGS="-D_DEBUG" CXXFLAGS="-D_DEBUG"` でビルド
2. シンボルテーブルで `_DEBUG` の存在を確認
3. After Effects で実行
4. DebugLog 出力を確認

### Phase 3: サンドボックス権限の確認
1. システム環境設定でフルディスクアクセス権限を確認
2. After Effects のセキュリティ設定を確認
3. より権限の緩いパス（`/Users/Shared`）を試す

### Phase 4: Console.app での確認
1. Console.app を起動
2. After Effects プロセスをフィルタ
3. プラグイン実行時のログを確認

---

## 📝 期待される出力

### 正常動作時のログ例

**test_3 (ID=36) を選択した場合**:
```
[12:34:56] [ColorPreset] Raw popup value: 36, Normalized index: 35, Mode: 1, Will call GetPresetPalette(36)
[12:34:56] [ColorPreset] Preset mode: Loading preset #36, First color: R=102 G=229 B=128
[12:34:56] [ColorPreset] Loaded 8 colors, Color[0]: R=0.40 G=0.90 B=0.50
```

**森 (ID=3) を選択した場合**:
```
[12:34:57] [ColorPreset] Raw popup value: 3, Normalized index: 2, Mode: 1, Will call GetPresetPalette(3)
[12:34:57] [ColorPreset] Preset mode: Loading preset #3, First color: R=102 G=229 B=128
[12:34:57] [ColorPreset] Loaded 8 colors, Color[0]: R=0.40 G=0.90 B=0.50
```

両方とも同じ RGB 値 (102, 229, 128) が表示されることで、test_3 と森が同じ色データを使用していることが確認できます。

---

## 🔧 推奨される次のステップ

1. **最も簡単な方法**: Phase 1 の常時ログ出力テストから開始
2. **デバッグビルドの確認**: `_DEBUG` マクロが正しく定義されているか確認
3. **代替パスの追加**: `/Users/Shared` など、より権限の緩い場所を追加
4. **Console.app の利用**: システムログから情報を収集
5. **必要に応じて**: After Effects のプラグイン開発ドキュメントでサンドボックス制限を確認

---

## 📚 参考情報

### 関連ファイル
- `SDK_ProcAmp.h` (WriteLog 関数の実装)
- `SDK_ProcAmp_CPU.cpp` (DebugLog の呼び出し箇所)

### 変更履歴
- コミット `40077d1`: デバッグログ追加（Windows専用）
- コミット `cda84b5`: Mac対応追加（パス展開バグあり）
- コミット `f862c69`: パス展開バグ修正

### 既知の問題
- Mac環境でログファイルが作成されない（原因調査中）
- 可能性: `_DEBUG` マクロ未定義、サンドボックス制限、権限問題

---

## ✅ チェックリスト

ローカル環境で以下を確認してください：

- [ ] `make CFLAGS="-D_DEBUG" CXXFLAGS="-D_DEBUG"` でビルド
- [ ] プラグインファイルのタイムスタンプが最新であることを確認
- [ ] After Effects のプラグインフォルダに正しくコピー
- [ ] After Effects を完全再起動
- [ ] `/tmp/SDK_ProcAmp_Log.txt` の存在確認
- [ ] `~/Desktop/SDK_ProcAmp_Log.txt` の存在確認
- [ ] テストコードで `/tmp/SDK_ProcAmp_Test.txt` の作成確認
- [ ] Console.app でログ確認
- [ ] システム環境設定 > セキュリティ > フルディスクアクセス確認

---

**作成日**: 2026-02-10  
**ステータス**: Mac環境でのデバッグログ出力問題の調査継続中  
**次のアクション**: ローカル環境で Phase 1 のテストから開始
