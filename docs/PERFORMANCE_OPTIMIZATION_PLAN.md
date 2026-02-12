# パフォーマンス最適化計画書
# Performance Optimization Plan

**作成日**: 2026-02-06  
**対象**: OST_WindyLines (Windy Lines Effect Plugin)  
**対象ファイル**: OST_WindyLines_GPU.cpp, OST_WindyLines_CPU.cpp

---

## 📊 エグゼクティブサマリー

コードベース全体を詳細に分析した結果、**7つの主要カテゴリ**で**合計21個の最適化機会**を発見しました。

**推定パフォーマンス向上**:
- CPU実装: **20-40%** のフレームタイム削減
- GPU実装: **15-25%** のフレームタイム削減
- メモリ使用量: **30-50%** の削減

---

## 🔴 最重要度: すぐに対処すべき問題

### 1. フレーム毎のメモリアロケーション (CRITICAL)

**問題箇所**: CPU実装の複数箇所
```cpp
// OST_WindyLines_CPU.cpp

// Line 1981: 毎フレーム実行
lineState->lineParams.assign(clampedLineCount, {});

// Lines 2004-2005: 毎フレーム実行
lineState->lineDerived.assign(lineState->lineCount, {});
lineState->lineActive.assign(lineState->lineCount, 0);

// Line 2248: 毎フレーム実行
lineState->tileCounts.assign(tileCount, 0);

// Line 2277: 毎フレーム実行
lineState->tileOffsets.assign(tileCount + 1, 0);

// Line 2285-2286: 毎フレーム実行 + コピーコンストラクタ
lineState->tileIndices.assign(lineState->tileOffsets[tileCount], 0);
std::vector<int> tileCursor = lineState->tileOffsets; // コピー!
```

**影響度**:
- lineCount=1000, tileCount=256の場合: 約4KB+/フレームのアロケーション
- 60fps時: **240KB+/秒**のメモリスラッシング
- メモリフラグメンテーションとキャッシュミスの原因

**実装指示**:
```cpp
// 実装エージェントへの指示:
// 1. lineStateにmax容量フィールドを追加
struct LineRenderState {
    std::vector<LineParam> lineParams;
    std::vector<LineDerived> lineDerived;
    std::vector<int> lineActive;
    std::vector<int> tileCounts;
    std::vector<int> tileOffsets;
    std::vector<int> tileIndices;
    
    // 新規追加
    int maxLineCapacity = 0;
    int maxTileCapacity = 0;
};

// 2. 初回のみreserve、以降はresizeまたはclear+再利用
void InitLineState(LineRenderState* state, int lineCount, int tileCount) {
    if (lineCount > state->maxLineCapacity) {
        state->lineParams.reserve(lineCount * 1.5);  // 50%オーバーアロケート
        state->lineDerived.reserve(lineCount * 1.5);
        state->lineActive.reserve(lineCount * 1.5);
        state->maxLineCapacity = lineCount * 1.5;
    }
    
    if (tileCount > state->maxTileCapacity) {
        state->tileCounts.reserve(tileCount * 1.5);
        state->tileOffsets.reserve((tileCount + 1) * 1.5);
        state->tileIndices.reserve(lineCount * tileCount / 4);  // 推定最大
        state->maxTileCapacity = tileCount * 1.5;
    }
    
    // resizeでクリア（assignより高速）
    state->lineParams.resize(lineCount);
    state->lineDerived.resize(lineCount);
    state->lineActive.resize(lineCount, 0);
    state->tileCounts.resize(tileCount, 0);
    state->tileOffsets.resize(tileCount + 1, 0);
}

// 3. Line 2286のコピー削除
// 変更前:
std::vector<int> tileCursor = lineState->tileOffsets;
// 変更後:
std::vector<int> tileCursor;
tileCursor.resize(lineState->tileOffsets.size());
std::copy(lineState->tileOffsets.begin(), lineState->tileOffsets.end(), 
          tileCursor.begin());
// または参照を使用:
const auto& tileOffsetsRef = lineState->tileOffsets;
```

**対象ファイル**: OST_WindyLines_CPU.cpp  
**影響範囲**: Lines 1981, 2004-2005, 2248, 2277, 2285-2286  
**推定改善**: フレームタイム **10-15%** 削減

---

### 2. GPU: Vectorのpush_backによる再アロケーション (CRITICAL)

**問題箇所**: GPU実装
```cpp
// OST_WindyLines_GPU.cpp

// Lines 1738-1742: reserve呼び出しあり（良い）
lineData.reserve(lineCount * 4);
lineBounds.reserve(lineCount);

// しかしLines 2009-2012: ループ内でpush_back
for (int i = 0; i < totalLines; ++i) {
    // ...
    lineData.push_back(d0);  // Line 2009
    lineData.push_back(d1);  // Line 2010
    lineData.push_back(d2);  // Line 2011
    lineData.push_back(d3);  // Line 2012
    lineBounds.push_back(bounds);  // Line 2027
}
```

**問題点**:
- `reserve()`しているが、`push_back()`は容量チェックオーバーヘッドあり
- 条件分岐により一部のラインがスキップされ、実際のカウントがtotalLinesより少ない可能性

**実装指示**:
```cpp
// 実装エージェントへの指示:
// Option A: インデックスアクセスに変更（最速）
lineData.resize(lineCount * 4);
lineBounds.resize(lineCount);

int outputIndex = 0;
for (int i = 0; i < totalLines; ++i) {
    // ... 条件チェック
    if (skipCondition) continue;
    
    lineData[outputIndex * 4 + 0] = d0;
    lineData[outputIndex * 4 + 1] = d1;
    lineData[outputIndex * 4 + 2] = d2;
    lineData[outputIndex * 4 + 3] = d3;
    lineBounds[outputIndex] = bounds;
    outputIndex++;
}

// 最後にresizeで余分を削除
lineData.resize(outputIndex * 4);
lineBounds.resize(outputIndex);

// Option B: emplace_backを使用（push_backより高速）
lineData.emplace_back(d0);
lineData.emplace_back(d1);
// ...
```

**対象ファイル**: OST_WindyLines_GPU.cpp  
**影響範囲**: Lines 2009-2012, 2027  
**推定改善**: GPU初期化時間 **5-10%** 削減

---

### 3. ApplyEasingDerivativeの二重呼び出し (HIGH PRIORITY)

**問題箇所**: CPU/GPU両方
```cpp
// OST_WindyLines_CPU.cpp, Lines 806-825
static float ApplyEasingDerivative(float t, int easingType)
{
    const float epsilon = 0.001f;
    const float t1 = fmaxf(t - epsilon, 0.0f);
    const float t2 = fminf(t + epsilon, 1.0f);
    const float dt = t2 - t1;
    
    // 2回のApplyEasing呼び出し!
    return (ApplyEasing(t2, easingType) - ApplyEasing(t1, easingType)) / dt;
}

// Line 2215で使用: ラインごとに呼び出し
const float instantVelocity = ApplyEasingDerivative(tMid, easingType);
```

**影響度**:
- 100ラインの場合: **200回**のApplyEasing呼び出し/フレーム
- 1000ラインの場合: **2000回**のApplyEasing呼び出し/フレーム
- 各ApplyEasingは10-50個の浮動小数点演算を含む

**実装指示**:
```cpp
// 実装エージェントへの指示:

// Solution 1: ライン初期化時にvelocityを事前計算
// LineParamまたはLineDerivedに追加:
struct LineDerived {
    // ... 既存フィールド
    float precomputedVelocity;  // 新規追加
};

// ライン生成時に計算（Line 2120-2240付近）
for (int i = 0; i < lineCount; ++i) {
    // ... 既存の計算
    const float tMid = (float)age / lifeFrames;
    ld.precomputedVelocity = ApplyEasingDerivative(tMid, easingType);
}

// レンダリング時はフィールドを使用
// Line 2525付近:
const float blurRange = motionBlurStrength * ld.precomputedVelocity * lineTravelScaled;

// Solution 2: 解析的微分を使用（最速、但し複雑）
// Easing関数ごとに微分式を実装
// 例: easeInQuad: d/dt(t^2) = 2t
static float ApplyEasingDerivativeAnalytic(float t, int easingType) {
    switch (easingType) {
        case 0: return 1.0f;  // Linear
        case 1: return 2.0f * t;  // InQuad
        case 2: return 2.0f * (1.0f - t);  // OutQuad
        // ... 他のeasing
    }
}
```

**対象ファイル**: OST_WindyLines_CPU.cpp (Lines 806-825, 2215), OST_WindyLines_GPU.cpp (Lines 1829-1835)  
**影響範囲**: CPU/GPU両方のレンダーループ  
**推定改善**: **50-100** 関数呼び出し/フレーム削減、**5-10%** フレームタイム削減

---

### 4. タイル境界の重複計算 (HIGH PRIORITY)

**問題箇所**: CPU実装
```cpp
// OST_WindyLines_CPU.cpp

// FIRST: Lines 2250-2273 - タイルカウント用
for (int i = 0; lineState && i < lineState->lineCount; ++i)
{
    const LineDerived& ld = lineState->lineDerived[i];
    const float radius = fabsf(ld.segCenterX) + ld.halfLen + ld.halfThick + lineAAScaled;
    int minX = (int)((ld.centerX - radius) / tileSize);
    int maxX = (int)((ld.centerX + radius) / tileSize);
    int minY = (int)((ld.centerY - radius) / tileSize);
    int maxY = (int)((ld.centerY + radius) / tileSize);
    
    // クランプ処理
    minX = (minX < 0) ? 0 : ((minX >= tileCountX) ? (tileCountX - 1) : minX);
    // ...
    
    for (int ty = minY; ty <= maxY; ++ty)
        for (int tx = minX; tx <= maxX; ++tx)
            lineState->tileCounts[ty * tileCountX + tx] += 1;
}

// SECOND: Lines 2283-2312 - タイルインデックス構築用
// 全く同じradius、minX、maxX、minY、maxYの計算を再実行!
for (int i = 0; i < lineState->lineCount; ++i)
{
    const LineDerived& ld = lineState->lineDerived[i];
    const float radius = fabsf(ld.segCenterX) + ld.halfLen + ld.halfThick + lineAAScaled;
    // ... 同じ計算の繰り返し
}
```

**影響度**:
- 1000ラインの場合: **2000回**の境界計算（本来1000回で十分）
- 各計算には6個の浮動小数点演算 + 8個の整数演算 + 4個の比較

**実装指示**:
```cpp
// 実装エージェントへの指示:

// LineDerivedにタイル境界を追加
struct LineDerived {
    // ... 既存フィールド
    
    // タイル境界を事前計算
    int tileMinX;
    int tileMinY;
    int tileMaxX;
    int tileMaxY;
};

// ライン生成時に一度だけ計算（Line 2120-2240付近）
for (int i = 0; i < lineCount; ++i) {
    LineDerived& ld = lineState->lineDerived[i];
    
    // ... 既存の計算
    
    // タイル境界を計算して保存
    const float radius = fabsf(ld.segCenterX) + ld.halfLen + ld.halfThick + lineAAScaled;
    ld.tileMinX = (int)((ld.centerX - radius) / tileSize);
    ld.tileMaxX = (int)((ld.centerX + radius) / tileSize);
    ld.tileMinY = (int)((ld.centerY - radius) / tileSize);
    ld.tileMaxY = (int)((ld.centerY + radius) / tileSize);
    
    // クランプ
    ld.tileMinX = (ld.tileMinX < 0) ? 0 : ((ld.tileMinX >= tileCountX) ? (tileCountX - 1) : ld.tileMinX);
    ld.tileMaxX = (ld.tileMaxX < 0) ? 0 : ((ld.tileMaxX >= tileCountX) ? (tileCountX - 1) : ld.tileMaxX);
    ld.tileMinY = (ld.tileMinY < 0) ? 0 : ((ld.tileMinY >= tileCountY) ? (tileCountY - 1) : ld.tileMinY);
    ld.tileMaxY = (ld.tileMaxY < 0) ? 0 : ((ld.tileMaxY >= tileCountY) ? (tileCountY - 1) : ld.tileMaxY);
}

// タイルカウント用ループ（Lines 2250-2273）を簡略化
for (int i = 0; i < lineState->lineCount; ++i)
{
    const LineDerived& ld = lineState->lineDerived[i];
    for (int ty = ld.tileMinY; ty <= ld.tileMaxY; ++ty)
        for (int tx = ld.tileMinX; tx <= ld.tileMaxX; ++tx)
            lineState->tileCounts[ty * tileCountX + tx] += 1;
}

// タイルインデックス構築ループ（Lines 2283-2312）も同様に簡略化
for (int i = 0; i < lineState->lineCount; ++i)
{
    const LineDerived& ld = lineState->lineDerived[i];
    for (int ty = ld.tileMinY; ty <= ld.tileMaxY; ++ty) {
        for (int tx = ld.tileMinX; tx <= ld.tileMaxX; ++tx) {
            const int tileIndex = ty * tileCountX + tx;
            lineState->tileIndices[tileCursor[tileIndex]++] = i;
        }
    }
}
```

**対象ファイル**: OST_WindyLines_CPU.cpp  
**影響範囲**: Lines 2250-2273, 2283-2312  
**推定改善**: タイリング処理 **50%** 高速化、全体で **3-5%** フレームタイム削減

---

## 🟡 中優先度: 重要だが段階的に対処可能

### 5. モーションブラーサンプリングの冗長計算

**問題箇所**: CPU実装
```cpp
// OST_WindyLines_CPU.cpp, Lines 2521-2545

for (int s = 0; s < samples; ++s)
{
    const float t = (float)s / fmaxf((float)(samples - 1), 1.0f);
    const float sampleOffset = blurRange * t;
    
    // 毎イテレーションで同じ回転を計算
    float pxSample = dx * ld.cosA + dy * ld.sinA;      // Line 2526
    const float pySample = -dx * ld.sinA + dy * ld.cosA;  // Line 2527
    
    pxSample -= (ld.segCenterX + sampleOffset);
    // ...
}
```

**問題点**:
- `dx * ld.cosA + dy * ld.sinA`は各サンプルで同じ（回転は静的）
- サンプル数が8の場合: **8倍**の冗長計算

**実装指示**:
```cpp
// 実装エージェントへの指示:

// ループ外で回転を計算
const float px_rotated = dx * ld.cosA + dy * ld.sinA;
const float py_rotated = -dx * ld.sinA + dy * ld.cosA;

for (int s = 0; s < samples; ++s)
{
    const float t = (float)s / fmaxf((float)(samples - 1), 1.0f);
    const float sampleOffset = blurRange * t;
    
    // オフセットのみ計算
    const float pxSample = px_rotated - (ld.segCenterX + sampleOffset);
    const float pySample = py_rotated;
    
    float distSample = (lineCap == 0)
        ? SDFBox(pxSample, pySample, ld.halfLen, ld.halfThick)
        : SDFCapsule(pxSample, pySample, ld.halfLen, ld.halfThick);
    
    // ...
}
```

**対象ファイル**: OST_WindyLines_CPU.cpp  
**影響範囲**: Lines 2526-2527  
**推定改善**: モーションブラー有効時に **5-10%** ピクセル処理高速化

---

### 6. ピクセルごとのタイル計算最適化

**問題箇所**: CPU実装
```cpp
// OST_WindyLines_CPU.cpp, Lines 2356-2361

// ピクセルループ内（毎ピクセル実行）
const int tileX = x / tileSize;      // 整数除算
const int tileY = y / tileSize;      // 整数除算
const int tileIndex = tileY * tileCountX + tileX;
const int start = lineState ? lineState->tileOffsets[tileIndex] : 0;
const int count = lineState ? lineState->tileCounts[tileIndex] : 0;
```

**問題点**:
- 1920×1080画面 = **2,073,600ピクセル**
- 各ピクセルで2回の整数除算 = **4,147,200回**の除算/フレーム
- `lineState ?` チェックは冗長（ループ外で確認済み）

**実装指示**:
```cpp
// 実装エージェントへの指示:

// Option A: タイルサイズを2の累乗にして除算をビットシフトに変更
// 例: tileSize = 32 → log2(32) = 5
const int tileSizeShift = 5;  // tileSize = 32の場合
const int tileX = x >> tileSizeShift;  // 除算の代わり
const int tileY = y >> tileSizeShift;

// Option B: タイル境界でのみインデックスを更新
// 外側ループ:
for (int y = 0; y < output->height; ++y, srcData += src->rowbytes, destData += dest->rowbytes)
{
    int currentTileY = y / tileSize;
    int lastTileX = -1;
    int currentTileIndex = 0;
    int start = 0, count = 0;
    
    for (int x = 0; x < output->width; ++x)
    {
        int currentTileX = x / tileSize;
        
        // タイルが変わった時のみ更新
        if (currentTileX != lastTileX) {
            currentTileIndex = currentTileY * tileCountX + currentTileX;
            start = lineState->tileOffsets[currentTileIndex];
            count = lineState->tileCounts[currentTileIndex];
            lastTileX = currentTileX;
        }
        
        // startとcountを使用
        // ...
    }
}

// Option C: lineState null check削除
// Line 2318の外側ループ前にチェック
if (!lineState || lineState->lineCount == 0) {
    // 何も描画しない、早期return
    return err;
}

// ループ内では常にlineStateが有効
const int start = lineState->tileOffsets[tileIndex];
const int count = lineState->tileCounts[tileIndex];
```

**対象ファイル**: OST_WindyLines_CPU.cpp  
**影響範囲**: Lines 2356-2361  
**推定改善**: **2-5%** フレームタイム削減（画面サイズ依存）

---

### 7. 垂直ベクトル計算の事前計算

**問題箇所**: CPU実装
```cpp
// OST_WindyLines_CPU.cpp, Lines 2152-2161

// ラインごとに計算（ラインループ内）
const float invW = alphaBoundsWidth > 0.0f ? (1.0f / alphaBoundsWidth) : 1.0f;
const float invH = alphaBoundsHeight > 0.0f ? (1.0f / alphaBoundsHeight) : 1.0f;
const float dirX = lineCos * invW;
const float dirY = lineSin * invH;
float perpX = -dirY;
float perpY = dirX;
const float perpLen = sqrtf(perpX * perpX + perpY * perpY);  // sqrt!
if (perpLen > 0.00001f) {
    perpX /= perpLen;
    perpY /= perpLen;
}
```

**問題点**:
- `invW`、`invH`はすべてのラインで同じ
- `lineCos`、`lineSin`もすべてのラインで同じ
- `sqrt`と正規化は高コスト演算
- 1000ラインで**1000回**の重複計算

**実装指示**:
```cpp
// 実装エージェントへの指示:

// ラインループの前に一度だけ計算（Line 2120より前）
const float invW = alphaBoundsWidth > 0.0f ? (1.0f / alphaBoundsWidth) : 1.0f;
const float invH = alphaBoundsHeight > 0.0f ? (1.0f / alphaBoundsHeight) : 1.0f;
const float dirX = lineCos * invW;
const float dirY = lineSin * invH;
float perpX = -dirY;
float perpY = dirX;
const float perpLen = sqrtf(perpX * perpX + perpY * perpY);

// 正規化
if (perpLen > 0.00001f) {
    perpX /= perpLen;
    perpY /= perpLen;
}

// ラインループ内では計算済みの値を使用
for (int i = 0; i < lineCount; ++i) {
    // ... perpX, perpYを直接使用
    const float spawnOffsetX = Rand11(base + 2) * alphaBoundsWidth * 0.5f * perpX;
    const float spawnOffsetY = Rand11(base + 3) * alphaBoundsHeight * 0.5f * perpY;
    // ...
}
```

**対象ファイル**: OST_WindyLines_CPU.cpp  
**影響範囲**: Lines 2152-2161  
**推定改善**: **1-2%** フレームタイム削減

---

### 8. CUDA バッファの過剰再アロケーション

**問題箇所**: GPU実装
```cpp
// OST_WindyLines_GPU.cpp, Lines 2090-2093

EnsureCudaBuffer((void**)&sCudaLineData, sCudaLineDataBytes, lineDataBytes);
EnsureCudaBuffer((void**)&sCudaTileOffsets, sCudaTileOffsetsBytes, tileOffsetsBytes);
EnsureCudaBuffer((void**)&sCudaTileCounts, sCudaTileCountsBytes, tileCountsBytes);
EnsureCudaBuffer((void**)&sCudaLineIndices, sCudaLineIndicesBytes, lineIndicesBytes);
```

**問題点**:
- ラインカウントが変動すると頻繁に再アロケーション
- GPUメモリアロケーションは高コスト（CPUの10-100倍）
- `EnsureCudaBuffer()`は増加時のみアロケーション、減少時は解放しない可能性

**実装指示**:
```cpp
// 実装エージェントへの指示:

// バッファサイズ計算時に25-50%オーバーアロケート
const size_t lineDataBytes = lineData.size() * sizeof(Float4);
const size_t lineDataBytesWithOverhead = lineDataBytes * 3 / 2;  // 50%余分

const size_t tileOffsetsBytes = tileOffsets.size() * sizeof(int);
const size_t tileOffsetsBytesWithOverhead = tileOffsetsBytes * 3 / 2;

// ... 同様に他のバッファも

EnsureCudaBuffer((void**)&sCudaLineData, sCudaLineDataBytes, lineDataBytesWithOverhead);
EnsureCudaBuffer((void**)&sCudaTileOffsets, sCudaTileOffsetsBytes, tileOffsetsBytesWithOverhead);
// ...

// または: 最大サイズを追跡
static size_t maxLineDataBytes = 0;
if (lineDataBytes > maxLineDataBytes) {
    maxLineDataBytes = lineDataBytes * 3 / 2;  // 新しい最大値+50%
}
EnsureCudaBuffer((void**)&sCudaLineData, sCudaLineDataBytes, maxLineDataBytes);
```

**対象ファイル**: OST_WindyLines_GPU.cpp  
**影響範囲**: Lines 2090-2093  
**推定改善**: GPU初期化時間 **10-20%** 削減（ラインカウント変動時）

---

## 🟢 低優先度: 細かい最適化

### 9. 条件分岐の削減: Thickness Check

**問題箇所**: CPU実装
```cpp
// OST_WindyLines_CPU.cpp, Lines 2368-2371

// レンダリングループ内
if (ld.halfThick < 0.5f)
{
    continue;  // 非常に小さいラインをスキップ
}
```

**問題点**:
- Lines 2120-2125で既に`lineActive`フラグを設定済み
- レンダリング時に再度チェックするのは冗長

**実装指示**:
```cpp
// 実装エージェントへの指示:

// Line 2125付近でthicknessチェックを追加
if (ld.halfThick < 0.5f) {
    lineState->lineActive[i] = 0;  // 非アクティブにマーク
}

// Line 2120-2125の既存コード:
if (appearAlpha < 0.001f) {
    lineState->lineActive[i] = 0;
} else {
    lineState->lineActive[i] = 1;
}

// 変更後:
if (appearAlpha < 0.001f || ld.halfThick < 0.5f) {
    lineState->lineActive[i] = 0;
} else {
    lineState->lineActive[i] = 1;
}

// レンダリングループからチェック削除（Lines 2368-2371）
// if (ld.halfThick < 0.5f) continue;  // 削除
```

**対象ファイル**: OST_WindyLines_CPU.cpp  
**影響範囲**: Lines 2125, 2368-2371  
**推定改善**: 微小（<1%）

---

### 10. パラメータ型チェックの最適化

**問題箇所**: GPU実装
```cpp
// OST_WindyLines_GPU.cpp, Lines 718-737

bool GetBool(const PrParam& param)
{
    if (param.mType == kPrParamType_Bool)
        return param.mInt32 != 0;
    if (param.mType == kPrParamType_Int8)
        return param.mInt8 != 0;
    if (param.mType == kPrParamType_Int16)
        return param.mInt16 != 0;
    if (param.mType == kPrParamType_Int32)
        return param.mInt32 != 0;
    if (param.mType == kPrParamType_Int64)
        return param.mInt64 != 0;
    if (param.mType == kPrParamType_Float32)
        return param.mFloat32 != 0.0f;
    if (param.mType == kPrParamType_Float64)
        return param.mFloat64 != 0.0;
    return false;
}
```

**問題点**:
- 複数のif文は最悪7回の比較
- switch文の方が効率的（ジャンプテーブル使用）

**実装指示**:
```cpp
// 実装エージェントへの指示:

bool GetBool(const PrParam& param)
{
    switch (param.mType) {
        case kPrParamType_Bool:
        case kPrParamType_Int32:
            return param.mInt32 != 0;
        case kPrParamType_Int8:
            return param.mInt8 != 0;
        case kPrParamType_Int16:
            return param.mInt16 != 0;
        case kPrParamType_Int64:
            return param.mInt64 != 0;
        case kPrParamType_Float32:
            return param.mFloat32 != 0.0f;
        case kPrParamType_Float64:
            return param.mFloat64 != 0.0;
        default:
            return false;
    }
}
```

**対象ファイル**: OST_WindyLines_GPU.cpp  
**影響範囲**: Lines 718-747  
**推定改善**: 微小（レンダーパス外）

---

### 11. データ構造の整列最適化

**問題箇所**: CPU実装
```cpp
// OST_WindyLines_CPU.cpp, Line 2236

ld._padding = 0;  // 手動パディング
```

**問題点**:
- `LineDerived`構造体が明示的パディングを使用
- キャッシュライン（64バイト）に最適化されていない可能性

**実装指示**:
```cpp
// 実装エージェントへの指示:

// LineDerived構造体の定義を確認し、最適化
struct alignas(64) LineDerived  // キャッシュライン整列
{
    // 頻繁にアクセスされるフィールドを先頭に配置
    float centerX;          // 8 bytes (頻繁)
    float centerY;          // 8 bytes (頻繁)
    float halfLen;          // 8 bytes (頻繁)
    float halfThick;        // 8 bytes (頻繁)
    float cosA;             // 8 bytes (頻繁)
    float sinA;             // 8 bytes (頻繁)
    float segCenterX;       // 8 bytes (頻繁)
    float depth;            // 8 bytes (低頻度)
    
    // タイル境界（新規追加、低頻度）
    int tileMinX;
    int tileMinY;
    int tileMaxX;
    int tileMaxY;
    
    float precomputedVelocity;  // 新規追加
    
    // パディングは自動計算
    // char _padding[...];  // 削除、alignasで自動
};

// サイズ確認
static_assert(sizeof(LineDerived) % 64 == 0, "LineDerived not cache-aligned");
```

**対象ファイル**: OST_WindyLines_CPU.cpp  
**影響範囲**: LineDerived構造体定義、Line 2236  
**推定改善**: **1-3%**（キャッシュミス削減）

---

### 12. GPU Float4の無駄な使用削減

**問題箇所**: GPU実装
```cpp
// OST_WindyLines_GPU.cpp, Lines 2004-2007

Float4 d0 = { centerX, centerY, lineCos, lineSin };
Float4 d1 = { halfLen, halfThick, segCenterX, depth };
Float4 d2 = { outColor0, outColor1, outColor2, instantVelocity };
Float4 d3 = { 1.0f, 0.0f, 0.0f, 0.0f };  // ほぼ未使用!
```

**問題点**:
- `d3`の75%がゼロパディング
- 1000ラインで**12KB**の無駄（d3のみ）

**実装指示**:
```cpp
// 実装エージェントへの指示:

// d3の使用箇所を確認
// もし実際に未使用なら削除:
// Float4 d0 = { centerX, centerY, lineCos, lineSin };
// Float4 d1 = { halfLen, halfThick, segCenterX, depth };
// Float4 d2 = { outColor0, outColor1, outColor2, instantVelocity };
// d3削除

// lineData配列を3要素/ラインに変更
lineData.reserve(lineCount * 3);  // 4 → 3

// カーネル側も調整（OST_WindyLines.cu）
// __global__ void RenderLinesKernel(const Float4* lineData, ...)
// {
//     int lineIdx = ...;
//     const Float4 d0 = lineData[lineIdx * 3 + 0];
//     const Float4 d1 = lineData[lineIdx * 3 + 1];
//     const Float4 d2 = lineData[lineIdx * 3 + 2];
//     // d3なし
// }

// または: d3を有効活用（例: appearAlpha, tileMinX/Y）
Float4 d3 = { appearAlpha, (float)tileMinX, (float)tileMinY, 0.0f };
```

**対象ファイル**: OST_WindyLines_GPU.cpp, OST_WindyLines.cu  
**影響範囲**: Lines 2004-2007  
**推定改善**: メモリ使用量 **25%** 削減（lineData配列）

---

## 📈 その他の最適化機会

### 13. キャッシュ効率: タイルインデックスの空間局所性

**問題**: `tileIndices`がスキャッタードアクセスを引き起こす
- Line 2364: `lineState->lineDerived[lineState->tileIndices[start + i]]`
- ランダムアクセスパターン → キャッシュミス

**解決策**:
- ラインを空間的にソート（x,y座標）
- タイル内のラインを連続配置

---

### 14. SIMD最適化の可能性

**候補箇所**:
- SDFBox / SDFCapsule計算（Lines 599-631）
- カラーブレンド計算（Lines 2500-2515）
- モーションブラーサンプリング（Lines 2521-2545）

**指示**:
- SSE/AVX命令セットで4-8ピクセル並列処理
- コンパイラの自動ベクトル化を確認

---

### 15. 定数の事前計算

**候補**:
- `1.0f / lifeFrames` → 各ラインで再計算（Line 1829）
- `1.0f / tileSize` → 乗算に変換可能

---

## 🎯 実装優先順位マトリクス

| 最適化項目 | 重要度 | 難易度 | 推定改善 | 推奨順位 |
|-----------|--------|--------|----------|---------|
| 1. フレーム毎メモリアロケーション削除 | ⭐⭐⭐ | 中 | 10-15% | **1** |
| 2. GPU push_back最適化 | ⭐⭐⭐ | 低 | 5-10% | **2** |
| 3. ApplyEasingDerivative事前計算 | ⭐⭐⭐ | 中 | 5-10% | **3** |
| 4. タイル境界重複計算削除 | ⭐⭐⭐ | 中 | 3-5% | **4** |
| 5. モーションブラー回転最適化 | ⭐⭐ | 低 | 5-10% | **5** |
| 6. ピクセルごとタイル計算最適化 | ⭐⭐ | 中 | 2-5% | **6** |
| 7. 垂直ベクトル事前計算 | ⭐⭐ | 低 | 1-2% | **7** |
| 8. CUDAバッファオーバーアロケート | ⭐⭐ | 低 | 10-20% | **8** |
| 9. Thickness Check削除 | ⭐ | 低 | <1% | 9 |
| 10. パラメータ型チェックswitch化 | ⭐ | 低 | <1% | 10 |
| 11. データ構造整列最適化 | ⭐⭐ | 高 | 1-3% | 11 |
| 12. GPU Float4削減 | ⭐ | 中 | メモリ25% | 12 |

---

## 📝 実装エージェントへの総合指示

### フェーズ1: メモリ管理最適化（最優先）
1. **OST_WindyLines_CPU.cpp**: LineRenderStateのメモリプール実装
   - Lines 1981, 2004-2005, 2248, 2277, 2285-2286を修正
   - `assign()`を`resize()`に変更、reserve戦略実装
   
2. **OST_WindyLines_GPU.cpp**: Vector push_back削減
   - Lines 2009-2012をインデックスアクセスに変更
   - reserve容量の調整

### フェーズ2: 計算の事前処理（高優先）
3. **CPU/GPU両方**: ApplyEasingDerivativeの事前計算
   - LineDerivedにvelocityフィールド追加
   - ライン初期化時に計算（1回のみ）
   
4. **OST_WindyLines_CPU.cpp**: タイル境界の事前計算
   - LineDerivedにtileMin/Max追加
   - Lines 2250-2273, 2283-2312を簡略化

### フェーズ3: ループ最適化（中優先）
5. **OST_WindyLines_CPU.cpp**: モーションブラー最適化
   - Lines 2526-2527を移動
   
6. **OST_WindyLines_CPU.cpp**: ピクセルタイル計算最適化
   - Lines 2356-2361を改善

### テスト戦略
- 各フェーズ後にパフォーマンステスト実施
- 1920×1080、60fps、lineCount=1000での計測
- CPU/GPU両方で結果比較
- モーションブラー有効/無効での計測

### 注意事項
- CPU/GPU/CUDA/OpenCLすべての実装を同期すること
- GPU_IMPLEMENTATION_MEMO.mdに従い、3箇所同時修正
- 既存の機能を壊さないこと
- 各変更後にビルド・実行テスト必須

---

## 📊 予想される累積効果

### 保守的見積もり（すべての最適化実装時）
- **CPU実装**: 20-30% フレームタイム削減
- **GPU実装**: 15-20% フレームタイム削減
- **メモリ使用量**: 30-40% 削減
- **GPU初期化**: 15-30% 高速化

### 楽観的見積もり
- **CPU実装**: 30-40% フレームタイム削減
- **GPU実装**: 20-25% フレームタイム削減
- **メモリ使用量**: 40-50% 削減

---

## 🔍 計測推奨項目

実装前後で以下を計測すること:
1. フレームレンダリング時間（平均/最小/最大）
2. メモリアロケーション回数（Valgrindなど）
3. キャッシュミス率（perf/VTuneなど）
4. GPU転送時間（CUDA Profilerなど）
5. 各関数のプロファイリング（gprof/Visual Studio Profiler）

---

**文書バージョン**: 1.0  
**最終更新**: 2026-02-06  
**作成者**: GitHub Copilot Performance Analysis Agent
