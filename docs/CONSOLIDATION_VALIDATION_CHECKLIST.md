# Consolidation Validation Checklist

対象: 共通化後の `OST_WindyLines_Common.h` 導入変更（Phase 1〜4）

関連仕様書: `docs/CODE_CONSOLIDATION_SPEC.md`

## 0. 事前条件

- 最新コードでビルドできること
- 検証用シーケンスに以下を用意
  - アルファ付き素材（要素範囲テスト用）
  - 不透明素材（通常描画確認用）
  - WindyLines を適用した比較クリップ（同一設定で CPU/GPU 切替）

---

## 1. ビルド確認

- [ ] Mac: Xcode でクリーンビルド成功
- [ ] Win: Visual Studio でクリーンビルド成功
- [ ] 起動時クラッシュなし（Premiere Pro でプラグイン読み込み成功）

期待結果:
- コンパイル/リンクエラーなし
- エフェクト追加時にクラッシュしない

---

## 2. CPU/GPU 基本一致

同一クリップ・同一パラメータで比較:

- [ ] GPU ON（通常）で描画確認
- [ ] GPU OFF（Software Only）で描画確認
- [ ] 見た目（線の位置/太さ/長さ/色）が一致

期待結果:
- フレーム間で大きな乖離がない

---

## 3. Easing 回帰（ApplyEasing/Derivative）

- [ ] Easing を代表値で確認（Linear / InOutSine / OutBack / InOutElastic / InOutBounce）
- [ ] 太さ変化（Derivative依存箇所）が不自然に跳ねない
- [ ] CPU/GPU切替時の挙動が一致

期待結果:
- 動きと速度変化が旧挙動と同等

---

## 4. Linkage 回帰（ApplyLinkageValue）

各項目（Length/Thickness/Travel）で確認:

- [ ] Linkage = OFF
- [ ] Linkage = WIDTH
- [ ] Linkage = HEIGHT
- [ ] Thickness 極小時でも最小値ガード（1.0）が効く

期待結果:
- OFF 時は手入力値基準、WIDTH/HEIGHT 時は要素範囲連動
- CPU/GPU で同じ結果

---

## 5. Color Preset 回帰（BuildPresetPalette）

- [ ] Single モード確認
- [ ] Custom モード確認
- [ ] Preset モードで複数プリセット（例: Rainbow/Pastel/Forest）切替
- [ ] CPU/GPU で同じ色配列になることを目視確認

期待結果:
- Preset 切替で色が正しく変わる
- Single/Custom の既存挙動に副作用なし

---

## 6. 追加の安全確認

- [ ] Line Cap（Flat/Round）で破綻なし
- [ ] Blend Mode（Back/Front/Back&Front/Alpha）で破綻なし
- [ ] Spawn Source（Full Frame / Element Bounds）で破綻なし

---

## 7. 受け入れ判定

以下を満たせばマージ可:

- [ ] ビルド成功（Mac/Win）
- [ ] CPU/GPU の見た目一致
- [ ] Easing / Linkage / Preset で回帰なし
- [ ] 主要モード切替で破綻なし

---

## 検証ログ（記録用）

- 実施日:
- 実施者:
- Premiere バージョン:
- OS:
- GPU:
- 結果サマリ:
- 問題点:
- スクリーンショット保存先:
