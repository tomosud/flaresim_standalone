# Unreal Engine 対応の調査

FlareSim を Unreal Engine（UE5）で使う方法の調査メモです。実装計画ではなく、選択肢・技術的な論点・概算の規模をまとめています。UE の API 名はエンジンのバージョンで変わるため、着手時に対象バージョンのソースで確認が必要です。

---

## 1. 結論

- **CUDA は障害になりません。** 原典（Nuke / AE 版）は CUDA ですが、今回の Web 版で計算カーネルを **WGSL（WebGPU のシェーダ言語）** に書き直しており、UE のシェーダ（HLSL / `.usf`）へはほぼ機械的に書き換えられます。CUDA 固有の機能（float の atomicAdd）は、Web 版ですでに **u32 の atomic による 64bit 固定小数点** に置き換えてあり、これは HLSL の `InterlockedAdd` でそのまま実現できます。NVIDIA 以外の GPU や Mac でも動く形になります。
- **用途で難しさが大きく変わります。**
  1. **オフライン（シネマティクス・映像）**：追加開発なしで今すぐ可能。Movie Render Queue でシーンリニアの EXR を出し、Web 版か Nuke 版でフレアを足す
  2. **エディタ / シーケンサでのプレビュー**：UE プラグイン（Scene View Extension + RDG のコンピュートシェーダ）で、光源数とペア数を絞れば実時間に近い速度で動く見込み
  3. **ゲームでの実時間描画**：今のアルゴリズム（点のスプラット）のままではフレーム予算に収まらない。研究論文の高速化手法（ゴーストを三角形メッシュとして描く等）が必要
- 推奨は **1 → 2 の順**。2 の小さな試作（PoC）で、実時間に足りるかを計測してから 3 を判断する。

## 2. 前提となる数字（Web 版の実測）

NVIDIA Turing 世代の GPU、1920×1080、253 ペアのレンズ、129 光源：

| 設定 | トレース数 | 時間 |
| --- | --- | --- |
| 瞳 64×64、3 波長 | 3.2 億 | 0.59 秒 |
| 瞳 16×16、3 波長 | 2,000 万 | 0.22 秒 |

計算量は `ペア数 × 光源数 × 瞳サンプル × 波長` に比例します。ゲームの 1 フレーム（16 ms）で使えるのはせいぜい 1〜2 ms なので、実時間には 2〜3 桁の削減が必要です。例：

| 削減 | 規模 |
| --- | --- |
| 光源を画像検出ではなく**ライトアクタ数個**にする（AE 版の手動光源と同じ考え方） | 129 → 4 |
| 推定強度の**上位ペアだけ**使う（Web 版の「Top 20」） | 253 → 20 |
| 瞳 16×16 ＋ 3 波長 | 約 600 本 × 3 |

→ 4 × 20 × 200 × 3 ≈ **5 万トレース**。上の実測から見て、スプラット方式のままでも 1 ms 前後に収まる見込みがあります（要計測）。ただし低いサンプル数ではボケたゴーストが点々に見えるので、ブラーか時間方向の平均（フレームごとに瞳サンプルをずらして TSR / TAA で均す）が必要です。

## 3. 選択肢

### A. オフライン合成（開発なし）

- Movie Render Queue で **トーンマップ前のシーンリニア EXR** を出力する（Linear / ACES の EXR 出力設定。太陽などの光源が 1.0 を大きく超える値で残っていることが条件）
- Web 版（このリポジトリ）または Nuke 版に読み込み、`flare.*` レイヤー付き EXR で書き出して合成
- 連番の一括処理は Web 版には未実装。必要なら Node.js の CLI 版（WebGPU は Dawn / `node-webgpu` など）か、Nuke 版を使う

### B. UE プラグイン（プレビュー〜実時間）

構成の候補：

| 要素 | 実装 |
| --- | --- |
| 描画への差し込み | `FSceneViewExtensionBase` を継承し、`PrePostProcessPass_RenderThread`、または `SubscribeToPostProcessingPass` でトーンマップ前のパスにコールバックを登録。HDR のシーンカラーに**加算**する（トーンマップ後に足すと HDR の明るさ関係が壊れる） |
| GPU 計算 | `FGlobalShader` のコンピュートシェーダを RDG（Render Dependency Graph）でディスパッチ。Web 版の 5 つの WGSL カーネル（ゴースト追跡・解決・ブラー・ヘイズ・スターバースト）を HLSL に移植 |
| 積算 | `RWStructuredBuffer<uint>` と `InterlockedAdd` で 64bit 固定小数点（Web 版と同じ）。SM6.6 の 64bit atomic が使える環境ではさらに単純にできる |
| 光源 | (1) ライトアクタ（Directional / Point / Spot）の画面位置と強度を CPU 側で計算して渡す、(2) SceneColor を縮小してしきい値で抽出し、Append バッファで光源リストを作る。最初は (1) が安定かつ安価 |
| レンズデータ | 1,370 本の JSON を `UDataAsset` か DataTable にインポート。ペアの事前選別（CPU、数 ms）はレンズ / 画角を変えたときだけ |
| ヘイズ・光芒 | UE 既存のブルーム / Convolution Bloom で代用可能。スターバーストの PSF（絞りの FFT）を Convolution Bloom のカーネル画像として使う手もある |
| 設定 UI | Post Process Volume の拡張は難しいので、Actor Component か `UDeveloperSettings` に `UPROPERTY` で公開 |

WGSL → HLSL の対応はほぼ一対一です：

| WGSL（Web 版） | HLSL（UE） |
| --- | --- |
| `@compute @workgroup_size(64)` | `[numthreads(64,1,1)]` |
| `var<storage, read> a: array<T>` | `StructuredBuffer<T>`（RDG の SRV） |
| `var<storage, read_write> acc: array<atomic<u32>>` | `RWStructuredBuffer<uint>` + `InterlockedAdd`（RDG の UAV） |
| `var<uniform> u: U` | シェーダパラメータ構造体（`BEGIN_SHADER_PARAMETER_STRUCT`） |
| `select(f, t, c)` | `c ? t : f` |
| バッチ分割＋`onSubmittedWorkDone` | 1 フレームで終わる規模に制限、またはフレームをまたいで分割・非同期コンピュート |

### C. 実時間向けのアルゴリズム変更（研究寄り）

- **Hullin ほか 2011「Physically-Based Real-Time Lens Flare Rendering」（SIGGRAPH 2011）**：瞳のサンプル格子を像面でのメッシュとして扱い、三角形をラスタライズする。点のスプラットと違い、少ない光線（数十×数十）でも滑らかなゴーストになり、面積変化から明るさも正しく出せる
- **Lee & Eisemann 2013「Practical Real-Time Lens-Flare Rendering」（EGSR 2013）**：光線追跡を多項式（行列）近似に置き換えてさらに高速化
- どちらも今回の光学モデル（レンズ形式・分散・コーティング・ペア）をそのまま入力にできる。UE 標準のレンズフレア（ブルームから作る画像ベースのゴースト）とは別物

## 4. 論点・リスク

- **フレーム予算**：光源数とペア数で計算量が掛け算で増える。上限を設けて、超えたら暗い光源から落とす（Web 版の Max Sources と同じ）
- **時間方向のちらつき**：低サンプル数では点々のノイズが毎フレーム変わる。瞳サンプルを固定する（安定だが模様が出る）か、フレームごとにずらして TSR に均させるか
- **GPU タイムアウト**：エディタで高品質設定にすると 1 回のディスパッチが長くなる。Web 版と同じく分割が必要
- **光源の画面外**：画面外の太陽でもフレアは出るべきなので、ライトアクタ方式が有利（画像検出では拾えない）
- **エンジンのバージョン差**：Scene View Extension の一部オーバーロードは非推奨・未呼び出しになっている。最終パスに差し込む場合は override output への書き込みが必須など、制約がある
- **ライセンス**：原典・本移植とも MIT で、UE プラグイン化に問題はない（著作権表示を残す）

## 5. 規模の目安（参考）

| 段階 | 内容 | 目安 |
| --- | --- | --- |
| A | MRQ の EXR → Web 版で合成する手順書 | 数日 |
| B-PoC | ライトアクタ光源、上位ペアのみ、ゴーストだけを加算するプラグイン。計測で実時間に足りるか判断 | 1〜2 週間 |
| B | 設定 UI、レンズアセット、画像ベース光源、ヘイズ / 光芒、MRQ 対応 | +3〜4 週間 |
| C | メッシュラスタライズ方式の研究実装 | 1〜2 か月 |

（1 人の、UE のレンダリングに慣れたエンジニアを想定した粗い見積もり）

## 6. 次に確かめること

1. 対象の UE バージョンと、使いたい場面（シネマティクスか、ゲーム内か）
2. B-PoC として、`SubscribeToPostProcessingPass` → RDG でゴースト追跡カーネル 1 本を動かし、光源 4・ペア 20・瞳 16² での GPU 時間を `stat gpu` / Unreal Insights で計測
3. 計測結果が 1 ms を大きく超えるなら C（メッシュ方式）を検討

## 参考

- [ISceneViewExtension::SubscribeToPostProcessingPass（Epic 公式 API）](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Engine/ISceneViewExtension/SubscribeToPostProcessingPass)
- [ISceneViewExtension（Epic 公式 API）](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Engine/ISceneViewExtension)
- [UE5 Noise Generation — Scene View Extension＋コンピュートシェーダの実例](https://github.com/lumixem/UE5-Noise-Generation)
- [Custom Vignette Pass & Editor Plugin（シェーダディレクトリの登録と FSceneViewExtensionBase の実装例）](https://velog.io/@imeamangryang/Unreal-Engine-Custom-Vignette-Pass-Editor-Plugin-제작)
- Hullin, Eisemann, Seidel, Lee. *Physically-Based Real-Time Lens Flare Rendering.* ACM TOG 30(4), SIGGRAPH 2011
- Lee, Eisemann. *Practical Real-Time Lens-Flare Rendering.* Computer Graphics Forum 32(4), EGSR 2013
