# FlareSim Web

実在レンズの設計データ（1,370本）に光線を通して、物理ベースのレンズフレア（ゴースト）をブラウザ上で計算するツールです。オープンソースの [flaresim_nuke](https://github.com/LocalStarlight/flaresim_nuke)（Nuke / CUDA 版）と [flaresim_ae](https://github.com/ken-eizo/flaresim_ae)（After Effects 版）を **WebGPU に移植** し、インストール不要のスタンドアローン Web アプリにしました。

- **HDR 表示**：HDR ディスプレイ＋Chrome / Edge 129 以降なら、1.0 を超える輝度をそのまま表示します（WebGPU `toneMapping: extended`）
- **OpenEXR 書き出し**：half / float、ZIP / 無圧縮。Nuke 版と同じく `flare.* / haze.* / starburst.*` を別レイヤーとして同じファイルに書けます
- **入力**：EXR（three.js の EXRLoader）、Radiance HDR、PNG / JPG / WebP / AVIF（白飛び付近を HDR に持ち上げるオプション付き）
- **計算は原典と同じ**：レンズ面の幾何、Cauchy 分散、Fresnel 反射、MgF₂ コーティング、3 フェーズのゴースト追跡、ペアの事前フィルタ、面積補正、FFT スターバースト、ヘイズ。C++ 原典との数値一致をテストで確認しています（[技術解説](docs/TECHNICAL.md#検証)）

## 使い方

### 光点モード（既定）

起動すると、黒い背景に光点が 1 つあるだけの **光点モード** で始まります。

- ビュー上を左ドラッグすると光点が動き、フレアがついてきます（右 / 中ボタンのドラッグで画面の移動）
- 光点は**表示だけのレイヤー**です。EXR / PNG に書き出されるのはフレア（ゴースト・ヘイズ・光芒）だけで、光点は入りません。ビュー上部の **光点** チェック（`L` キー）で、光点を消した書き出しと同じ状態を確認できます
- **Sources → Light point** で明るさを決めます。輝度（cd/m²）、シーンの白に対する EV、視直径（度）を表示し、輝度を直接入力することもできます。「Scene white」は、シーンリニア値 1.0 を何 cd/m² とみなすか（カメラ露出の基準）です
- 明るさのプリセット：太陽、車のヘッドライト、ハロゲン電球、LED 街灯、白熱電球、ろうそく、満月（値は目安）。太陽と月は視直径も画角から合わせます
- 光点の大きさ（Diameter）の範囲から光線を散らすので（Area sampling）、大きい光源ほどゴーストが滑らかにボケます

### 画像モード

1. ヘッダーの **画像** に切り替えるとテストシーン（夕景＋太陽＋街灯）、**画像を開く** やドロップで自分の EXR / HDR / 写真を読み込みます
2. **Sources → スポイトで Threshold を選ぶ** を押してビュー上の明るい部分をクリックすると、その場所の輝度（検出に使う Downsample ブロックの平均）が Threshold に入ります。そこから値を上げると、暗い光源から外れていきます（上限 100,000）
3. 右パネルの **Lens** でレンズを検索して選びます。断面図にゴーストの光路が出ます（**Ghost pairs** の行をクリックするとその反射ペアをハイライト）
4. **Sources → Threshold** で光源にしたい明るさを決めます（`Sources` 表示モードで検出位置を確認できます）
5. **Flare Gain / Haze Gain / Starburst Gain** はキャッシュ済みの結果を拡大縮小するだけなので即時に反映されます
6. **EXR 書き出し** でシーンリニアのまま保存します

ビュー操作：ドラッグで移動、ホイールでズーム、`F` でフィット、`B` で背景（入力画像）の表示切り替え、`1`〜`6` で表示モード切替、`P` でプレビューモード。背景をオフにして EXR を書き出すと、フレアだけの RGB（アルファなし）になり、加算合成用の素材として使えます。スライダーを動かしている間は自動で 1/4 程度の解像度・少ない光線数で計算し、離すと本来の解像度と品質で描き直します。描き終わったあとも手を止めている間は、瞳サンプルをずらした計算を足していき（**Ghost → Progressive passes**、既定 8 回）、点々が消えて滑らかになっていきます。ヘッダーに WebGPU が使っている GPU 名を表示します（内蔵 GPU やソフトウェア描画なら赤く表示）。ダブルクリックで手動光源（AE 版の Light Position）をその位置に置き、ハンドルをドラッグして動かせます。

> 動作環境：WebGPU が使える Chrome / Edge 113 以降（HDR 出力は 129 以降）。Safari / Firefox は WebGPU の対応状況次第です。

## ローカルで確認する

```bash
run.bat
```

`http://localhost:8080/` を開いてサーバを起動します（Node.js があれば `tools/serve.mjs`、無ければ PowerShell の簡易サーバ）。ポートを変えるなら `run.bat 9000`。ES モジュールと WebGPU のため、`index.html` を直接ダブルクリックでは動きません。

## GitHub Pages で公開する

ビルド工程はありません。静的ファイルをそのまま配信します。

1. このリポジトリを GitHub に push する
2. GitHub のリポジトリ設定 → **Pages** → **Source** を **GitHub Actions** にする
3. `main` に push すると [.github/workflows/pages.yml](.github/workflows/pages.yml) が `index.html / style.css / src / lenses` を配信します

[.github/workflows/parity.yml](.github/workflows/parity.yml) は push のたびに flaresim_nuke の C++ をビルドし、光学計算が原典と一致するかを確かめます。

## ファイル構成

| パス | 内容 |
| --- | --- |
| [index.html](index.html), [style.css](style.css) | UI |
| [src/optics.js](src/optics.js) | レンズモデル・Fresnel・ゴースト追跡（CPU）、ペア選別、瞳サンプリング、分光テーブル |
| [src/shaders.js](src/shaders.js) | WGSL：ゴースト追跡カーネル、ボックスブラー、ヘイズ、スターバースト、表示 |
| [src/gpu.js](src/gpu.js) | WebGPU レンダラ（バッチ分割、固定小数点の積算、読み戻し） |
| [src/sources.js](src/sources.js) | 明るい光源の検出、クラスタリング、手動光源 |
| [src/starburst.js](src/starburst.js) | 絞り形状の FFT による回折 PSF |
| [src/exr.js](src/exr.js) | OpenEXR 書き出し（自前実装） |
| [src/imageio.js](src/imageio.js) | 画像読み込み、テストシーン |
| [src/diagram.js](src/diagram.js) | レンズ断面図 |
| [lenses/lenses.json](lenses/lenses.json) | 1,370 本のレンズ設計データ（`tools/build-lenses.mjs` で生成） |
| [tests/](tests/) | C++ 原典との一致テスト |
| [docs/TECHNICAL.md](docs/TECHNICAL.md) | 技術解説 |
| [docs/UNREAL_RESEARCH.md](docs/UNREAL_RESEARCH.md) | Unreal Engine 対応の調査 |

レンズデータを作り直すとき：

```bash
node tools/build-lenses.mjs path/to/flaresim_nuke/lenses/lens_files lenses/lenses.json
```

## クレジットとライセンス

MIT License（[LICENSE](LICENSE)）。光学モデルとレンズ形式は Eamonn Nugent 氏の [blackhole-rt](https://github.com/space55/blackhole-rt)、Nuke 版は Steve Watts Kennedy 氏の [flaresim_nuke](https://github.com/LocalStarlight/flaresim_nuke)、AE 版の拡張（手動光源、Golden angle サンプリング）は ken-eizo 氏の [flaresim_ae](https://github.com/ken-eizo/flaresim_ae) に由来します。レンズ名は識別のためのもので、各メーカーがシミュレーション結果を保証するものではありません。本ツールは非公式の移植で、原作者の承認を受けたものではありません。
