[English](./README.md) · **日本語**

# Spectral Glass

> **ライブデモ**: <https://saqoosha.github.io/Spectral-Glass/>（WebGPU: Chrome/Edge 120+ または Safari 18+。HTML-in-canvas 背景: `CanvasDrawElement` フラグ付き Chrome のみ。それ以外のブラウザでは Picsum 単体にフォールバック。）

Apple "Liquid Glass" 風の浮遊ピル、三角プリズム、回転キューブ、揺れる波打つプレート、ラウンドブリリアントカットのダイヤモンドを通じて、**物理的に正確なスペクトル分散** をリアルタイムに描画する WebGPU デモ。Three.js の `MeshPhysicalMaterial.dispersion` を含む多くの Web 実装が使う「R/G/B の IOR をズラす」近似ではなく、可視光スペクトル全体を波長ごとにサンプルし、CIE 1931 色合わせ関数で最終色を再構成しています。

![Picsum の夕焼け写真の上で README ページを屈折させる4つのガラスキューブ](docs/images/demo-default.png)

上: デフォルトのオープニングシーン。回転する4つのガラスキューブ（`n_d = 1.272`、`V_d = 2.0`、屈折強度 `0.15`、透視 FOV 45°、N = 16）が、複合背景の上に乗っています。背景は HTML-in-canvas をサポートするブラウザでは「この README ページ自体（Chrome の `CanvasDrawElement` 試験 API でライブ HTML として）+ Picsum 写真」、サポートしないブラウザでは「同じ Picsum 写真のみ」。
すべてのキューブの明るいエッジに見えるスペクトル フリンジは、シェーダーが複合背景を波長ごとに 60 fps でリアルタイムに分割している様子です。上の夕焼けのような彩度の高い写真では分散が写真自身の色と混ざり合い、より平坦またはモノクロ寄りの Picsum 引きでは波長ごとの分割が純粋な虹として読み取れます。

## なぜ R/G/B をズラすだけではダメか

3サンプルの RGB IOR は、見るからに *3バンド* の虹になります。本物のガラスは連続したスペクトル。分散が強くなると違いがはっきり見えます — 3サンプル版はあらゆる屈折エッジに固い R/G/B フリンジを出し、8サンプルのスペクトル版は連続した虹に解像します。

![3サンプル vs 8サンプル スペクトル — キューブのエッジを拡大](docs/images/rgb-vs-spectral-zoom.png)

左: `N = 3`、ピクセルジッタと時間蓄積をオフにして 3バンド構造を露出させた状態。右: `N = 8` でスペクトルパイプライン全開（層化ジッタ + CIE 再構成 + EMA ヒストリ）。同じ回転ガラスキューブ（`n_d = 1.7`、`V_d = 4`）、同じグレースケール背景写真、同じフレーム — 違うのは波長ごとのサンプル数とスムージングパイプラインだけ。

ライブデモでは **`Z`** を押し続けると `N = 3` への強制 AND temporal jitter のピン留め（オフ）が同時に走ります — 3バンド RGB 構造は両方を切り替えてはじめて出てくるので、ホットキーが両方をまとめてトグルします。離すと設定サンプル数（デフォルトのオープニングシーンは `N = 16`）と Tweakpane のジッタ設定が復元されます。ジッタが有効だと、典型的な分散強度では `N = 3` でも `N = 8` に近く見えます — ピクセル毎ハッシュと EMA ヒストリがバンドを連続した虹に滑らかに戻すから。

## クイックスタート

```bash
bun install
bun run dev        # http://localhost:5173
bun run test       # 数学モジュールの Vitest
bun run build      # tsc --noEmit + vite build
```

WebGPU 対応ブラウザ（Chrome / Edge 120+、Safari 18+）が必要。スクロール可能な HTML テキスト背景は Chrome の HTML-in-canvas `CanvasDrawElement` パスを使います。サポートしないブラウザは自動的に Background コントロールを **Picsum only** に切り替えます。

## コントロール

| 入力 | アクション |
|---|---|
| シェイプをドラッグ | キャンバス上で動かす（cube / diamond は円形ヒット半径） |
| **`Z`**（押し続け） | `N = 3` に強制し、同時に temporal jitter もオフに固定 — 3バンド RGB 構造が実際に見えるように（離すと両方復元） |
| **Space** | アクティブなシェイプのインスタンスをランダム位置にシャッフル（pill / prism / cube / plate は 4 個、diamond は 1 個） |
| **`R`** | 新しいランダム Picsum 写真をロード（**Random photo** と同じ。**HDR env** がオフのときのみ） |
| **`T` / `S` / `B` / `F`** | ダイヤモンドの視点プリセット — **T**op（テーブルがカメラ向き）/ **S**ide（ガードルプロファイル）/ **B**ottom（キュレットがカメラ向き）/ **F**ree（自由回転）。他のシェイプではノーオペ。 |
| Tweakpane | IOR、Abbe、サンプル数、シェイプ（pill / prism / cube / plate / diamond）、寸法、波振幅 + 波長（plate のみ）、**diamond size** + view preset + **Wireframe** / **Facet color** + **TIR debug**（pink = バウンス予算枯渇でも refract still TIR、orange = analytic exit ミス）+ **TIR max bounces** 1…32（デフォルト 6、高くすると TIR ピクセルで重くなる）（diamond のみ）、屈折強度、投影（ortho / perspective）、FOV、temporal jitter、refraction mode、**Stop the world**（回転/波を凍結しつつ AA は収束を続ける）、**AA** モード — `None` / `FXAA`（単一フレーム空間フィルタ）/ `TAA`（サブピクセルジッタ + モーションベクタによるヒストリ再投影）、**Environment** — **HDR env** オン: Poly Haven パノラマ（1K/2K/4K）+ exposure + rotation + random、**オフ**: それらを隠して **Random photo**（Picsum 背景。反射用に従来の reflSrc パス）。HTML-in-canvas 対応 Chrome では **Background** が **Picsum only** と **Picsum + text (HTML)** を切り替え。 |
| プリセット | Subtle pill · Prism rainbow · Rotating cube · Wavy plate · Diamond |
| Materials | 実在ガラス10種（water → BK7 → SF flints → diamond → moissanite）+ ファンタジー4種（n_d 最大 3.5、V_d 最小 2） |

プリセット値は意図的に「主観的なスナップショット」として選んでいます:

| プリセット | インスタンス数 | 主な値 |
|---|---:|---|
| Subtle pill | 4 | `N=8`、`n_d=1.517`、`V_d=6.5`、屈折 `0.035`、pill `400×115×62`、edge `100` |
| Prism rainbow | 4 | `N=16`、`n_d=1.600`、`V_d=12`、屈折 `0.155`、prism `393×149×117` |
| Rotating cube | 4 | `N=16`、`n_d=1.272`、`V_d=2.0`、屈折 `0.150`、cube `230`、edge `44` |
| Wavy plate | 4 | `N=16`、`n_d=1.272`、`V_d=2.0`、屈折 `0.200`、plate `346×346×60`、edge `10.5`、wave `17 / 535` |
| Diamond | 1 | `N=16`、`n_d=2.418`、`V_d=55`、屈折 `0.200`、size `400`、free tumble、TIR bounces `6`、Brown Studio 2 `2K`、exposure `0.75`、rotation `-1.0995` |

すべてのプリセットは、透視 FOV 45°、Exact refraction、temporal jitter on、AA `None`、history alpha `0.5`、`paused` / `debugProxy` クリアを共通設定にしています。Diamond 以外のプリセットは HDR env を無効化し（スペクトル分割が写真に乗るように）、Diamond プリセットだけが上記表の Brown Studio 2 パノラマで HDR env を有効化します。

**Perf** パネルは、ブラウザが WebGPU `timestamp-query` を公開していればデフォルトで **GPU ms** を表示します（アダプタが対応していない場合は GPU 行が空になる — 想定動作）。`?perf=1`（または `?perf`）を付けると `window._perf` にもサンプルがログされます — **これは timestamp query が利用可能なビルドのみ**。それ以外ではログフックは仕込まれません。UI の **Show proxy** をオンにすると、すべての proxy フラグメントがピンクで塗られ、ラスタライズされたシルエットが見えます。

## 技術アプローチ

- **WebGPU + WGSL、2パス。** 安価なフルスクリーン bg パス（写真 + ヒストリ）の後に、ピル毎のインスタンス化された 3D キューブメッシュ proxy。重いピクセル単位の屈折シェーダーは proxy シルエット内のフラグメントだけで動作。バックフェイスカリング（CCW-outward 3D → Y フリップ後の CW NDC）により、カバーされた各ピクセルにつき正確に1回 invocation。
- **3D SDF、5シェイプ。** Pill（XY スタジアム + Z 方向の丸み）、prism（YZ の二等辺三角形を X 方向に押し出し）、回転キューブ（rounded box + フレーム毎の `rot * (p - center)` を `cubeRotation(time)` で）、揺れる **wavy plate**（厚みが一定の四角スラブで、中央面が `waveAmp · sin(kx+t) · sin(ky+t)` で Z 方向に曲がり、両面が中央面に追随することで厚みを均一に保つ）、**round brilliant cut diamond**（58ファセットのトルコフスキー理想ポリトープを D_8 折りたたみで5平面評価 + テーブルキャップ + ガードル円柱に圧縮、解析的バックエグジットと **設定可能な TIR バウンスチェーン**（1–32 内部反射、デフォルト 6）で特徴的な輝きを再現）。
  Cube、plate、diamond の proxy 頂点は `transpose(rot)` で変換され、ラスタライズされたシルエットがシェーダーの回転と完全に追随 — √3 のバウンディングボックスたるみが無い。Diamond は他シェイプの cube AABB ではなく専用の46三角ハル proxy を出すので、シャープなファセットシルエットが AABB スラックでフラグメントを浪費しません。
- **平行投影または透視投影。** UI トグル。Ortho は Liquid Glass のフラットな美学を保ち、perspective は `(w/2, h/2, cameraZ)` のピンホールカメラ（ユーザー指定 FOV から `cameraZ = (height/2) / tan(fov/2)` を導出）を使います。
- **Cauchy + Abbe IOR。** glTF の `KHR_materials_dispersion` 式による波長依存屈折率。
- **Wyman-Sloan-Shirley CIE XYZ**（JCGT 2013）の解析的近似 — ルックアップテーブル無し。
- **2面屈折。** 前面ヒットはプライマリスフィアトレース、背面エグジットは波長毎の inside-trace（Exact モード）または共有ヒーロー波長トレース（Approx モード、Wilkie 2014）。Cube、plate、diamond は inside-trace を完全にスキップ — バックフェイス エグジットが解析的（cube/plate はスラブ交差で、波長毎の SDF 評価が pill/prism より約 10× 少ない、diamond は全 57 ファセット平面 + ガードル円柱への ray-polytope テストで、SDF 評価ゼロ・約 60 dot 積のみ — 演算数では inside-trace と同程度だがマーチループも NaN しがちなグラディエントも無く、正確なファセット法線にアクセスできる）。
  Diamond の正確なファセット法線は、ファセットエッジでの有限差分縮退を解消し、これまでタンブル中に TIR ピクセルがちらついていた問題を修正しました。
- **波長毎の Fresnel。** 青波長は IOR が高い → Schlick Fresnel が高い → ダイヤモンドやプリズムに目に見える青いリム（高屈折率結晶の典型的な「ファイア」）。
- **波長毎の sRGB 重み付け。** 各サンプル写真ピクセルは `xyzToSrgb(cmf(λ))` で重み付けされ、短波長サンプルは青、長波長は赤に寄与。これにより屈折 UV が一致するときは写真の色を保ち、ずれる箇所では本物の色収差を生みます。
- **空間 + 時間ジッタ。** `hash21` で各ピクセルが波長フェーズを持ち、隣接ピクセルが異なる λ をサンプル — 目とヒストリ蓄積でノイズが平均化され、N=8 の層化が N=16 均一に見えます。**Temporal jitter** をオフにすると、スペクトルフェーズとヒーロー波長が固定され、オン/オフの差がジッタ済みシェーダーパスに隠されず可視化されます。
- **TIR フォールバック。** バックフェイスで `refract()` がゼロを返したとき、その波長は外部反射に振り替えられ、落ちません — キューブ内に黒い穴が空きません。**diamond**（exact 屈折のみ）では、まずバウンスチェーンが走ります: 最大 **N** 回の内部反射（UI "TIR max bounces"、1…32、デフォルト 6）、毎回現在のファセットで反射し、`diamondAnalyticExit` で次のエグジットを探し、外への refract を試行。これがブリリアントカットの輝きの源。チェーンが有効なエグジット方向を見つけずに尽きた場合、サンプルは silhouette `bg`（HDR マップが有効なら envmap-backed `reflSrc`）にブレンド。**Approx モード** はチェーンをスキップして共有ヒーロー `reflSrc` TIR パスを使い、`heroLambda` ジッタによるちらつきを回避。
- **HDR 環境マップ（統合シーン）。** 背景、屈折、反射のすべてが、本物のリニア HDR パノラマ（Poly Haven CC0 HDRI、studio / indoor / outdoor / sunset / night カテゴリから 1K / 2K / 4K 解像度でキュレート、デフォルト 2K）をサンプル。bg は視線方向（perspective ではスカイボックス風、ortho ではフラット）、屈折光線はエグジット方向（古典的 IBL 屈折で UV-parallax 近似を回避）、反射は反射方向。明るい HDR ハイライトが Fresnel リムを駆動 — 上から見たダイヤモンドが、平面写真では出せない「光を集めてクラウンから返す」典型的な輝きを見せます。従来の Picsum 写真 bg + UV オフセット屈折パスとの A/B 切り替えが可能、exposure と yaw スライダーで再ダウンロード無しに調整可能。
- **HTML-in-canvas 背景。** `GPUQueue.copyElementImageToTexture` を公開している Chrome ビルドでは、スクロール可能な DOM テキストパネルを、ガラスが屈折するのと同じテクスチャにラスタライズできます。API が無いか、リピート コピーが失敗する場合は **Picsum only** にフォールバックしつつ、スクロール可能・インタラクティブなまま動作。
- **時間蓄積。** `rgba16float` ピンポンヒストリと EMA ブレンド（α は **History α** スライダー経由でユーザーチューン可能、デフォルト 0.5、シーン変更フレーム後の 1 フレームだけ 1.0 でキューブの残像を防止）。**Stop the world** がシーンを凍結すると、ブレンドは漸進平均 α = max(1/n, 1/256) に切り替わり、ノイズは収束ランプで 1/√n で減少、最終的に 256 サンプルのスライディングウィンドウ（残差 ~6 %）で底打ち。1/256 の床は fp16 精度に必須 — それより小さい α だとサンプル寄与がゼロに丸められ、シルエットが何分もかけて黒くフェードします。詳細は `main.ts pausedFrames` を参照。
- **モーションベクタ再投影付き Temporal AA。** 各フレームのプライマリ光線はピクセル毎ハッシュでサブピクセルジッタ。ヒストリは `fragCoord + (projected_prev_world − projected_curr_world)` で読まれるので、ジッタが相殺され、回転駆動のワールドモーションだけが読みをずらします。静止シーンはピクセル中心で正確にヒストリを読む — 反復的なバイリニアブラーなし — 一方タンブルするキューブやプレートは動きの下で屈折テクスチャがシャープに保たれます。ホストは現在および前フレームの `cubeRot` / `plateRot` を事前計算しユニフォームとしてアップロード。Cube と plate は解析エグジット再投影を取得、pill / prism は再投影なしの読みにフォールバック。
- **ポストプロセスパス。** シーンはリニア `rgba16float` をキャンバスサイズの中間バッファに書き、第2パスが AA モードに応じてコピーまたは FXAA フィルタしてスワップチェーンへ。sRGB OETF はそこで一度だけ適用（スワップチェーンが既に `*-srgb` ならアイデンティティ）。FXAA とシーンが同じリニアピクセルを共有し、エンコーディングがシェーダーに散らばらないように。
- **FXAA（オプション）。** ポストパスで 9-tap FXAA 3.x — エッジ検出のためのルマは知覚（sRGB）空間、色のブレンドはリニア空間。TAA の代替: 時間ジッタ無し、ゴーストゼロ、エッジが少しソフト。1080p で `~0.3 ms`。
- **写真 mipmap。** アップロードされた写真は完全な mip チェーン付き（フルスクリーンブリットダウンサンプル）。波長毎の屈折サンプルは2項から LOD を選択: `-log2(cosT) - 1`（鋭角でのミニフィケーション）+ cube / plate での `(1 - max(|nLocal|)) · 8`（丸まったリムでの法線回転エイリアシング対策）。`[0, 6]` にクランプ。
- **localStorage 永続化。** 検証付きロード（NaN や偽 enum を拒否）、古い `taa: boolean` → `aaMode` のレガシー移行、トレイリングエッジ デバウンス保存、pagehide フラッシュ。

## プロジェクト構造

```
src/
├── main.ts                     フレームループ + glue（+ T/S/B/F ダイヤモンドビューホットキー）
├── math/                       純粋な数学モジュール（ユニットテスト済）
│   ├── cauchy.ts               波長 → IOR（glTF 定式化）
│   ├── wyman.ts                Wyman CIE XYZ 近似
│   ├── srgb.ts                 XYZ → リニア sRGB 行列 + OETF
│   ├── sdfPill.ts              3D ピル SDF（WGSL 版のミラー）
│   ├── sdfPrism.ts             三角プリズム SDF（WGSL 版のミラー）
│   ├── sdfCube.ts              Rounded box / cube SDF（WGSL 版のミラー）
│   ├── cube.ts                 タンブルキューブの rz·rx 回転列
│   ├── plate.ts                タンブルプレートの rx·ry 回転列
│   ├── diamond.ts              トルコフスキー理想ブリリアントカット比率、
│   │                           ファセット平面の導出、WGSL `const` 出力、
│   │                           タンブル + 固定ビュー回転行列
│   └── diamondExit.ts          解析 ray-polytope バックエグジットの JS ミラー
│                               （Phase B リグレッション参照）
├── hdr.ts                      ミニマルな Radiance .hdr デコーダ（Phase C）
├── envmap.ts                   HDR envmap テクスチャローダ + GPU アップローダ
├── envmapList.ts               キュレート済み Poly Haven HDRI スラグ + ランダムピッカー
├── htmlBgTexture.ts            HTML-in-canvas テクスチャコピー + サポート判定
├── persistence.ts              localStorage: 検証ロード、デバウンス保存、pagehide フラッシュ
├── photo.ts                    Picsum fetch → GPU テクスチャ（グラデフォールバック付）
├── pills.ts                    ピル状態 + シェイプ対応ポインタドラッグ
├── perfStats.ts                ローリング CPU/GPU HUD 統計
├── shapeParams.ts              シェイプ固有パラメータ → frame フィールド
├── spectralSampling.ts         Temporal-jitter / hero-wavelength フィールドビルダ
├── ui.ts                       Tweakpane バインディング（シェイプセレクタ、プリセット、マテリアル）
├── webgpu/
│   ├── device.ts               アダプタ + デバイス + エラーハンドラ
│   ├── history.ts              ピンポン ヒストリテクスチャ
│   ├── pipeline.ts             Bg + proxy パイプライン + 共有バインドグループ + encodeScene
│   ├── postprocess.ts          中間 rgba16f ターゲット + passthrough/FXAA パイプライン + encodePost
│   ├── mipmap.ts               フルスクリーンブリット mipmap ジェネレータ（photo.ts が使用）
│   ├── perf.ts                 GPU タイムスタンプハーネス（サポート時はデフォルト）
│   └── uniforms.ts             型付き uniform バッファライタ
└── shaders/
    ├── fullscreen.wgsl         フルスクリーン三角形バーテックスシェーダー
    ├── postprocess.wgsl        Passthrough + FXAA フラグメントシェーダー + sRGB OETF
    ├── dispersion/
    │   ├── frame.wgsl          Uniforms + envmap サンプリングヘルパ
    │   ├── sdf_primitives.wgsl Pill/prism/cube/plate SDF
    │   ├── scene.wgsl          シーン SDF 集約 + シェイプディスパッチ
    │   ├── trace.wgsl          スフィアトレース + 解析エグジット
    │   ├── spectral.wgsl       Cauchy IOR + CIE/Wyman スペクトルヘルパ
    │   ├── proxy.wgsl          インスタンス proxy バーテックスパス
    │   └── fragment.wgsl       Background + dispersion フラグメントシェーダー
    └── diamond.wgsl            ダイヤモンド固有 WGSL: `sdfDiamond`（D_8 折り）、
                                 `diamondAnalyticExit`（TIR バウンスチェーンが使う ray-polytope
                                 バックエグジット）、wireframe + facet-color + TIR debug オーバーレイ、
                                 正確な凸ハル proxy メッシュ、TAA ピルインデックスピッカー

tests/                          各数学モジュールの Vitest ユニットテスト
docs/
└── ARCHITECTURE.md             フレームパス、uniform レイアウト、SDF & トレーシング詳細
```

`src/math/` の数学モジュールは `src/shaders/dispersion/*.wgsl` および `src/shaders/diamond.wgsl` の関数と 1:1 でミラー — vitest スイート（`bun run test` 実行、現在 200 ケース、追加に応じてカウントは変動）がシェーダーの参照実装の役割を果たします。Diamond の平面係数はパイプラインビルド時に `diamond.ts` からシェーダーソースに注入されるので、ホスト側数学と GPU 側定数がドリフトしません。

## 設計

- [アーキテクチャノート](docs/ARCHITECTURE.ja.md) — モジュールマップ、フレームパス、uniform レイアウト、proxy メッシュ + カメラ、波長毎ループ（空間層化、波長毎 Fresnel、TIR フォールバック）、計測パフォーマンス

## パフォーマンス

Apple Silicon（1292×1073、画面上に4インスタンス、WebGPU `timestamp-query`、≥ 30 サンプルの p50）:

| 設定 | GPU 時間 |
|---|---:|
| pill N=8  | 1.70 ms |
| pill N=32 | 6.42 ms |
| cube N=8  | 1.05 ms |
| cube N=16 | 1.38 ms |
| cube N=32 | 1.97 ms |
| cube N=64 | 3.21 ms |

すべて 16.67 ms vsync 予算内。背景ピクセルのコストはほぼゼロ。pill / prism / cube / plate ピクセルでは波長毎ループが支配的。Cube と plate は同じ `N` で pill より目に見えて安価です — バックフェイス エグジットが解析的スラブ交差で、pill/prism がまだ払う波長毎スフィアトレースが要らないから。Plate はその上に 3 ニュートン反復を積んで波打つ表面に乗せます。Apple の TBDR は既に背景を効率的にカリングしているけれど、ディスクリート GPU はこの proxy パスからより多くの利益を得ます。

## 参考文献

1. Khronos. [**KHR_materials_dispersion**](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_dispersion/README.md) — ここで使う Cauchy + Abbe 定式化。
2. Wyman, Sloan, Shirley (2013). [**Simple Analytic Approximations to the CIE XYZ Color Matching Functions.**](https://jcgt.org/published/0002/02/01/) JCGT 2(2).
3. Wilkie et al. (2014). [**Hero Wavelength Spectral Sampling.**](https://jo.dreggn.org/home/2014_herowavelength.pdf) EGSR.
4. Peters (2025). [**Spectral Rendering, Part 2.**](https://momentsingraphics.de/SpectralRendering2Rendering.html)
5. Heckel. [**Refraction, dispersion, and other shader light effects.**](https://blog.maximeheckel.com/posts/refraction-dispersion-and-other-shader-light-effects/)

## ステータス

技術デモ / 手法の概念実証。ライブラリではありません。プロダクション Web サイト統合もしていません。スペクトル屈折テクニックを自分のプロジェクトに引き込みたい場合、見るべきファイルは `src/shaders/dispersion/`、`src/shaders/diamond.wgsl`、および `src/math/` のミラーヘルパ。

## クレジット

HDR 環境マップは [Poly Haven][1] からオンデマンドで取得され、各撮影者（主に Greg Zaal、Dimitrios Savva、Sergej Majboroda 他）が個別に制作したものです。すべての Poly Haven HDRI は [CC0][2] でリリース — 使用に attribution は不要ですが、感謝してここに列挙。彼らのバンドウィズを大量に使う場合は [Poly Haven をサポート][3] してください。

Picsum 背景写真は [Unsplash ライセンス][5] のもと [picsum.photos][4] から取得。

プロジェクトコード自体は [MIT ライセンス](LICENSE)。

[1]: https://polyhaven.com/hdris
[2]: https://creativecommons.org/public-domain/cc0/
[3]: https://www.polyhaven.com/support
[4]: https://picsum.photos/
[5]: https://unsplash.com/license
