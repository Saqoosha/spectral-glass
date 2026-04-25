[English](./ARCHITECTURE.md) · **日本語**

# アーキテクチャ

2 パスの WebGPU レンダラ。ピル毎の 3D proxy メッシュ内でピクセル単位の SDF スフィアトレース、その下に安価なフルスクリーン背景パス — 重い屈折シェーダーは proxy が実際にカバーするフラグメントだけで動作。

## フレームパス

```
┌──────────────────────────────────────────────────────────────────────┐
│  毎 RequestAnimationFrame:                                          │
│                                                                      │
│  1. 必要に応じて canvas + history + post-intermediate をリサイズ      │
│  2. params → pills（hx/hy/hz/edgeR）にプッシュ                       │
│  3. writeFrame → uniform バッファ（688 B: スカラー + 6×mat3 + plate │
│     + diamond 32B + envmap 16B ブロック + pills）                  │
│  4. シーンパス（intermediate(rgba16f) + history[write] に書き込み）:│
│     a. bg サブパス: フルスクリーン三角形 → fs_bg（active bg + history）│
│     b. proxy サブパス: インスタンス 3D proxy メッシュ → fs_main        │
│          フラグメント毎カメラ光線（ortho または perspective）        │
│          シーン SDF をスフィアトレース                                │
│          miss なら: bg を返す（オーバーカバー proxy フラグメント）  │
│          hit なら: ピクセル毎層化 λ ジッタ、各 λ について:           │
│                    refract → inside-trace → refract out             │
│                    photo を uv_λ でサンプル（TIR → reflect）        │
│                    波長毎 Fresnel ミックス                          │
│                    xyzToSrgb(cmf(λ)) で重み付けして累積             │
│                  history[read] と EMA ブレンド（historyBlend）      │
│                  リニアを intermediate @location(0) に書き込み       │
│                  リニアを history[write] @location(1) に書き込み     │
│  5. ポストパス（intermediate を読み swapchain に書き込み）:         │
│     aaMode === 'fxaa' ? fs_fxaa : fs_passthrough                   │
│     ここで sRGB OETF を一度適用（swapchain が非 sRGB なら）         │
│  6. history.current をフリップ                                       │
└──────────────────────────────────────────────────────────────────────┘
```

シーンとポストは1つのコマンドバッファにエンコードされ、まとめて submit されます（`main.ts` ループ）。Intermediate はキャンバスサイズに保たれ、キャンバスサイズが変わるたびに `resizeIntermediate` で再アロケート。

2つのパイプラインは1つの明示的なバインドグループレイアウトを共有します（`frame` がバーテックス・フラグメント両ステージから見えるように）。2つのバインドグループはパイプライン作成時に事前構築され（ヒストリ読み出しスロット毎に1つ）、`history.current` に基づいてスワップ — フレーム毎のバインドグループアロケーションは無し。

### Proxy メッシュ

`vs_proxy` はシェイプ毎の proxy メッシュを emit。pill/prism/cube/plate は `CUBE_PROXY_VERT_COUNT` 頂点の単位キューブ（36 頂点、12 三角、CCW-outward 巻き）を `halfSize` にスケール（SDF が既に `edgeR` を考慮）。Diamond は `DIAMOND_PROXY_VERT_COUNT` 頂点の正確な凸ハル（138 頂点、46 三角）を、`diamondProxyVertex` 内でトルコフスキー定数から合成（`src/shaders/diamond.wgsl` 参照）。Diamond 以外のシーンはデフォルトの4インスタンス、diamond シェイプ/プリセット切替えはライブインスタンスリストを1つにトリムしてブリリアントカットを単一オブジェクトとして表示。Draw コールは1インスタンスにつき `max(CUBE_PROXY_VERT_COUNT, DIAMOND_PROXY_VERT_COUNT)` 頂点を発行。`vs_proxy` 冒頭の `maxVerts` ガードが、diamond 以外のシェイプの上限範囲をオフスクリーン縮退ポジションにクリップ。
`shape == cube` / `plate` / `diamond` のときは頂点が `transpose(cubeRot)` / `transpose(plateRot)` / `transpose(diamondRot)` で変換され、ラスタライズされたシルエットがシェーダーのワールド空間シェイプと完全に一致。Plate は Z 半分エクステントを `waveAmp` だけ拡張し、波打つ表面が突き抜けないようにします。Pill と prism はワールド空間で軸整列なので、その AABB cube proxy が既にシェイプをタイトにカバー。Back-face culling（`frontFace: 'cw'`、Y フリップした投影が 3D から NDC への巻きを反転するため）により、カバーされた各ピクセルにつき 1 フラグメント invocation を残します。

### カメラ

投影は uniform フラグ（ortho / perspective）。Ortho ではピクセル毎光線が `ro = (px, py, 400)` から `-Z` に平行。Perspective ではカメラ `(w/2, h/2, cameraZ)` から発散し、ピクセルの `z = 0` ワールド点を通過。`cameraZ` は CPU でユーザー指定 FOV から導出: `cameraZ = (height/2) / tan(fov/2)`。`vs_proxy` の `projectWorld` が同じ変換をミラーし、ラスタライズされた三角形がフラグメントシェーダーの光線がトレースするのと同じピクセルに着弾。

## モジュール責務

| モジュール | 担当 |
|---|---|
| `src/webgpu/device.ts` | アダプタ / デバイス取得、canvas コンテキスト構成、リサイズ、`device.lost` + `uncapturederror` ハンドラ。 |
| `src/webgpu/pipeline.ts` | 共有明示バインドグループレイアウト付きの Bg + proxy パイプライン（async 作成）、事前構築バインドグループ2つ、`encodeScene` が呼び出し側エンコーダにシーンパスを記録。 |
| `src/webgpu/postprocess.ts` | ポストプロセスパイプライン（passthrough + FXAA）、キャンバスサイズの中間 `rgba16float` レンダーターゲット、ポスト UBO（sRGB フラグ）、バインドグループ + リサイズロジック、`encodePost` が呼び出し側エンコーダにポストパスを記録。 |
| `src/webgpu/mipmap.ts` | `photo.ts` がアップロード後の mipmap チェーン生成に使うフルスクリーン三角形ブリット。デバイス毎パイプライン + サンプラキャッシュ（`WeakMap<GPUDevice,…>`）。`pushErrorScope` が無効なパイプラインをキャッチし、悪いフォーマットがサイレントにキャッシュされないように。 |
| `src/webgpu/uniforms.ts` | `FrameParams` 型 + バッファライタ。モジュールスコープの `Float32Array` スクラッチ。 |
| `src/webgpu/history.ts` | ピンポン `rgba16float` テクスチャペア。リサイズで再作成。 |
| `src/webgpu/perf.ts` | アダプタが対応していれば GPU タイムスタンプクエリハーネス（ピンポン readback）— Tweakpane **GPU ms** がデフォルト。`?perf` URL で `window._perf` サンプルログを追加。シーンパスのみ。FXAA ポストパスは HUD に含まれません。 |
| `src/photo.ts` | Picsum fetch → `ImageBitmap` → mipmap 付き GPU テクスチャ。Fetch/decode 失敗時はグラデフォールバック（GPU アップロードエラーは `uncapturederror` に流す）。`destroyPhoto` が `main.ts` のキュードレイン後クリーンアップパスを担当。 |
| `src/htmlBgTexture.ts` | Chrome HTML-in-canvas サポートチェックと、`GPUQueue.copyElementImageToTexture` の GPU テクスチャへのアップロード。API が無いか paint コピーが繰り返し失敗すると `main.ts` が Picsum にフォールバック。 |
| `src/pills.ts` | ピル状態（ドラッグで変更）+ ポインタイベントライフサイクル（discriminated-union ドラッグ状態）。 |
| `src/ui.ts` | `Params` の Tweakpane バインディング、プリセット、マテリアルボタン、サポートゲート Background コントロール、シェイプ/プリセット駆動のインスタンス数同期。 |
| `src/main.ts` | すべてを配線、`try/catch` 内で RAF ループを実行、`photoRevision` でリロードレース保護、HTML 背景フォールバック、シェイプ対応インスタンス数（`diamond` = 1、他 = 4）を担当。 |
| `src/math/{cauchy,wyman,srgb,sdfPill,sdfPrism,sdfCube,camera,cube,plate,diamond,diamondExit}.ts` | 同名 WGSL でミラーされる純粋関数。Vitest スイートが参照。`cube.ts` / `plate.ts` / `diamond.ts` がタンブル回転（cube は rz·rx、plate は rx·ry、diamond は Rx·Ry）をホストで事前計算し、シェーダーが SDF 評価毎の cos/sin を回避。`diamond.ts` はトルコフスキー導出のファセット平面係数 AND 解析エグジットが反復する展開済み法線配列を含む WGSL `const` ブロックも生成 — 単一情報源。`diamondExit.ts` は ray-polytope 解析エグジットをミラーし、その挙動を GPU 無しで vitest リグレッションでピン留め。 |
| `src/hdr.ts` | Radiance .hdr (RGBE) デコーダ + ラウンドトリップエンコーダ。純粋 JS、GPU 依存なし — 合成 encode→decode ラウンドトリップでテスト。Poly Haven が出荷する adaptive-RLE フォーマット（幅 8-32767）をサポート。レガシー per-pixel RLE は明確なメッセージで throw。 |
| `src/envmap.ts`, `src/envmapList.ts` | HDR 環境パノラマローダ。`envmap.ts` が URL から fetch、`hdr.ts` でデコード、RGB float → RGBA half-float に変換（`F16_MAX_FINITE = 65504` クランプで明るい HDR ピクセルが +Inf にオーバーフローしてリニアサンプラ経由で NaN を播くのを防止）、リニアフィルタ IBL サンプリング用に rgba16f テクスチャにアップロード。`envmapList.ts` は Poly Haven CC0 HDRI を studio / indoor / outdoor / sunset / night カテゴリで 1K / 2K / 4K 解像度でキュレートし、UI Random ボタン用の `pickRandomSlug` ヘルパを公開。 |
| `src/shaders/dispersion/*.wgsl` | 分割シェーダーバンドル（`pipeline.ts` 結合順序参照）: `frame`（uniforms + envmap sample）、`sdf_primitives`、`scene`（sceneSdf 集約）、`trace`（スフィアトレース + 解析エグジット）、`spectral`、`proxy`（`vs_proxy`）、`fragment`（`fs_bg` / `fs_main`）。物理分割は保守性のためで、GPU モジュールも `fs_main` エントリも 1 つのまま。 |
| `src/shaders/diamond.wgsl` | ダイヤモンド固有ジオメトリ: sdfDiamond、`diamondAnalyticExit`、`diamondAnalyticHit` / `diamondAnalyticHitScene`（インスタンス間の前面ヒットピッカー）、diamondProxyVertex、ファセットデバッグヘルパ。`sdf_primitives` と `scene` の間に結合され、`sceneSdf` が `sdfDiamond` を呼べるように。 |
| `src/shaders/postprocess.wgsl` | Passthrough + FXAA フラグメントシェーダー。FXAA はエッジ検出のため知覚（sRGB）ルマで動作し、リニア空間で色をブレンド。Swapchain が非 sRGB のとき sRGB OETF を適用。 |
| `src/persistence.ts` | スキーマバージョン付き localStorage 読み書き、フィールド検証、レガシー `taa: boolean` → `aaMode` 移行、トレイリングエッジ デバウンス保存（+ pagehide 用 `flush()`）。 |

## Uniform レイアウト

WGSL `Frame` 構造体を正確にミラー（std140 風ルール）:

```
offset   0 │ resolution.xy,  photoSize.xy                        (16 B)
offset  16 │ n_d, V_d, sampleCount, refractionStrength           (16 B)
offset  32 │ jitter, refractionMode, pillCount, applySrgbOetf    (16 B)
offset  48 │ shape, time, historyBlend, heroLambda               (16 B)
offset  64 │ cameraZ, projection, debugProxy, taaEnabled         (16 B)
offset  80 │ cubeRot:        mat3x3<f32>                         (48 B)
offset 128 │ cubeRotPrev:    mat3x3<f32>                         (48 B)
offset 176 │ plateRot:       mat3x3<f32>                         (48 B)
offset 224 │ plateRotPrev:   mat3x3<f32>                         (48 B)
offset 272 │ diamondRot:     mat3x3<f32>                         (48 B)
offset 320 │ diamondRotPrev: mat3x3<f32>                         (48 B)
offset 368 │ waveAmp, waveFreq, waveLipFactor, sceneTime         (16 B)
offset 384 │ diamondSize, diamondWireframe, diamondFacetColor,   (32 B)
           │   diamondTirDebug, diamondTirMaxBounces, _pad×3
offset 416 │ envmapExposure, envmapRotation, envmapEnabled,      (16 B)
           │   _envmapPad
offset 432 │ pills[0..8]     各 pill は:                          (32 B 各)
           │   center.xyz, edgeR,   halfSize.xyz, _pad
```

合計 688 バイト（80 B ヘッド + 6 × 48 B 回転行列 + 16 B plate wave/scene-time ブロック + **32 B** diamond パラメータブロック + 16 B envmap パラメータブロック + 8 × 32 B pills）。Uniform サイズは固定 — `pillCount` を超える pills はゼロ。

- `shape` は SDF を選択（0=pill、1=prism、2=cube、3=plate、4=diamond）。
- `pillCount` がインスタンスループをゲート。UI シェイプ変更とプリセットクリックで正確に保たれる: `diamond` は 1 インスタンス、他のシーンプリセットは 4。
- `time` はノイズストリーム — wall-clock 秒、シーンが pause していても常に進行し、TAA ジッタと波長層化がフレーム間で decorrelate し続ける。
- `sceneTime` はモーションストリーム — フレーム毎の `dt` で累積、`params.paused` のときスキップ。回転行列と plate の波フェーズを駆動。`time` から分離されているので "Stop the world" がモーションを凍結しても AA 収束は凍結されない。
- `cubeRot` / `plateRot` / `diamondRot` は現在フレームの回転（cube は rz·rx、plate は rx·ry、diamond は固定 20° 前傾の Rx·Ry）。`sceneTime` からホストで事前計算され、シェーダーは各 SDF 評価で複数の cos/sin の代わりに mat-vec を1回するだけ。`cubeRotPrev` / `plateRotPrev` / `diamondRotPrev` は前フレームの `sceneTime` から計算した同じ行列 — `reprojectHit` に渡され TAA モーションベクタヒストリ読みを駆動。Pause 時は現在行列と等しいので、再投影がアイデンティティに崩壊し、ヒストリ読みがピクセル中心に着弾（反復的バイリニアブラー無し）。Diamond の固定ビュープリセットでは、`uniforms.ts` が同じ canonical pose 行列を `diamondRot` と `diamondRotPrev` の両方に書き、ゼロモーションベクタを生成して凍結シェイプが TAA でにじまないように。
- `diamondSize` はピクセル単位のガードル直径（`ui.ts` のスライダー）。`diamondWireframe` / `diamondFacetColor` / `diamondTirDebug` はデバッグフラグ。`diamondTirMaxBounces`（1…32、デフォルト 6）が exact モードの TIR バウンスループの上限。TIR debug は未解決ピクセルをホットピンク（バウンス予算枯渇、refract out もまだ TIR）かオレンジ（`diamondAnalyticExit` ミス）で塗る。`shape != diamond` のときはすべて無視。
- `taaEnabled` がテンポラルアンチエイリアシングをトグル。オン時、`fs_main` がプライマリ光線をピクセル毎ハッシュで ±0.5 px の範囲でジッタし、`fragCoord + (projected_prev_world − projected_curr_world)` でヒストリを読む — デルタ内でジッタが相殺し、静的シーンはピクセル整列ヒストリを読み、動くシェイプは屈折テクスチャをシャープに保つ。`Params` の `aaMode === 'taa'` で駆動（`src/ui.ts` 参照）。`aaMode === 'fxaa'` はポストパスで FXAA を実行、`aaMode === 'none'` はどちらも無し。
- `waveAmp` / `waveFreq` が plate の中央面変位を駆動（`waveAmp · sin(waveFreq · x + 2·sceneTime) · sin(waveFreq · y + 2·sceneTime)`）。他シェイプでは無視。
- `waveLipFactor = 1/√(1 + (waveAmp·waveFreq)²)` は事前計算された Lipschitz 安全係数で、`sdfWavyPlate` がその出力に乗じ、スフィアトレース ステップが SDF 評価毎の `inverseSqrt` を回避しつつ真の距離内に留まる。波の2つの偏微分が同時に `amp·k` に達することは無いので、この境界はタイト — 詳細は `src/webgpu/uniforms.ts` の導出を参照。デフォルトでは古い hardcoded 0.6 に対し ≈ 0.92 — 同じ安全マージンでステップあたり ~53 % 多く進む。
- `historyBlend` は **History α** スライダー（現在 `defaultParams()` と全プリセットで `0.5`、Misc フォルダでユーザーチューン可能）をデフォルトとして steady state に使用、シーン変更後の 1 フレーム（プリセットクリック、写真リロード、シェイプ切替え、ピルシャッフル、pause トグル）だけ 1.0 になり、古いテンポラルヒストリが残らないように。"Stop the world" 中は漸進平均 `α = max(1/n, 1/256)` に切り替わり、ノイズが収束ランプで 1/√n で減少し、最終的に 256 サンプルのスライディングウィンドウ（残差 ~6 %）で底打ち。1/256 の床は `rgba16float` ヒストリテクスチャに必須: それより小さい α は新サンプル寄与を fp16 量子（中間グレー周辺で ≈ 0.0005）以下に押し下げ、高コントラストエッジピクセルで寄与が 0 に丸められ、`(1 − α) · prev` の減衰で何分もかけてシルエットが黒線にフェードしてしまう。詳細は `main.ts pausedFrames` を参照。
- `heroLambda` はフレーム毎にジッタされた波長 [380, 700] — Approx モードが共有バックフェイストレースで使用。**Temporal jitter** がオフのとき、`spectralSamplingFields()` が 540 nm に固定し `jitter = -1` を書く。WGSL は negative jitter を「各スペクトル層の中心を使う」と扱い、トグルが安定で見える A/B 状態を生成。
- `cameraZ` / `projection` が ortho vs perspective を駆動（CPU が UI の FOV と canvas height から `cameraZ` を導出）。
- `debugProxy` が視覚チェックのため全 proxy フラグメントをピンクに塗る。
- `applySrgbOetf` はレイアウトパリティのため UBO スロットに残されているが、シーンシェーダーはもう読まない — ポストパスがすべての sRGB エンコーディングを所有。ホストはまだ 0/1 をスロットに書き、`tests/uniformsLayout.test.ts` を満足させる。スロット回収には そのテスト + `uniforms.ts` + WGSL 構造体を一緒に触る必要あり。

## ポストプロセスパス（AA / sRGB OETF）

シーンパスはリニア RGB をキャンバスサイズの `rgba16float` 中間バッファに書く。第2レンダーパスがその中間を読み、swapchain に書く:

- `aaMode === 'none'` → `fs_passthrough` がコピーし sRGB OETF を適用（swapchain が既に `*-srgb` ならアイデンティティ）。
- `aaMode === 'fxaa'` → `fs_fxaa` が 9-tap FXAA 3.x スタイルの空間フィルタを実行（エッジ検出に sRGB 空間ルマ、色のブレンドにリニア）、その後同じ sRGB エンコード。
- `aaMode === 'taa'` → `fs_passthrough` 再び。TAA は既にシーンパスで実行（サブピクセルジッタ + モーションベクタヒストリ再投影）、ポストはエンコード以外何もしない。

両シーンカラーターゲット（`@location(0)` 中間、`@location(1)` ヒストリ）を `rgba16float` に保つことで、FXAA がヒストリ EMA が既に見ているのと同じリニアピクセルで動作でき、sRGB OETF が一箇所に集約される。ポスト UBO は 16 B（`applySrgbOetf` フラグ1つ + 3 パディングフロート）。フラグだけが起動時に書かれる — セッション寿命中定数。

`src/webgpu/postprocess.ts` が中間のライフサイクルを所有。テクスチャはリサイズで同期的に解放されます — WebGPU は中間をカラーアタッチメントとして既に名指ししているコマンドバッファから強い参照を保持しているからです（写真リロードはフレームをまたいで写真がサンプルされるので `queue.onSubmittedWorkDone` をドレイン。中間は単一フレーム）。

## 写真 mipmap

`src/photo.ts` は `src/webgpu/mipmap.ts` のフルスクリーン三角形ブリットパイプラインで生成された完全な mip チェーン付きで写真をアップロード。フラグメントシェーダーの波長毎屈折サンプルは2項に基づいて LOD を選ぶ:

- **Grazing incidence** — `-log2(cosT) - 1` — サンプルフットプリントが鋭角で `~1/cosT` で増加。極端な角度で項が境界内に留まるよう、cosT は log 内で ≥ 0.02 にクランプされる。
- **Rounded-rim curvature** — `(1 - max(|nLocal|)) · 8` — cube / plate の丸まったエッジでは前面法線が `edgeR` 幅のスクリーン領域で ~90° 回転し、`cosT` から独立に屈折 UV ヤコビアンを膨張させる。Pill / prism はこの項をスキップ。

2項の和は `[0, 6]` にクランプ。個別の項は 0 で床打ちされない。ほぼヘッドオンのピクセルは負の grazing 寄与を出して曲率ブーストの一部を相殺し、最終クランプが残りを捕まえる。サンプラのトリリニアフィルタリング（`mipmapFilter: 'linear'`）がサブレベルブレンディングを与える。背景サンプル（bg + reflection フォールバック）はスクリーンピクセルと 1:1 なので LOD 0 に留まる。

## なぜ波長毎 sRGB 重み付け？

教科書的なパス — `cmf(λ) * L(λ)` を XYZ に累積、その後 `XYZ → sRGB` を一度 — は、`L(λ)` が写真の輝度から導出されたスカラーの場合に崩壊: chroma がすべて失われ、残る色は `xyzToSrgb(sum(cmf))` が出すもの（フラットホワイトスペクトルのわずかなサーモンティント、CMF の和が D65 でないため）だけ。

波長毎重み付け — `xyzToSrgb(cmf(λ)) * L_rgb(uv_λ)` — は、各波長に独自の sRGB プライマリ色を与え、フラット UV ケースで写真 RGB を保つ:

- 一様入力（全 λ で同じ UV）: `L * sum(lambdaRgb) / sum(lambdaRgb) = L`。chroma を正確に保つ。
- 変化入力（λ 毎に異なる UV）: 赤波長サンプルが R チャネル、青が B に寄与、古典的色収差。

正規化分母は同じ波長毎プライマリ和なので、出力はどんな `N` でもニュートラルに保たれる。

## SDF とスフィアトレーシング

5 シェイプ。`sceneSdf` が `shape` uniform でディスパッチ:

- **Pill** — XY の 2D スタジアムシルエット（`roundedBox` を `edgeR` で縮め、最短縮小半軸で丸める）、Z に同じ rounded-corner トリックで `|z|` に押し出し。
- **Prism** — **YZ** の二等辺三角形断面（頂点 +Z、底辺 −Z）を X に押し出し。`halfSize.x` が押し出し長、`halfSize.y` が底辺半幅、`halfSize.z` が頂点高。上から見たシルエットは長方形。三角形の傾いた YZ 面が光線を横方向に曲げ、写真のコントラストエッジで古典的プリズム虹を生む。
- **Cube** — 標準 rounded box。`local = frame.cubeRot * (p - center)`。`cubeRot` は X+Z 周りに 0.31 + 0.20 rad/s でタンブルする rz·rx 回転。ホストで `time` から計算され、各 SDF 評価が 1 mat-vec だけになるように uniform でアップロード。
- **Plate** — 厚い四角スラブを **厚さ一定の曲げシート** として扱う。Plate のローカル（回転後）フレームで、中央面 `z* = waveAmp · sin(waveFreq · x + 2t) · sin(waveFreq · y + 2t)` を定義。両 Z 面が中央面に追随するので、スラブ厚はどこでも均一（厚さがパルスする素朴な `sdBox - wave` ではない）。SDF は `sdBox(p with z ← p.z − z*) × waveLipFactor` — Lipschitz ファクタが z シフトが x/y に追加する追加グラディエントを補正。回転は `plateRot`（rx·ry が 0.30 + 0.20 rad/s、レート比は互いに素な小整数 (2:3) を選んでいて、合成方向は ~63 秒で繰り返す — 実用上「ループしない」と読める長さ。明示的な周期導出は `src/math/plate.ts` 参照）としてホストで事前計算。
- **Diamond** — round brilliant cut を **トルコフスキーの 1919 年「理想」プロポーション** にピン留めした凸ポリトープ: 53 % テーブル（vertex-to-vertex、GIA "bezel point" 規約）、34.5° クラウン、40.75° パビリオン、22° スター、39.9° アッパーハーフ、42° ロワーハーフ、2 % ガードル厚。58 ファセットが D_8（八角）対称折りたたみで 7 距離項に縮約され、`max(...)` の半空間距離で SDF を得る。折りたたみは 8 重方位反復を π/8 ウェッジに 3 ミラーリフレクション（x の abs、y の abs、`y = x · tan(π/8)` 線でリフレクト）で縮約。平面係数は `src/math/diamond.ts` で導出され、`const` 宣言としてシェーダーに注入されるので、ホスト数学と GPU 定数がドリフトしない。アッパーハーフ角度は 39.9°（物理カット仕様が通常引用する 42° ではなく）にピン留め — 平面が φ = 0 の実際のガードル縁にアンカーされる場合、bezel-star-UH 3 方向接合がウェッジ内に座る必要があり、40° を超えると π/8 ミラーから逃げ bezel カイトがコーナーで閉じなくなる。`diamond.test.ts` がコーナー通過不変条件と 40° 上限の両方をピン留め。回転は `Rx·Ry`（固定 20° 前傾 + 垂直 Y 軸スピン）で、`diamondRot` としてホストで事前計算 — cube/plate と同じ uniform パターン。4つのビュープリセット（`free` / `top` / `side` / `bottom`、`T` / `S` / `B` / `F` ホットキーにバインド）が、参照イラストとファセットジオメトリをクロスチェックするための canonical pose で `diamondRot` をスワップ。2つのデバッグオーバーレイ — `diamondWireframe`（top-two-plane-gap smoothstep でファセットエッジ描画）と `diamondFacetColor`（ファセットクラス毎フラットシェード: table=red、bezel=green、star=blue、upper-half=yellow、girdle=cyan、lower-half=magenta、pavilion=orange）— が屈折で信号を濁さずカバレッジ + 隣接性を表面化。SDF コードは `src/shaders/diamond.wgsl` に住み、`diamondAnalyticHit` / `diamondAnalyticHitScene`（diamond インスタンス間の解析的前面ヒットピッカー）と `diamondProxyVertex` の正確な凸ハル proxy メッシュ（46 三角: 6-tri テーブル fan + 16-tri クラウン台形 + 16-tri ガードル帯 + 8-tri パビリオン円錐）と並んでいる。
  バックフェイス エグジットは **解析的**（`diamondAnalyticExit`、これも `src/shaders/diamond.wgsl`）: 光線を 57 個の展開ファセット平面（8 bezel + 8 star + 16 upper half + 16 lower half + 8 pavilion + 1 table cap）すべてとガードル円柱に対してテストし、最小正の t が勝つ。出てくる正確なファセット法線は、これまで TIR ピクセルを外部反射フォールバックに送り、タンブル中に「他の面が突然現れる」アーティファクトを生んでいた、ファセットエッジでの有限差分グラディエント縮退を回避。TIR では diamond パスが **bounded internal bounce loop** を実行（`diamondTirMaxBounces` からカウント、小さな origin nudge と miss ガード付き）し、ブリリアントカットの輝きが意味のあるパスを使うように。チェーンは exact モードのみ — approx モードの共有 hero-wavelength エグジットは `heroLambda` ジッタでちらつくので、approx は Phase A の `reflSrc` TIR フォールバックを保つ。
  ランタイム シェイプ/プリセット同期は意図的に diamond インスタンスを 1 つレンダリング。pill/prism/cube/plate は 4 インスタンスレイアウトを保つ。

スフィアトレースはピクセル毎光線 origin と方向（上記 Camera 参照）から開始、`HIT_EPS = 0.25` と `MIN_STEP = 0.5` でマーチ。Pill と prism では、バック表面エグジットが `-sceneSdf` をマーチして得られる（inside-trace）、`maxInternalPath()`（生きている任意の pill を貫く最長コード）でキャップ。**Cube**、**plate**、**diamond** はショートカットを使う:

- `cubeAnalyticExit` — cube の回転後ローカルフレームでの ray-box スラブ交差、その後 rounded-box SDF で 2 ニュートンステップで丸まったリムにスナップ。波長毎 48 反復 inside-trace と 6 評価 finite-diff 法線を O(1) 閉形式に置き換え。Cube ケースで約 7–8× 高速。
- `plateAnalyticExit` — plate の回転後ローカルフレーム（XYZ、最早エグジット軸を選択）での ray-box スラブ交差、その後 Z 面エグジットなら真の波打つ表面 `z = faceSign · halfZ + z*(x,y)` に対して t を 3 ニュートンステップで精緻化。X/Y 面エグジットはフラット（精緻化無し）。波打つ Z 面の法線は `(−faceSign · ∂z*/∂x, −faceSign · ∂z*/∂y, faceSign)` を正規化、最終ニュートンステップから回収した cos(kx+2t) / cos(ky+2t) を使用。Cube と同じ ~10× の波長毎 SDF 評価削減。
- `diamondAnalyticExit` — 光線を 57 個の展開ファセット平面（8 bezel + 8 star + 16 UH + 16 LH + 8 pavilion + 1 table cap）すべてとガードル円柱（t に関して二次）に対してテストし、最小正の t が勝つ。Cube と plate と違ってニュートン精緻化は無い — ポリトープは区分線形なので、最初のヒット平面の方程式が正確なエグジットそのもの。平面の正確な外向き法線を返し、これまで TIR ピクセルを前面反射ハックに送っていたファセットエッジ有限差分縮退を解決。`src/math/diamondExit.ts` の JS ミラーが挙動をピン留め。

法線は pill/prism ではシーン SDF の中心差分から得られる — シェードされたピクセル毎に 6 追加 SDF 評価（軸毎 1 ペア）、安価。Cube と plate はエグジットポイントで解析グラディエントを使う（finite-diff が収束する値と一致）ので、丸まったリム / 波打つ表面はソフトな屈折を保つ。

### 波長毎ループ

```
for i in 0..N:
  pxJit  = hash21(pixel ⊕ time) - 0.5      // 符号付き [-0.5, 0.5) — 下記注参照
  λ      = mix(380, 700, (i + 0.5 + pxJit) / N)
  ior    = cauchyIor(λ, n_d, V_d)
  r1     = refract(-z, nFront, 1/ior)
  // エントリは r1 に沿って 1 MIN_STEP 内側にバイアス — `h.p` は表面上に住み
  // （HIT_EPS 内）、解析エグジットのスラブ数学 (h - roL) / rdL が
  // 正確な境界エントリで 0/0 = NaN を計算してしまうから。
  roEntry = h.p + r1 * MIN_STEP
  if approx:       (pExit, nBack) ← heroLambda での共有バックフェイストレース
  else if cube:    (pExit, nBack) ← cubeAnalyticExit(roEntry, r1, analyticIdx)
  else if plate:   (pExit, nBack) ← plateAnalyticExit(roEntry, r1, analyticIdx)
  else if diamond: (pExit, nBack) ← diamondAnalyticExit(roEntry, r1, analyticIdx)
  else:            pExit = insideTrace(h.p, r1, internalMax), nBack = -sceneNormal(pExit)
  r2     = refract(r1, nBack, ior)
  // 失敗モードルーティング — 下記「失敗モードフォールバック」参照。
  L      = TIR && diamond && !approx ? bounce_chain(r1, nBack, pExit; cap=diamondTirMaxBounces) :
           TIR                        ? reflSrc :
           NaN || OOB                 ? bg      :
                                        photo[uv_with_offset(r2)]
  F_λ    = schlickFresnel(cosT, ior)  // 波長毎 Fresnel
  accum += mix(L, reflSrc, F_λ) * xyzToSrgb(cieXyz(λ))
```

ピクセル毎層化（`hash21`）は隣接ピクセルが異なる波長を選ぶことを意味する — 目とテンポラル蓄積で空間ノイズが平均化され、N=8 層化が N=16 均一に見える。

`pxJit` の `- 0.5` は各層にジッタを中心化し、`t = (i + 0.5 + pxJit)/N` が `[i/N, (i+1)/N)` 内に留まるようにする。このシフトが無いと、最後の層 (i = N-1) が `t = 1` を越えて λ > 700 nm の見えない範囲に溢れ、CIE マッチング関数が事実上ゼロのところで、N=3 の場合 ~30 % のピクセルが赤サンプルを失い、再正規化器がフラットホワイト背景を黄色に変えてしまう。同じ off-by-half-stratum は大きな N でも存在した（N=8 で 20 nm オーバーフロー、N=16 で 10 nm）が、ヒストリ蓄積でアーティファクトがマスクされていた。

### 失敗モードフォールバック

波長ループはバックフェイスサンプルで 3 種類の失敗モード（TIR、NaN、UV out-of-bounds）を持つ。TIR は 2 行に分かれる（diamond exact モードのバウンスチェーン vs 残り全部 reflSrc にフォールバック）ので、下表は合計 4 つの TIR/NaN/OOB 行をリスト:

| 失敗 | 検出 | フォールバック | 理由 |
|---|---|---|---|
| Real TIR (non-diamond) | `dot(r2, r2) < 1e-4` AND not NaN | `reflSrc` | 物理的に正しい — 波長は前面で完全に反射される |
| Real TIR (diamond, exact mode) | `shape == diamond` AND `!useHero` | `diamondTirMaxBounces` までのバウンスループ（デフォルト 6、最大 32）: 各ステップで反射、nudge した origin で `diamondAnalyticExit` を呼び、`refract` out を試行。成功で photo/envmap をサンプル。チェーンが尽きたら `bg` か envmap `reflSrc`（そのパスの古い facet-unrelated `reflSrc` スタンドインではない）にブレンド。`diamondTirDebug` が枯渇ピクセルをピンク（まだ TIR）かオレンジ（解析エグジット ミス）に塗る。 |
| Real TIR (diamond, approx mode) | `shape == diamond` AND `useHero` | `reflSrc`（Phase A フォールバック） | Approx モードは hero のエグジットを λ 間で共有。共有 origin から heroLambda がフレーム間でジッタしながらバウンスチェーンを実行すると TIR 境界ちらつきが出る。ここは λ 毎ソリューション（Phase C）が来るまで reflSrc に留める。 |
| NaN r2 | `r2dot != r2dot`（自己比較で NaN を捕まえる） | `bg`（ローカルピクセルの photo サンプル） | ミスパス隣接と同じ色をレンダして、シルエットがクリーンに保たれる代わりに 1 個の明るい反射サンプルが見えないように |
| UV out of bounds | `coverUv(uvOff)` が [0, 1]² 内にない | `bg` | ミラーリピートサンプラがそうでなければ大きく外れた UV を無関係な写真領域に折り返す（明るい写真 → 白いスペックル、暗い → 黒） |

前面 / プリループ bg フォールバックゲートは似ていて、スフィアトレース ミス、縮退グラディエント `sceneNormal`、plate 面接合クリース（`plateCreaseAt`）をカバーする — すべて同じ「周囲 bg / シルエット隣接と一致」理由で bg にルーティング。

`reflSrc` 自体も OOB チェックを取得し（反射 UV `h.p.xy + refl.xy * 0.2` が写真外に着地する可能性）、反射サンプルがそうでなければミラーリピートでガベージになる場合は `bg` にフォールバック。

### Approx (hero wavelength) モード

`refractionMode == approx` はフレーム毎に 1 inside-trace を `frame.heroLambda` で実行し、そのエグジットポイント/法線を全 N 波長で共有。テンポラル蓄積がフレーム毎エラーを平均化。テクスチャバンドウィズが Apple Silicon で支配的なので TBDR でわずかな高速化（高 N で ~10-15 %）。

## エラーハンドリング

- `device.lost` + `uncapturederror`: ログ + `#fallback` で表示。
- シェーダーコンパイルエラー: `getCompilationInfo()` がすべてのメッセージをログし、エラーで throw（デフォルト WebGPU はこれらを不透明なバリデーションエラーに飲み込む）。
- 写真 fetch 失敗: バンドルされたグラデーションテクスチャへのグレースフルフォールバック（例外ではない）。
- レンダーループ例外: `try/catch` がエラーを表面化しサイレントに固まる代わりにループを停止。
- リロードレース: 単調 `photoRevision` カウンタが古い async 結果を破棄。古いテクスチャは `queue.onSubmittedWorkDone` 後にのみ破棄。
- タイピングターゲットホットキーフィルタ: Tweakpane の数値入力でのポインタイベントは `Space`/`Z`/`R` を発火しない。

## テスト

数学モジュールと uniform 配線はユニットテスト済（現在 200 テスト、すべてパス — 正確なカウントは新ケース追加でドリフト、`bun run test` 参照）:

- `cauchyIor` の d 線、単調性、`V_d` 感度、1.0 クランプ。
- `cieXyz` の 555 nm 付近の Y ピーク、650 nm での赤支配、450 nm での青、UV/IR でほぼゼロ。
- `xyzToLinearSrgb` の D65 white、Y のみ輝度バイアスグレー。
- `linearToGamma` のアイデンティティ端点、リニアセグメント、累乗カーブセグメント。
- `sdfPill3d` の符号、対称性、トップ面ゼロ交差、丸まったエッジ平滑性。
- `sdfPrism` の内部符号、遠方場正性、頂点/底辺エッジ値、両ミラー対称、頂点狭まり。
- `sdfCube` の内部、遠方場、面ゼロ交差、対称性、丸まったコーナー平滑性。
- `cameraZForFov` の 60°/90°、FOV 単調性、height 線形性、スライダー境界。
- `cubeRotationColumns` の t=0 アイデンティティ、正規直交性、元の rz·rx 導出と一致、WGSL パッド付きレイアウト、パッドスロットがゼロ、非有限時間を拒否。`plateRotationColumns` / `diamondRotationColumns` がそれぞれの rx·ry と Rx·Ry 合成で同じ不変条件に従う。
- Diamond ジオメトリ — トルコフスキー定数の妥当性（クラウン/パビリオン高さ比、合計高さ 0.55–0.65）、ファセット平面法線が単位長、bezel/pavilion の Y 成分がゼロ（φ=0 軸）、upper+lower half が φ=0 と φ=π/8 の両方の共有ガードルリムコーナーを通過（外接八角形リムへのアンカー回帰を捕捉）、UH 傾きが 40° ウェッジ妥当性上限以下に留まる、LH 傾きがパビリオン角度以上で表面化。
- Diamond 展開平面配列（Phase B）— クラス毎カウント（8 / 8 / 16 / 16 / 8）、最初のエントリが基本ウェッジ平面と一致して SDF と解析エグジットが一貫、法線が単位長 + クラス内で共有傾き、連続法線が予期される回転ステップ（π/4 または π/8）で異なる、各平面が指定共有ガードルコーナーを通過（解析パスでも「ファセットが共有コーナーで会う」不変条件をピン留め）。
- Diamond 解析エグジット（`diamondAnalyticExit` JS ミラー）— 軸整列光線が予期されるクラスと法線でキュレット / テーブルからエグジット、z=0 の水平光線がガードル円柱からエグジット、ガードル帯のすぐ上/下の光線がクラウン/パビリオン ファセットからエグジット（帯拒否テスト）、エグジット法線が常に `dot(n, rd) > 0` を満たす（光線が半空間を出る）、エグジットポイントが報告ファセットの平面方程式を満たす（which-class と what-point の一貫性）、垂直光線がシリンダの `a > 1e-6` ガードが発火しても table cap を見つける。
- Diamond ビュープリセット — `top` がローカル +Z をワールド +Z として保持、`side` がローカル +Z をワールド +Y に回転、`bottom` がワールド -Z に回転。3つすべてベクトル長を保持（正規直交）。
- `uniform layout drift detector` が WGSL `struct Frame` 宣言をパースしフィールドセット + 順序（`diamondRot` / `diamondRotPrev` / diamond 32B params ブロックを含む）をピン留めし、編集者に `src/webgpu/uniforms.ts` も更新するよう促す。
- スペクトル サンプリング フィールド — temporal jitter on でランダムジッタ + hero wavelength を書く、off で WGSL が使う負ジッタセンチネルと固定 540 nm hero wavelength を書く。

WGSL 版は対応する TS モジュールで手動ミラー。TS テストが参照として動作。シェーダー正しさはそれを越えて視覚的に検証 — 自動 GPU テストは無い。

## パフォーマンス

Apple Silicon (Metal 3) で WebGPU `timestamp-query` で計測（`?perf=1` URL フラグが `window._perf.samples` を公開）。下記の数値は 1292×1073 で画面上 4 インスタンス（diamond プリセットは意図的に 1）、≥ 30 サンプルの p50:

| 設定 | GPU 時間 |
|---|---:|
| pill N=8  | 1.70 ms |
| pill N=32 | 6.42 ms |
| cube N=8  | 1.05 ms |
| cube N=16 | 1.38 ms |
| cube N=32 | 1.97 ms |
| cube N=64 | 3.21 ms |

Cube は同じ `N` で pill より目に見えて安価 — バックフェイス エグジットが `cubeAnalyticExit`（単一解析スラブ交差 + rounded-box SDF への 2 ニュートン精緻化ステップ）で、pill/prism パスがまだ波長毎に走らせる per-λ スフィアトレース `insideTrace` + finite-diff 法線とは違う。以前のスフィアトレース cube では N=8 で ~4 ms、N=32 で ~9 ms かかっていたので、解析パスは同じサンプル数で ~4-5× 高速。Plate も同じトリックに従い（`plateAnalyticExit`、波打つ Z 面で ~3 ニュートン反復）、表面が曲がっているにも関わらず pill/prism 帯ではなく cube/plate 帯に着地。

すべての設定で 60 fps をドロップフレーム無しで維持。TBDR ハードウェア（Apple M シリーズ）では背景ピクセルが既に効率的にカリングされている。proxy パスは主にシェイプシルエット外でゼロ重いフラグメントを emit することで助ける。ディスクリート（非 TBDR）GPU はより大きな相対勝利を見る。

### コスト内訳

Pill / prism ピクセル (N=8):
- 最大 64 sphereTrace SDF 評価（前面トレース）+ 8 × 最大 48 inside-trace 評価 + 6 法線評価 + 8 × 6 バック法線評価 — 波長毎バックトレースが支配的。
- 8 写真テクスチャタップ + 1 反射タップ + 1 ヒストリタップ。
- 8 Cauchy + Wyman CIE 評価 + 8 波長毎 Fresnel ミックス。
- Apple Silicon では通常テクスチャバンドウィズ束縛。

Cube ピクセル (N=8):
- 上記と同じ 64 sphereTrace + 6 前面法線評価、しかし波長毎バックトレースは `cubeAnalyticExit` — O(1) スラブ交差 + rounded-box SDF への 2 ニュートン精緻化ステップで丸まったリムにエグジットをスナップ。Pill/prism パスの 8 × (48 + 6) inside-trace + バック法線評価が波長毎 ~40 ALU op に縮約。
- テクスチャタップ + スペクトル数学は pill/prism と同じ。

Plate ピクセル (N=8):
- 前面スフィアトレースは pill/prism より同じサイズで ~35 % 少ないステップで動作 — `waveLipFactor` ≈ 0.92（以前の保守的 0.6 に対し）が、各ステップが内側に留まったまま真の距離をより多く消費できるようにするから。
- 波長毎バックトレースは `plateAnalyticExit`: スラブピック + 波打つ Z 面に対する 3 ニュートン反復（反復毎に 2 cos + 2 sin）+ 解析グラディエント法線。Cube の解析パスとほぼ同じコスト範囲。
- テクスチャタップ + スペクトル数学は pill/prism と同じ。
