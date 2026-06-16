# 施工仕様書: 本家 CUDA SSD Expert-Streaming の移植 (Part 2)

対象ブランチ: `merge/upstream-best`
目的: DeepSeek-V4-Pro 1.6T (~400GB @ q2) を **この7GPU箱単独**で、experts を Optane(DATA1)+1024GB RAM から streaming して推論する。Flash も維持。
出典: 本家クリーンクローン `/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4-upstream` (HEAD e34a808)。

---

## 1. ゲーティング判定 — plain Blackwell sm_120 で動く（Spark限定ではない）

- ランタイムゲートは `ds4.c:79` の `ds4_backend_supports_ssd_streaming()`:
  ```c
  if (backend == DS4_BACKEND_CUDA) {
  #if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))
      return true;   // 非Apple CUDA ビルドなら何でも true
  ```
- `ds4_cuda.cu` の streaming エンジン(`cuda_stream_expert_cache*`, `cuda_stream_selected_cache*`, `cuda_model_copy_to_device_streamed`, `ds4_gpu_stream_*` 群, 行2757–3340)に `#ifdef DS4_CUDA_SPARK_HBM_CACHE` も sm_121 ガードも**無い**（HEAD で `grep -c DS4_CUDA_SPARK_HBM_CACHE` = 0）。
- Spark の「起動時HBM常駐テンサキャッシュ」(別機能)は `15f42aa`→`bf3ff6d` でビルドゲート化されたが `1704eca` で**ゲート自体が撤去**、今は `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB`(既定96GiB)の実行時バジェット制御。
- **ビルド: `make cuda CUDA_ARCH=sm_120`**（`cuda-spark` は -arch 無し＝NG）。

## 2. 階層モデル — host-RAM層 = OS ページキャッシュ。env一発で常駐化

- ソース = mmap した GGUF (`ds4.c:1970`)。**別の SSD キャッシュファイルは無い。CLI で渡すモデルパスが SSD ソース**。
- GPU 常駐 hot 層 = device LRU `g_stream_expert_cache` (`ds4_cuda.cu:171`)。decode 毎に `cuda_stream_selected_cache_begin_compact_load` (2875) が probe → hit=D2D copy / miss=LRU evict+model load。
- cold load `cuda_model_copy_to_device_streamed` (2152): `g_model_fd>=0` なら O_DIRECT pread→pinned staging→async H2D、各chunk後に **FADV_DONTNEED + MADV_DONTNEED でページを捨てる**(2231–2232)。これが既定（小箱前提）。
- **`DS4_CUDA_KEEP_MODEL_PAGES=1`** で両 DONTNEED をスキップ(`ds4_cuda.cu:988,969`)→ 読んだ expert は 1024GB ページキャッシュに残留。初回フルパス後は Pro 全 experts が RAM 常駐、以後 LRU miss は RAM→HBM コピー、Optane は cold/圧迫時のみ。**専用 host-RAM プールの新規実装は不要**（将来 pinned 保証が欲しければ `g_stream_expert_host_cache` を足すのは小工事）。

## 3. Pro 1.6T を Optane+1024GB で回す設定

モデルを Optane に: `/run/media/tonoken3/DATA1/DeepSeek-V4-Pro-q2.gguf`（モデルパス＝SSDソース、別フラグ無し）。

CLI（`ds4_cli.c:1479`, `ds4_server.c:11588`, `ds4_bench.c:248` で同一パース、help `ds4_help.c:165`）:
- `--ssd-streaming` → `ds4_gpu_set_ssd_streaming(true)` (`ds4.c:25730`)
- `--ssd-streaming-cache-experts N|NGB` → GPU-LRU バジェット (`ds4_ssd.c:46` → `ds4_cuda.cu:2769`)
- `--ssd-streaming-preload-experts N` / `--ssd-streaming-cold`

Env（`ds4_cuda.cu`）:
- **`DS4_CUDA_KEEP_MODEL_PAGES=1`** … 1024GB を活かす要。必須。
- **`DS4_CUDA_NO_DIRECT_IO=1`** … 定常運用は page cache 経由(O_DIRECT はキャッシュをバイパスするので常駐戦略と相反)。cold start の Optane 直読みは O_DIRECT が速い→使い分け。
- `DS4_CUDA_STREAMING_EXPERT_CACHE_N=<experts>` … per-GPU 常駐 LRU 数（上限 61*384=23424、既定 512）
- `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB`（既定16）… per-GPU VRAM 余裕
- `DS4_CUDA_ENABLE_STREAMING_EXPERT_HOTLIST=1` … `ds4_streaming_hotlist.inc` の Pro `{layer,expert}` 人気表で LRU を起動シード
- `DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE=1` … hit/miss/cap ログで _CACHE_N を調整

推奨初期設定（per-GPU ~16GB、KV/scratch に 3GB 確保、~12GB を expert cache）:
```bash
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_ENABLE_STREAMING_EXPERT_HOTLIST=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=3
export DS4_CUDA_STREAMING_EXPERT_CACHE_N=<~12GiB / per_expert_bytes>
export DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE=1
# ベンチ前に warm: cat model.gguf > /dev/null （400GB を page cache 常駐させる）
./ds4-server --ssd-streaming /run/media/tonoken3/DATA1/DeepSeek-V4-Pro-q2.gguf  [TP/PP flags]
```

## 4. 移植手順（merge/upstream-best へ、順序付き）

ウチには下層 I/O 配管(`g_model_direct_fd`, `cuda_model_stage_read`, `cuda_model_drop_file_pages`, `cuda_pread_full`, `ds4_gpu_set_model_fd`)は merge-base から有る。**欠けているのは expert-cache 層全部**。
方針: 本家 `g_stream_expert_cache`(GPU-LRU)を裏倉庫にし、ウチの `routed_moe_launch` 内の手製 compact gather を、本家 `cuda_stream_selected_cache_begin_compact_load`(dedup+LRU probe+SSD fallback+D2D)に**streamingモード時だけ**差し替える。非streaming時は既存 `g_moe_compact_*` を維持。

1. **streaming statics+helpers を `ds4_cuda.cu` へ移植**: 構造体 `cuda_stream_expert_cache_slot`/`cuda_stream_expert_cache`(156–185), `cuda_stream_selected_cache`(132–154); globals `g_stream_expert_cache`/`g_stream_selected_cache`/`g_ssd_streaming_mode`(94,194–200); enum `DS4_CUDA_STREAM_EXPERT_DEFAULT(=512)/MAX(=23424)`(38–39); 1412–2240 帯の helper 群と **`cuda_model_copy_to_device_streamed`(2152)**（唯一欠けてる I/O helper、ウチの stage_read 上に乗る）。
2. **public API 移植**(extern "C", 2757–3340): `ds4_gpu_set_ssd_streaming`, `_set_streaming_expert_cache_budget`, `_expert_bytes`, `_configured/current_count`, `_reset_route_hotness`, `_release_resident`, `_budget_for_expert_size`, `_seed_selected`, `_begin_selected_load`, **`_prepare_selected_batch`(decode hook)**, `_seed_experts`, `cuda_stream_selected_cache_begin_compact_load`(2875)。`ds4_gpu_set_model_fd_for_map`(2668)はウチの既存版(~2730)と調停。
3. **`ds4_gpu.h` に宣言追加**: `ds4_gpu_stream_expert_table` 構造体(80–90)＋prototypes(58–125)。`_prepare_selected_batch` は既存ガード `#if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))` で囲う。
4. **`routed_moe_launch`(ウチ `ds4_cuda.cu:12526`)の調停**: compact 分岐先頭(~12640)に streamingモード分岐を追加 — `ds4_gpu_stream_expert_table` を {model_map,size,layer,n_total_expert,gate/up/down_offset,gate/down_expert_bytes} から構築、既存 host_selected の D2H(12695–12703)を読み、`ds4_gpu_stream_expert_cache_prepare_selected_batch(&table, host_selected, n_tokens, n_expert)` 呼び、`gate_w/up_w/down_w = g_stream_selected_cache.{gate,up,down}_ptr`、remap=`slot_selected_tensor`。**既存 `g_moe_compact_*` ブロックは else(非streaming)として残す**。本家 remap は dedup 済(同 expert 共有スロット)で優秀→採用。
5. **`ds4.c` のエンジン設定配線**(ウチの `ds4_engine_*` 初期化相当): `ds4_gpu_set_ssd_streaming(e->ssd_streaming)`, slab-size pin+boosted層カウント(25736–25768), `ds4_gpu_set_streaming_expert_cache_budget`, `--ssd-streaming` の span 制限 `ds4_gpu_set_model_map_spans`(25775–25839)。engine config 構造体に `ssd_streaming*` フィールド＋CLI/server/bench パース追加。`ds4_backend_supports_ssd_streaming`(`ds4.c:79`)が無ければ追加。
6. **2ファイル verbatim コピー**: `ds4_streaming_hotlist.inc`(Pro `{layer,expert}` 表, `ds4.c:829` で include)、`ds4_ssd.c`/`ds4_ssd.h`(plain C helper、Makefile に `ds4_ssd.o` 有るか確認)。

## 5. 最大リスク（要設計判断）

1. **単一GPUグローバル vs ウチの7GPU PP/TP（最重要）**: `g_stream_expert_cache`/`g_stream_selected_cache`/`g_ssd_streaming_mode` は device-index されてない。ウチの `g_moe_compact_*[DS4_CUDA_MAX_DEVICES]` 同様に **per-device 配列化**しないと PP ステージ間で thrash/破損。seed/budget も全 device 反復。本家は multi-GPU streaming を走らせたことが無い。
2. **CUDA Graph capture との両立**: ウチの `routed_moe_launch` には capture-active 高速路(12680–12690)が有り、host D2H 読みや host LRU 簿記が**できない**。`begin_compact_load` は host LRU + selected id の D2H をする→**capture 中は不可**。gather を capture の外で(replay 前に compact buffer を pre-seed)走らせるか、streamed 層は graph を切る。明示的に解く設計シーム。
3. **O_DIRECT vs page-cache**: fd 設定時の既定 O_DIRECT は常駐させたい page cache をバイパス。`DS4_CUDA_NO_DIRECT_IO=1` と `DS4_CUDA_KEEP_MODEL_PAGES=1` は**セットで**。
4. **boosted/mixed精度層**: slab allocator が off-size 層をキャッシュ迂回。均一 q2 Pro なら no-op だが、boosted なら該当層が毎step mmap copy に落ちる(遅)。起動ログ(`ds4.c:25754`)で boosted/routed 分割を監視。
5. **ウチの古い `cuda_model_stage_read`/fd 配管が本家HEADと乖離の可能性**: 流用前に diff、必要なら本家の quad-buffer event モデル(1014–1104)に rebase。

難易度: **中〜高**。streaming ロジック自体は sm_120 で無改造コンパイル＆調停アンカー(`begin_compact_load`≒ウチの compact gather)がほぼ一致するので書き直しではなく ~1500行 graft。難所は streaming ではなく multi-GPU 化と graph capture シーム。
