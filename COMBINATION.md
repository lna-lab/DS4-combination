# DS4-combination — best-of-both 統合プロジェクト

2つの DS4 系譜を、Mac + CUDA のヘテロ構成クラスタで最大活用するために統合する作業リポジトリ。
2人の Claude Code（CUDAノード側・Macノード側）が、この repo を共有拠点として協働する。

## 系譜と判定
- **ベース = `Tonoken3/DS4-For-SM120`**（ウチのフォーク）: 1台内マルチGPU CUDA = Pipeline Parallel(7GPU)+Tensor Parallel+CUDA Graph+compact-MoE。これが crown jewel で、統合の土台。
- **上流 = `antirez/ds4`**（DwarfStar）: 共通祖先 `ec6a82a`、上流は +90 commits 進化。良い部分だけ選択的に取り込む（フル merge はしない）。
- 検証済み判定:
  - 上流 **Q4 MoE 高速化 = 取り込まない**（Q4_K専用、ウチは IQ2/Q2 でデッドコード。ウチが既に IQ2 で同等以上）。
  - 上流 **distributed = layer-pipeline 専用**（複数マシンで層分担、トークン毎にネット越え＝decode は遅い）。「単機に乗らないモデルを両ノード合算で動かす」用途では有効 → Part 3。
  - DSV4 KV snapshot は **v1/v2 バイト一致**（v2 は distributed 保存パスを足しただけ）→ クロスバックエンド handoff が安い。

## ハードウェア
- **CUDAノード** (Linux): 7× RTX PRO 2000 Blackwell (sm_120, ~16GB×7≈112GB HBM), **CPU 1024GB RAM**, **Optane SSD = /run/media/tonoken3/DATA1**, 10GbE。
- **Macノード**: Mac Studio M2 Ultra 192GB (Metal), 10GbE。

## 3つの能力（toolkit）
| # | 能力 | 何が嬉しいか | 状態 |
|---|------|------------|------|
| 1 | **KV snapshot 橋** | prefill@7GPU → snapshot → Mac local decode。そこそこのモデルを高速に（per-token ネット遅延を回避＝一回転送のみ） | 進行中 |
| 2 | **CUDA SSD streaming** | Pro 1.6T(~400GB) を GPU箱単独で（experts を Optane+1024GB RAM から）。`SPEC_SSD_STREAMING_CUDA.md` 参照 | 計画完了、未着手 |
| 3 | **distributed layer-pipeline** | 単機に乗らない巨大モデルを Mac+CUDA 合算で | 未着手 |

## 役割分担（2人の Claude）
- **CUDAノード側 Claude（このリポを書いている私）**: CUDA バックエンド(`ds4_cuda.cu`)＋バックエンド中立な共有ファイル(`ds4.c`/`ds4.h`/`ds4_cli.c`)の編集、CUDA ビルド＆7GPU スモーク、push。
- **Macノード側 Claude**: Metal ビルド＆検証。当面の任務 = この枝をビルドし、**CUDAノードが吐いた DSV4 v2 snapshot を Metal で load → decode 継続できることを確認**（橋の Mac 側）。`ds4_metal.m` 等 Metal 固有のみ編集し、共有ファイルは CUDA 側と衝突しないよう調整。
- **共有契約 = DSV4 snapshot 形式 ＋ 橋プロトコル**（`SPEC_KV_HANDOFF.md` に明文化予定）。各ノードのバイナリは別ビルドでよい。

## Mac コードベース方針（決定）
このリポ（=ウチのフォークベース）から Mac も Metal ターゲットをビルドする。ウチのフォークの `ds4_metal.m` は上流より古く **SSD streaming 非搭載**だが、**192GBに収まるモデル(Flash/中型)の decode には十分**で、橋の Mac 側はこれで成立する。Pro 1.6T は GPU箱(Part 2)で回すので Mac で Pro を回す必要は当面ない。将来 Pro-on-Mac が欲しければ、Mac-Claude が上流の Metal SSD streaming をこの repo にマージする（その時の課題）。

## 現状 / 次の一手
- [x] Part 0: 分岐解析・判定（上記）。
- [x] Part 1-a: DSV4 payload **v1↔v2 整合**（emit=2, load は {1,2} 受理）。`ds4.c`。CUDA ビルドで検証済。
- [ ] Part 1-b: snapshot export/import CLI（`ds4_session_save_snapshot`/`load_snapshot` を露出）。
- [ ] Part 1-c: 10GbE 転送グルー。
- [ ] Part 1-d: CUDA ビルド＋ prefill→save→転送→load→decode スモーク。
- [ ] Part 2: `SPEC_SSD_STREAMING_CUDA.md` に沿って CUDA SSD streaming 移植。
- [ ] Part 3: distributed layer-pipeline。

作業枝: `merge/upstream-best`。
