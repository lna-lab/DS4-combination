# 施工仕様書: KV snapshot 橋（GPU-prefill → remote-decode handoff）— Part 1

両ノード（CUDA / Metal）が守る**共有契約**。これさえ一致すれば、各ノードのバイナリは独立にビルドして良い。

## 1. 契約の核 — DSV4 v2 snapshot 形式

- `ds4_session_save_payload(s, FILE*, ...)` が書く「DSV4 payload」= magic `0x34565344` + version + 13×u32 ヘッダ + checkpoint tokens + **次トークンの logits(vocab×f32)** + per-layer KV。
- **次トークン logits を含む**ので、ロード側は再デコード1手すら踏まず保存時点の分布から続けられる。
- **F32 ディスク形式＝バックエンド可搬**（CUDA は F32 書き／Metal は読んで F16 化）。FP8-KV も F32 段階で量子化されるのでバイト差は出ない。
- **v1↔v2 はバイト完全一致**。CUDAノードは emit=**2**（上流 Metal と互換）、load は **{1,2} 受理**（`ds4.c` の `DS4_SESSION_PAYLOAD_VERSION` / `_LEGACY`）。

## 2. CLI 表面（`ds4_cli.c`、両ノード共通＝同一ソース）

実装済み（`run_kv_bridge`、バックエンド中立 API のみ使用＝Metal 安全）:

| フラグ | 動作 |
|---|---|
| `--save-kv FILE` | prefill（または `--load-kv` 復元）後に DSV4 payload を FILE へ書く |
| `--load-kv FILE` | FILE から DSV4 payload を復元し、保存時の次トークン分布から decode 継続 |
| `--prefill-only` | prefill（＋任意の `--save-kv`）で停止、生成しない |

例:
```bash
# 送り手（prefill ノード）: 長文を prefill して KV を保存、生成せず終了
./ds4 -m <MODEL> -p "<long prompt>" --save-kv snap.dsv4 --prefill-only
# 受け手（decode ノード）: KV を復元して 200 トークン生成
./ds4 -m <MODEL> --load-kv snap.dsv4 -n 200
```

## 3. handoff フロー

```
CUDAノード(7GPU, TP/PP prefill)            10GbE              Macノード(M2 Ultra, Metal decode)
  ./ds4 -p ... --save-kv snap --prefill-only  ──(scp/rsync)──►  ./ds4 --load-kv snap -n N
  → DSV4 v2 payload を書く                  一回転送のみ        → 復元 → ローカル decode（per-token ネット遅延ゼロ）
```

## 4. 絶対条件 — layout parity（両ノードで完全に同じモデル）

`ds4_session_load_payload` はヘッダで層数・head_dim 等を照合し、食い違えば "does not match current runtime" で弾く。
→ **両ノードが寸分違わぬ同一 GGUF をロードすること**。現在の対象:
`DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf`
（`antirez/deepseek-v4-gguf`、97.59GB、`./download_model.sh q2-q4-imatrix`）。CUDAノードは DATA1(Optane) に取得中。

## 5. 転送

v1 = ファイルを scp/rsync で 10GbE 越し（一回転送、per-token ではない）。将来 = TCP ソケット送受信を `ds4` に内蔵（Part 1-c）。

## 6. 要検証ポイント（スモーク）

- **`--load-kv` 後の最初の `ds4_session_sample` が、保存済み次トークン分布から正しく出るか**（README は「再デコード不要で継続可」と明記。設計上はそのはずだが実測で確認）。
- 同一プロンプトで「通常 decode」と「prefill→save→load→decode」の出力が一致するか（決定論: `--temp 0`）。
- クロスバックエンド: CUDA で保存した snap を Metal で load → 継続が正しいか。

## 7. ノード別 TODO

- **CUDAノード（実装済/進行中）**: 橋 CLI 実装済・ビルド green。モデル DL 完了後に prefill→save スモーク＋snap を Mac へ転送。Part 1-c（TCP転送内蔵）。
- **Macノード（Part 1-d）**: (a) この枝を Metal ビルド、(b) **Mac 単機で往復自己テスト**（`--save-kv /tmp/s --prefill-only` → `--load-kv /tmp/s -n 50`、Metal で save/load が回るか）、(c) CUDA 産 snap を load → decode。
