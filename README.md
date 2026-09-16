# nanochat-ja

[karpathy/nanochat](https://github.com/karpathy/nanochat) をベースに、**トークナイザ学習 → 事前学習
→ SFT → チャット**まで、単一の **RTX 3090 (24GB)** だけでフルスクラッチで作った**日本語ネイティブ**LLM。

> このリポジトリは [karpathy/nanochat](https://github.com/karpathy/nanochat)(MITライセンス、
> [`LICENSE`](LICENSE) 参照)のForkです。学習・推論の中核フレームワーク(トークナイザ、GPTモデル、
> オプティマイザ、データローダ、推論エンジン、SFTループ)は本家のものをそのまま使用しています。
> このForkで追加したのは、日本語データパイプライン、フルスクラッチの日本語トークナイザ、日本語SFT
> データセット、日本語ベンチマークでの評価、簡易チャットフロントエンドです。詳細は以下に記載します。

## 結果

| ステージ | 指標 | 結果 |
|---|---|---|
| トークナイザ(vocab 65,536、CC-100 jaの10億文字で学習) | 日本語テキストへの圧縮率 | 7.64 bytes/token(GPT-2: 2.06、GPT-4: 2.85 と比べて**63〜73%効率的**) |
| 事前学習(depth=12、約4.88億パラメータ) | Validation bits-per-byte | 0.818 →(学習量を token:param 比 12→26 に拡張)**0.801** |
| ベースモデル | [JCommonsenseQA](https://github.com/yahoo-japan/JGLUE)(2-shot) | ランダムベースライン20.0%に対して**24.1%**(270/1119) |
| SFT(`llm-jp/magpie-sft-v1.0`、ネイティブ日本語) | Validation bits-per-byte | 1.205 → **0.451** |

SFT後の会話例:

```
User: 日本の首都はどこですか?
Assistant: 日本の首都は東京です。東京は2010年に天皇陛下が即位したことを契機に「首都令」という
新たな法律により東京都が定められました。...

User: おすすめの日本食を教えてください。
Assistant: もちろんです、以下にいくつかのおすすめの日本食を紹介します：
1. 寿司（すし）: 新鮮な魚介類を使用し、酢飯と新鮮な魚介類で巻いたものです。...
2. うどん: うどんは小麦粉と水で作った小麦粉の麺で、その独特の食感と軽快な味わいが特徴です。...
```

(コンシューマ向けGPU1枚・約17 GPU時間で学習した約4.88億パラメータのモデルなので、日付のハルシネー
ションや算数の失敗は依然として発生します。うまくいった点・いかなかった点は
[Design notes](#design-notes--学んだこと) に正直にまとめています。)

## Quickstart

このパイプラインのどこにもAPIキーや認証情報は必要ありません — 以下のダウンロードはすべて
Hugging Faceへの匿名の公開リクエストです。

```bash
uv sync --extra gpu   # CPU/MPSの場合は --extra cpu (かなり遅くなります)
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"  # データ・チェックポイントのキャッシュ先
```

### 1. 事前学習データ(日本語CC-100)

```bash
python -m dev.prepare_cc100_ja --num-source-files 8   # 約16GBのダウンロード、約96億文字
```

[`range3/cc100-ja`](https://huggingface.co/datasets/range3/cc100-ja) からshardをダウンロードし、
`nanochat/dataset.py` が期待するparquet shard形式に再構成します。

### 2. トークナイザ

```bash
python -m scripts.tok_train --vocab-size 65536 --max-chars 1000000000
python -m scripts.tok_eval
```

日本語コーパスのみを使ってフルスクラッチで学習します。30億文字で実行した際にBPEのペアカウント処理中
にOOMしたため、10億文字に落として学習しました(詳細は [Design notes](#design-notes--学んだこと))。

### 3. 事前学習(ベースモデル)

```bash
python -m scripts.base_train --depth=12 --device-batch-size=16
```

`--depth=12` は24GBのGPUに収まるよう選んだ約4.88億パラメータのモデルサイズです。nanochatはこの値
から他のハイパーパラメータ(幅、バッチサイズ、学習率、weight decay、学習量)をすべて自動で導出しま
す。`--device-batch-size` はより小さいGPUではさらに下げる必要があるかもしれません(本家READMEの
OOM対処法がそのまま使えます)。ダウンロード済みコーパスをさらに使って学習を延長する場合:

```bash
python -m scripts.base_train --depth=12 --device-batch-size=16 \
  --resume-from-step=<最後のstep> --target-param-data-ratio=26
```

### 4. SFT(対話ファインチューニング)

```bash
python -m scripts.chat_sft --total-batch-size=32768 --magpie-epochs=1
```

[`llm-jp/magpie-sft-v1.0`](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)(翻訳ではなく
ネイティブ日本語)を使用します。事前学習のデフォルトバッチサイズ(524,288トークン)は数十億トークン
規模の事前学習向けのサイズで、約13万行のSFTデータセットには大きすぎるため、`--total-batch-size` を
明示的に指定しています(詳細は [Design notes](#design-notes--学んだこと))。

### 5. 会話してみる

```bash
python -m scripts.chat_cli                 # 対話型CLI(本家のスクリプト)
python -m dev.chat_relay "こんにちは"        # 単発実行、会話状態はディスクに保存して継続
python -m dev.webapp                        # ローカルWeb UI(http://127.0.0.1:8000)
```

### 評価

```bash
python -m dev.eval_jcommonsenseqa           # JCommonsenseQA(日本語常識推論)、データは自動ダウンロード
python -m scripts.base_eval                 # 本家のCOREメトリクス(英語ベンチマークのため日本語限定
                                             # モデルの評価としては参考程度。一応動作確認用に記載)
```

## Design notes / 学んだこと

**なぜdepth=12なのか(本家のGPT-2級ターゲットであるd20/d26ではなく)?** 本家のspeedrunは
8×H100ノードを前提としています。RTX3090 1枚では、計算量予算(3090のピークBF16 FLOPs × 使える
時間 ÷ 各depthでのFLOPs/token)から、数時間規模の学習にはdepth 9〜12あたりがcompute-optimalという
試算になりました。depth=12を選んだのは、これがnanochat内部の基準depth(muP的なスケーリング式が
d12を基準にキャリブレーションされている)でもあり、十分に検証されたハイパーパラメータ帯に収まる
ためです。

**なぜGPT-2/GPT-4のトークナイザを流用せず、ゼロから学習したのか?** それらは英語向けに最適化されて
おり、日本語コーパスのみでフルスクラッチ学習したBPEトークナイザの方が日本語テキストをはるかに効率
よく圧縮できます(上の結果表を参照)。vocabサイズを65,536(本家デフォルトの32,768ではなく)にした
のは、日本語の文字種の多さがより大きな語彙から恩恵を受けるためで、かつnanochatのvocabパディング
処理に都合の良い2の冪でもあります。トレードオフとして `value_embeds` と `lm_head` はvocabサイズに
比例するため、デフォルト比でパラメータ数がおよそ倍になりました — 圧縮率向上の価値はありましたが、
その分VRAMの余裕は小さくなり、24GBのカードで `--device-batch-size` を32から16に下げる必要があり
ました。

**なぜ事前学習にCC-100 jaを使ったのか?** 実は「Japanese SlimPajama」という名前の独立した日本語
コーパスは存在しません(SlimPajama自体は英語のみのコーパスで、他プロジェクトのデータミックスの中で
別の日本語コーパスと並ぶ一成分として時々使われるだけです)。代わりに選んだのが
[`range3/cc100-ja`](https://huggingface.co/datasets/range3/cc100-ja) — [CC-100](https://data.statmt.org/cc-100/)
の日本語部分をparquetで再shardしたものです。規模が大きく、nanochatのローダーがほぼそのまま受け付
ける形式であり、日本語LLMプロジェクトでの採用実績もあります。

**事前学習を安全に延長する。** `scripts/base_train.py` の `--resume-from-step` は、同じ学習
スケジュールの*中断された*実行を再開するために設計されています — このときチェックポイントに保存
された学習率(LR)状態を復元するのは正しい挙動です。しかし、*より大きな* `--target-param-data-ratio`
を指定してさらに学習を延長する場合は事情が異なります: チェックポイントのLRは旧スケジュールの
(すでに減衰しきった)値を反映しており、新しいスケジュール用のウォームアップし直した値ではありま
せん。そのままだと、延長学習の間ずっとほぼゼロの学習率で(気づかないまま)学習してしまうことに
なります。このForkでは `optimizer.load_state_dict()` の前後で新しく計算したLRをスナップショット・
復元することでこれを修正しています(モーメンタムバッファ等は通常通りチェックポイントから復元され
ます)。詳細は `scripts/base_train.py` の差分を参照してください。

**SFTデータセット: 3回の試行錯誤。**
1. 最初は[`kunishou/databricks-dolly-15k-ja`](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)
   (英語Dollyの機械翻訳、CC-BY-SA-3.0)を、事前学習のバッチサイズ(524,288トークン)をそのまま
   引き継いで使用 — 14,000行のデータセット全体がわずか約5ステップで消費されてしまい、実質的に
   ほとんど学習が進みませんでした。
2. バッチサイズを32,768に修正し、同じ14,000行を4エポック学習 — 訓練lossは大きく下がったものの、
   validation bpbは*悪化*しました(1.04 → 1.26)。小規模・反復・翻訳データに対する典型的な過学習
   です。1エポックに減らすことで過学習は解消しました(bpb 1.04 → 0.79)が、生成品質はまだ不安定
   で(例: ある実行では「日本の首都」の質問に誤答)。
3. [`llm-jp/magpie-sft-v1.0`](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)
   (Apache-2.0)に完全に切り替え: 件数は約9倍で、翻訳ではなく[Magpie](https://arxiv.org/abs/2406.08464)
   手法による*ネイティブ生成*の日本語データ(質問・回答とも日本語対応の強力なモデルが生成)です。
   1エポック(958ステップ、約9.5分)でvalidation bpbは1.20から0.45まで改善し、反復ループでの
   崩壊もなく、一貫して正確で話題に沿った回答が得られました。ステップ1で使っていた未使用の
   `tasks/japanese_dolly.py` は削除し、現在は `tasks/magpie_ja.py` を使用しています。

**Ampere固有の知見。** Flash Attention 3(`kernels-community/flash-attn3` パッケージ経由)はHopper
だけでなく、このAmpere(SM 86)のカードでも動作しました — nanochatのsliding window attentionパター
ンではSDPAへのフォールバックがスループットに大きく影響するため、どのGPUでも一度確認する価値があり
ます。観測したスループットは定常状態で約58,000 tok/sec、MFU約74〜75%でした。

## リポジトリ構成

`nanochat/`、`scripts/`(下記2ファイルを除く)、`runs/`、`tests/` 以下はすべて本家のコードをその
まま使用しています。このForkで追加した部分:

```
dev/
├── prepare_cc100_ja.py     # range3/cc100-jaのダウンロード+nanochatのshard形式への再構成
├── eval_jcommonsenseqa.py  # 日本語常識推論ベンチマーク(データは自動ダウンロード)
├── gen_test.py             # 複数プロンプトでの簡易生成テスト(ベースモデル)
├── gen_test_sft.py         # 同上、チャット/SFTモデル向け
├── chat_relay.py           # 会話状態を保存しながら1回ずつ実行するCLIチャット
└── webapp.py               # 最小限のローカルFlaskチャットUI(127.0.0.1のみでリッスン)
tasks/
└── magpie_ja.py            # llm-jp/magpie-sft-v1.0用のTaskラッパー
scripts/
├── base_train.py           # 追加: --resume-from-stepで学習量を変更した際にLRを正しく
│                            #   再ウォームアップする修正(Design notes参照)
└── chat_sft.py             # 追加: デフォルトの英語SFTミックス(SmolTalk/MMLU/GSM8K)を
                             #   tasks.magpie_ja.MagpieJa に置き換え
```

## 使用データセット(本リポジトリには含まれません)

コーパス・チェックポイント・生成サンプルはいずれもこのリポジトリにはコミットされていません — 上記
のスクリプトはすべて初回実行時に自分でデータをダウンロードし、`$NANOCHAT_BASE_DIR` にキャッシュし
ます。各データセットはそれぞれの元のライセンスに従うため、個人的な実験を超えた利用の前には各リンク
先を確認してください。

| データセット | 用途 | ライセンス |
|---|---|---|
| [`range3/cc100-ja`](https://huggingface.co/datasets/range3/cc100-ja)([CC-100](https://data.statmt.org/cc-100/)由来) | 事前学習 | 作成者による追加の制限なし。[Common Crawl利用規約](https://commoncrawl.org/terms-of-use)に準拠 |
| [`llm-jp/magpie-sft-v1.0`](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0) | SFT | Apache-2.0 |
| [`leemeng/jcommonsenseqa-v1.1`](https://huggingface.co/datasets/leemeng/jcommonsenseqa-v1.1)([JGLUE](https://github.com/yahoo-japan/JGLUE)由来) | 評価のみ | CC-BY-4.0 |
| [`kunishou/databricks-dolly-15k-ja`](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja) | 初期のSFT実験(現在は不使用、Design notes参照) | CC-BY-SA-3.0 |

## 本家 nanochat について

このセクションの残りは本家から引き継いだ内容で、基盤となるフレームワークについては現在もそのまま
当てはまります。

nanochatはAndrej Karpathy氏によるミニマルな単一GPUノード向けLLM学習ハーネスです — トークン化、
事前学習、ファインチューニング、評価、推論をすべて1つのハック可能なコードベースに収め、単一の
複雑さダイヤル(`--depth`)を中心に構成されています。本家README全文、GPT-2 speedrunリーダーボード、
ガイド、ディスカッションについては[本家リポジトリ](https://github.com/karpathy/nanochat)を参照
してください。

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT(本家から継承、[`LICENSE`](LICENSE) を参照)。学習データ・データセットのライセンスは上記の
通りこれとは別であり、本リポジトリ自体のライセンスの影響を受けません。
