# Open Agent Memory Benchmark

[English](README.md) | [简体中文](README_CN.md) | 日本語

**各システムのネイティブ API を通じて、セルフホスト型 Agent Memory システムを比較する、再現可能でエビデンス重視のベンチマークです。**

## v0.1.0 の結果

**均衡 LME-60 スクリーニングでは、Hindsight が 57/60（95.0%）で最も高い回答精度を記録しました。一方、Mem0 は回答可視コンテキストが最も少なく、392.6k トークンで、Hindsight より 59%、OpenViking より 58% 少ない結果でした。**

| プロバイダー | 回答精度 | 回答可視コンテキストトークン |
|---|---:|---:|
| Hindsight | **57/60 (95.0%)** | 合計 954.4k / 平均 15.91k |
| Mem0 | 52/60 (86.7%) | **合計 392.6k / 平均 6.54k** |
| OpenViking | 51/60 (85.0%) | 合計 927.6k / 平均 15.46k |

**[v0.1.0 の完全なレポートを開く](https://rocke2020.github.io/open-agent-memory-benchmark/eval_results/v0.1.0/comparison-20260913-133717-82378/report.html)** · [公開評価スナップショット全体を見る](eval_results/v0.1.0/)

**カバレッジ：** 3 つのプロバイダーで共通する 60 問の LongMemEval を評価し、180/180 件の判定済みプロバイダー別結果を生成しました。

この 60 問のスクリーニングでは、Hindsight が最も高い精度を記録しました。Mem0 の際立った結果はコンテキスト使用量の少なさで、392.6k の回答可視コンテキストトークンにより 52/60（86.7%）を達成し、Hindsight より 59% 少ない使用量でした。OAMB はこのサンプルから決定的な勝者を宣言しません。対応のある統計分析については完全なレポートを参照してください。回答可視コンテキストトークンは、回答モデルに実際に提示された検索エビデンスを測るものであり、プロバイダー内部のトークン使用量ではありません。これは均衡の取れた 60 問のスクリーニング比較であり、500 問すべてを用いた LongMemEval の完全再現でも、普遍的なプロバイダーランキングでもありません。

### 実利用上の要点

この実行で OpenViking の最も明確な弱点となったのは速度です。OAMB は、生成処理を使わずネイティブリランキングを無効にした `/api/v1/search/find` という直接的な OpenViking の検索経路を使用していますが、検索レイテンシの中央値は 0.79 秒（p95 は 1.30 秒）で、Hindsight の 0.17 秒、Mem0 の 0.12 秒より長い結果でした。質問ごとの履歴について、取り込み開始からインデックス作成完了までの中央値は 1,190 秒（19.8 分）で、Hindsight は 761 秒、Mem0 は 842 秒でした。これらは本ワークロードで観測された時間であり、環境に依存しない製品ベンチマークではありません。それでも、検索レイテンシと新しいメモリが利用可能になるまでの待ち時間が重要なリアルタイム Agent にとって、明確な実用上の懸念です。

Hindsight は、ここで対象とした基本的なメモリタスクで優れた結果を示しています。v0.1.0 はユーザープロファイルの品質を評価していません。別の実利用ではプロファイリングが Hindsight の比較的弱い領域であることが示唆されているため、これは本リリースの測定範囲外にある実用上の注意点であり、このベンチマークが確立した結果ではありません。

Mem0 Cloud（Platform v3）とセルフホスト型 Mem0 OSS 2.0.19 の間にも、重要な機能差があります。Platform v3 は時間入力と時間を考慮したランキングをネイティブに提供しますが、評価対象の OSS リリースにはこれらの機能がないため、追加の適応がなければ時間に関する質問は実用上の弱点になります。OAMB は履歴日付を与えるカスタム抽出プロンプトで補っているため、temporal-reasoning 9/10 は強化された OAMB と Mem0 OSS のパイプラインを反映したものであり、OSS のネイティブな時間機能を示すものではありません。この境界の詳細は、本 README の後半にある互換性に関する注意で説明しています。

## 概要

Open Agent Memory Benchmark（OAMB）は、1 つの共通評価プロトコルの下で、ネイティブ REST API を通じて Hindsight、Mem0、OpenViking を比較します。v0.1.0 プロファイルでは、各プロバイダーに同じ均衡選択された 60 問の LongMemEval を使用します。6 種類の質問タイプから各 10 問を選び、合計 180 件のプロバイダー別結果を生成します。

OAMB の特長の 1 つは、**回答精度**、**回答可視コンテキストトークン**、**インデックス作成トークン**、**検索レイテンシ**、**インデックス作成時間**という 5 指標のビューです。定義と詳細な結果は、[v0.1.0 の完全なレポート](https://rocke2020.github.io/open-agent-memory-benchmark/eval_results/v0.1.0/comparison-20260913-133717-82378/report.html)を参照してください。

OAMB に先立ち、私は AMB で[全 500 問の Hindsight 再現](https://github.com/rocke2020/agent-memory-benchmark/tree/deepseek-provider)を実施しました。この実験では、公開された Hindsight 0.4.17 の構成を再現し、外部モデルスタックだけを変更しました。抽出と判定には DeepSeek V4 Flash 0731、回答には DeepSeek V4 Pro 0813 を使用しました。この研究により、モデル選択は評価対象そのものを変え、実行ごとの非決定性も結果を変え得ることを再確認しました。LongMemEval には曖昧な問題が少数ありますが、データセット全体としては幅広い比較に十分有用です。コストを抑えるため、OAMB は全 500 問を再実行せず、Hindsight 0.9.2 を含む固定バージョンの各プロバイダーを均衡 LME-60 で比較します。

OAMB は、実行前に質問、モデルの役割、回答/判定ポリシー、比較設定を固定します。各プロバイダーはネイティブのストレージおよびインデックス作成動作を維持します。再試行、失敗、部分的な取り込み、コスト、取得不能な測定値はすべて可視化され、普遍的な総合スコアは作りません。検索ではクエリ埋め込みを許可しますが、生成的なクエリ書き換え、リフレクション、再ランキングは無効です。

## プロバイダーのバージョン

OAMB は現在、次の正式リリースと不変のソースコミットを固定しています。

| プロバイダー | リリース | ソースコミット |
|---|---|---|
| Hindsight | [0.9.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.9.2) | [`ebad478240d3171bb88201ececda5e8d9883d22d`](https://github.com/vectorize-io/hindsight/commit/ebad478240d3171bb88201ececda5e8d9883d22d) |
| Mem0 | [2.0.19](https://github.com/mem0ai/mem0/releases/tag/v2.0.19) | [`dc82354e143c2581d505d581a00286d6ef8c3605`](https://github.com/mem0ai/mem0/commit/dc82354e143c2581d505d581a00286d6ef8c3605) |
| OpenViking | [0.4.19](https://github.com/volcengine/OpenViking/releases/tag/v0.4.19) | [`f3afef11637f2d7c11e4b1f36ed2f90630737cdc`](https://github.com/volcengine/OpenViking/commit/f3afef11637f2d7c11e4b1f36ed2f90630737cdc) |

[`provider-services/versions.env`](provider-services/versions.env) が実行時バージョン固定の信頼できる情報源です。そこに記録されたイメージダイジェストとソースアーカイブのハッシュにより、プロバイダーの準備を再現できます。

## クイックスタート

Git、Python 3.11 以降、`uv`、Compose 対応の Docker、`curl`、`jq`、`shasum`、および OpenAI 互換モデルエンドポイントの認証情報が必要です。準備と評価では課金対象のモデル呼び出しが発生する場合があります。処理は隔離されたプロバイダー状態を作成し、既存のプロバイダー/データベースデータを保持します。

### 1. 設定

リポジトリをクローンし、非公開設定を準備します。

```bash
git clone https://github.com/rocke2020/open-agent-memory-benchmark.git
cd open-agent-memory-benchmark
cp .env.example .env
chmod 600 .env
```

`.env` を編集し、`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_LIGHT_MODEL`、`LLM_DEEP_MODEL` を設定します。`LLM_URL_TYPE=openai_chat` はそのままにしてください。[テンプレート](.env.example)では、両方のモデルプロファイルに唯一の生成モデルとして `deepseek-flash`（DeepSeek-V4.1-Flash）を使用します。light プロファイルはプロバイダーの抽出と判定を担当し、deep プロファイルは回答を担当します。[ベンチマーク設定](configs/benchmark.yml)が役割の割り当て、思考強度、実行制御の信頼できる情報源です。

埋め込みサービスを再利用するには、その OpenAI 互換ベース URL を設定します。既定のプロファイルでは、`qwen3-embedding:0.6b`、1,024 次元ベクトル、およびプロバイダーのソースセッション入力に十分なコンテキスト容量が必要です。既存のホストローカルサービスを使う場合：

```dotenv
OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:18000/v1
OAMB_EMBEDDING_API_KEY=
```

キー不要のサービスでは、埋め込み API キーを省略または空にできます。OAMB はホスト側のプローブを変更せず、この HTTP IPv4 ループバック URL を Docker コンテナ向けに変換します。リモート HTTPS エンドポイントにも対応します。ループバック URL では HTTPS および IPv6 が拒否されます。

URL が未設定、または `change-me` のままの場合、事前チェックはローカルのフォールバックを使用します。macOS では同階層の `vllm-metal` インストールとキャッシュ済みモデルファイルが必要です。Linux では Ollama が必要で、初回利用時に埋め込みモデルをダウンロードする場合があります。[ローカル埋め込みヘルパー](scripts/start_local_embedding)を参照してください。

### 2. 事前チェックと実行

依存関係、データセット、固定バージョンのプロバイダーサービス、モデルの準備状態を整えます。

```bash
./precheck.sh
```

`precheck: PASS` を確認してから続行してください。その後、完全な比較を実行します。

```bash
./run.sh --full_test
```

各プロバイダーは設定された制限内で並行実行されます。各プロバイダーは 30 秒ごとに状態、経過時間、保存済み質問数を表示します。完了した結果はアトミックに保存されます。最終比較には 180 件すべての結果と最新のエビデンス検証が必要です。

その他の実行モード：

```bash
./run.sh --full_test --resume  # 完了済みの質問を再利用し、未完了分を新しい隔離スコープで再実行
./run.sh --smoke_test         # 任意：3 プロバイダーすべてで 1 問の診断を実行
./run.sh --dry-run            # モデルやプロバイダーを呼び出さず、固定されたプランを確認
```

引数なしの `./run.sh` も smoke モードを選択します。Smoke の結果は分離され、完全実行の前提ではありません。

[configs/benchmark.yml](configs/benchmark.yml) の `execution` で並行度を設定します。

| 設定 | 制限対象 |
|---|---|
| `max_parallel_providers_per_dataset` | 並行するプロバイダー評価 |
| `max_parallel_history_ingestions_per_provider` | プロバイダーごとに並行して取り込む独立履歴 |
| `max_parallel_questions_per_provider` | プロバイダーごとの検索、回答、判定パイプライン |

同じ設定ファイルの `retrieval.top_k` で、共通のネイティブ候補上限を設定します。この値は解決済みプランに固定され、すべてのプロバイダーアダプターで使用されます。Mem0 は `top_k`、OpenViking は `limit` として送信します。Hindsight の recall エンドポイントには件数パラメーターがないため、プロバイダーの順位を維持して正規化された候補の先頭 N 件だけを残します。

1 つの履歴内のセッションは逐次処理されます。これらの制限は、プロバイダー内部の埋め込み/モデルリクエストを制限しません。新規実行は事前チェック済みプランを使用します。各 `--resume` は、未完了の処理に対して現在の YAML から履歴と質問の上限を読み取ります。この 2 つの値を変更しても完了済みの結果は無効になりません。その他のプラン設定は固定されたままで、実行中のプロセスは YAML を再読み込みしません。`resolved-plan.json` は編集しないでください。

再試行と試行ごとのタイムアウト設定は、これらの制御とともに設定ファイルに記載されています。実行全体や支出の上限ではありません。ローカルのタイムアウトは、リモート処理や課金が停止したことを証明しません。

### 3. プロバイダーサービスの停止

実行中の完全評価とそのプロバイダーコンテナを停止するには、`Ctrl-C` を 1 回押します。このリポジトリのプロバイダーサービスコンテナを直接停止するには、次を実行します。

```bash
./provider-services/bin/provider-services stop
```

停止しても、Docker ボリューム、プロバイダー/データベースデータ、保存済み結果、ログ、エビデンスは保持されます。プロバイダー処理の終了後、一時的なライフサイクルマーカーだけが消去されます。中断した完全実行を続けるには、上記の resume コマンドを使用してください。

## 結果とトラブルシューティング

完全なレポートは `outputs/full-test/<run-label>/`、smoke レポートは `outputs/smoke-test/<run-label>/` に書き込まれます。自己完結型 HTML は自動的に開きます。ヘッドレス環境では `OAMB_NO_OPEN=1` を設定してください。レポートは引き続き生成され、パスが表示されます。

### 個別のプロバイダー結果からレポートを生成

Hindsight、Mem0、OpenViking が個別に完了し、1 つのディレクトリ配下に別々のスナップショットとして保存されている場合、プロバイダーを再実行せずにレポートを生成できます。

```bash
./run.sh --generate-report --result-dir=./eval_results/v0.1.0
```

ディレクトリには、各プロバイダーにつき完全な 60 問のスナップショットが正確に 1 つずつ必要で、それぞれに `resolved-plan.json` と `results/<provider>.json` が含まれていなければなりません。バージョン付きの履歴スナップショットでは、ルートに `case-manifest.json` を置き、新規実行が現在のマニフェストを使う場合でも評価済み ID を保持できます。このコマンドは、固定された判定モデルへの接続にルートの `.env` を必要とします。一致する分析キャッシュがない場合は、有界の LLM 分析呼び出しを 1 回行います。レポート入力と分析設定が変わらなければ、`<result-dir>/report-analysis-cache` のコンテンツアドレス型キャッシュを再利用し、モデルを再度呼び出しません。事前チェックやプロバイダーサービスは不要で、元のスナップショットを変更せず、新規作成専用の `comparison/` ディレクトリを書き込みます。その名前がすでに存在する場合は、タイムスタンプ付きの後続ディレクトリを作成します。すべてのソースプラン/結果ハッシュを記録し、保存済みプランから同じ比較制御を証明できない場合は、該当する首位の主張を抑制します。

HTML は比較の焦点を保つため、すべて利用不能な副次的計測と Wilson 区間の説明を省略します。「質問結果」セクションには、少なくとも 1 つのプロバイダーが誤答した質問だけを表示します。完全な 60 問のマトリクスと統計エビデンスは `report.json` に残ります。

### 保存済み精度の確認

回答精度は主要指標の 1 つです。完了または中断した完全実行では、保存済み結果マップを使って `correct / judged` を計算し、`saved / 60` は完了カバレッジとして別に報告します。

```bash
result_file=".../$provider.json"
jq -r '
  [.[] | select(.evaluation.disposition == "judged")] as $judged
  | ($judged | map(.evaluation.numerator) | add // 0) as $correct
  | ($judged | map(.evaluation.denominator) | add // 0) as $total
  | if $total == 0 then "accuracy: unavailable; completion: \(length)/60"
    else "accuracy: \($correct)/\($total) = \(((10000 * $correct / $total | round) / 100))%; completion: \(length)/60"
    end
' "$result_file"
```

各実行は `run: log=<path>` を表示し、端末出力を `outputs/tmp/` に保存します。続けて `tail -f <path>` を実行してください。完全実行の進捗は保存済みの `results/{hindsight,mem0,openviking}.json` から取得され、resume の件数には以前に完了した質問も含まれます。

観測されたプロバイダーの挙動と失敗については、[調査記録](docs/investigations/README.md)を参照してください。

## 謝辞

評価ハーネス、方法論、プロンプト、結果を公開した Vectorize の [Agent Memory Benchmark（AMB）](https://github.com/vectorize-io/agent-memory-benchmark)とそのコントリビューターに感謝します。AMB は OAMB の評価フローと設計における重要な参考資料です。OAMB は独立して実装されています。

また、データセットと判定基準を提供した [LongMemEval の著者](https://github.com/xiaowu0162/LongMemEval)、およびオープンソースのメモリシステムを提供する [Hindsight](https://github.com/vectorize-io/hindsight)、[Mem0](https://github.com/mem0ai/mem0)、[OpenViking](https://github.com/volcengine/OpenViking) の各コミュニティにも感謝します。直接再利用したプロンプト素材には、[出典表記と通知](prompt-packs/README.md)を保持しています。

## プロジェクトポリシー

[セキュリティ](SECURITY.md)、[データセットの出典](DATASETS.md)、[サードパーティコンポーネント](THIRD_PARTY.md)を参照してください。

OAMB は [Apache-2.0](LICENSE) ライセンスで提供されます。データセットとサードパーティ成果物には、それぞれのライセンスが適用されます。

## Mem0 の時間入力互換性に関する注意

OAMB v0.1.0 が評価するのはセルフホスト型 Mem0 OSS 2.0.19 であり、Mem0 Platform v3 ではありません。両者の時間機能は異なります。[Mem0 Platform v3 Temporal Reasoning](https://docs.mem0.ai/platform/features/temporal-reasoning) は、履歴データの書き込みに使用する `timestamp`、検索の基準時刻を指定する `reference_date`、および時間を考慮したランキングをネイティブに提供します。Mem0 のドキュメントでは、この機能は OSS SDK では利用できないと明記されています。

セルフホスト型 OSS プロファイル向けに、OAMB は LongMemEval の取り込み互換性を補強しています。各ソースセッションのタイムスタンプを `metadata.created_at` に保持し、Mem0 の抽出モデルが「昨日」や「先週」などの表現を解釈するときに、そのタイムスタンプを Observation Date として使用するよう指示するカスタム抽出プロンプトを渡します。この 2 つの入力は目的が異なります。`metadata.created_at` は保存時の出所情報を保持してエビデンス検証を可能にし、カスタムプロンプトは抽出時に履歴時刻の意味を与えます。

この補強によって、Mem0 OSS の検索に `reference_date`、時間フィルタリング、または時間を考慮したランキングが追加されるわけではありません。OAMB は保存されたタイムスタンプを検証しますが、回答モデルに渡すのは抽出済みのメモリテキストであり、`created_at` ではありません。したがって、v0.1.0 の temporal-reasoning 9/10 は、適応済みの OAMB と Mem0 OSS によるエンドツーエンドのパイプラインを測定したものです。Mem0 Platform v3 のネイティブ Temporal Reasoning を測定したものではなく、`metadata.created_at` が精度に独立して寄与したことを示すものでもありません。完全なエビデンス境界については、[Mem0 の評価と時間入力に関する調査](docs/investigations/mem0-longmemeval-evaluation-and-temporal-parity.md)を参照してください。

## 使用中の生成AIモデル：DeepSeek V4.1 Flash

OAMB v0.1.0 は、3 つのプロバイダーのメモリ処理、回答、判定において、**DeepSeek V4.1 Flash を唯一の生成モデルとして使用します**。この低コストな選択は v0.1.0 の比較に十分な品質を実現し、ベンチマークをより手頃なコストで再現できるようにします。セマンティックベクトル埋め込みには、別の非生成モデル `qwen3-embedding:0.6b` を使用します。
