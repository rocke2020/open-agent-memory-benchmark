# Open Agent Memory Benchmark

[English](README.md) | [简体中文](README_CN.md) | 日本語

## 概要

Open Agent Memory Benchmark（OAMB）は、Open Agent Memory（OAM）の独立した評価リポジトリで、LME-500 と BEAM-10M の評価手順を公開しています。データセットの準備、実行、レポートを扱い、Hindsight、Mem0、OpenViking との比較評価も別に提供します。

保存済みの OAM v0.0.1 開発版評価は、[LongMemEval の全 500 問](eval_results/lme500/README.md)と [BEAM-10M の完全な 10 会話、200 問](eval_results/beam10m/README.md)を対象にしています。LME-500 の精度は **94.0%（470/500）**、BEAM-10M の再試行ポリシー適用後の最終平均ルーブリックスコアは **61.56%** で、**142/200** 問が 0.5 のしきい値に達しました。これらのレポートは完了済み評価を再利用しており、セットアップや新たな評価呼び出しは不要です。

保存済みの質問、回答、判定、注入されたコンテキスト全文は、オフライン HTML レポートで閲覧できます：[LME-500](eval_results/lme500/report.html) · [BEAM-10M、Hindsight の保存済みコンテキストの再評価を含む](eval_results/beam10m/report.html)。いずれかのファイルをダウンロードして、現在のブラウザーで直接開いてください。

**公開された OAM v0.0.1 評価は、抽出、回答、判定に `deepseek-v4.1-flash` を使用し、ランキングの最高点より日常の Agent タスクを重視しています。モデルの選択は結果に影響します：** Hindsight は回答に **Gemini 3.1 Pro**（`gemini-3.1-pro-preview`）を使用し、[BEAM-10M で **64.1%** と報告しています](https://vectorize.io/benchmarks)。保存済みコンテキストを DeepSeek で再評価した結果は、[条件を満たした 43 件の max 思考強度での回答再試行後に **52.16%**](eval_results/beam10m/README.md)で、平均ルーブリックスコアには約 **12 ポイント**の差があります。

[ネイティブ OAM ワークフロー](docs/native-oam.md)は、`./precheck.sh --oam` と `./run.sh --oam` による LME-500 と BEAM-10M の実行を提供します。リポジトリには、固定されたデータセットのダウンロード/エクスポートスクリプト、記録済みの回答・判定プロンプト、設定可能な回答・判定モデル、およびレビュー後に max 思考強度で 1 回だけ回答を再試行する手順が含まれています。レポートは初回とポリシー適用後の最終スコアを分けて保持します。小規模診断は実行経路を確認するもので、全データセットで公開スコアに近づくかは別途測定する必要があります。

**全 LME-500 と BEAM-10M に対するネイティブ OAM 評価が OAMB の主目的です。** 副次的な比較プロファイルは、Hindsight、Mem0、OpenViking のネイティブ REST API を使用します。共通の均衡選択された LongMemEval 60 問、6 種類から各 10 問を評価し、180 件のプロバイダー別結果を生成します。

オフラインレポートは 5 つの指標を個別に示します。

- **回答精度**：検索されたメモリを使った回答の正しさ。
- **コンテキストトークン**：回答モデルに提示された検索エビデンスの量。
- **インデックス作成トークン**：メモリ構築に使った生成モデルについて提供元が報告したトークン数。埋め込みとプロバイダーのユーザー初期設定は除きます。
- **検索レイテンシ**：プロバイダーのメモリ検索リクエストの所要時間。
- **インデックス作成時間**：履歴の最初の書き込みから検索可能になるまでの時間。

OAMB は、実行前に質問、モデルの役割、回答/判定ポリシー、比較設定を決定します。各プロバイダーはネイティブのストレージおよびインデックス作成動作を維持します。再試行、失敗、部分的な取り込み、コスト、取得不能な測定値はすべて可視化され、普遍的な総合スコアは作りません。検索ではクエリ埋め込みを許可しますが、生成的なクエリ書き換え、リフレクション、再ランキングは無効です。

## データセット

| データセット | 評価対象 | 実行ワークフロー | 採点 |
|---|---|---|---|
| LongMemEval S、全 LME-500 | 500 問 | [LME-500 設定](configs/oam-lme500.yml)によるネイティブ OAM | 質問カテゴリー別の正誤判定 |
| BEAM-10M | 完全な 10 会話、10 カテゴリーの 200 問 | [BEAM-10M 設定](configs/oam-beam10m.yml)によるネイティブ OAM | 各質問のルーブリック項目を平均し、全質問を等しく重み付けして平均 |
| LongMemEval S、均衡 LME-60 | 6 種類から各 10 問、計 60 問 | 副次的な Hindsight、Mem0、OpenViking の REST 比較。180 件のプロバイダー別結果 | 質問カテゴリー別の正誤判定 |

BEAM は、回答の差し控え、矛盾の解消、イベント順序、情報抽出、指示への準拠、知識更新、複数セッション推論、好みへの準拠、要約、時間推論を対象とします。平均ルーブリックスコアと、0.5 以上のスコアを得た質問の割合は異なる指標です。

比較ワークフローでは `./precheck.sh` が LongMemEval をダウンロードします。単独で `./scripts/download/longmemeval.sh` を実行することもできます。`./scripts/download/beam10m.sh` は、固定された公式 BEAM-10M Parquet シャードをダウンロードし、評価時と同じ JSON シリアライズと SHA-256 で `datasets/beam/10m.json` をエクスポートします。両ヘルパーは、既存ファイルが一致しなければ保持したまま拒否します。ダウンロードしたデータセットは Git に含めません。[データセットの出典](DATASETS.md)を参照してください。

[ネイティブ OAM ガイド](docs/native-oam.md)で隔離されたデプロイを準備し、`./precheck.sh --oam`、`./run.sh --oam` の順にいずれかの全データセットを実行します。両データセット用の合成診断設定もあります。新規実行は評価した OAM のリビジョンと設定を記録し、保存済みの [LME-500](eval_results/lme500/README.md) と [BEAM-10M](eval_results/beam10m/README.md) レポートは過去の結果を保持します。

## プロバイダーのバージョン

OAMB は **OAM v0.0.1** を主な対象とし、比較対象には以下のリリースとソースコミットを使用します。

| プロバイダー | リリース | ソースコミット |
|---|---|---|
| Open Agent Memory（OAM） | **0.0.1 開発スナップショット** | `32f97cca5c5c1fd2546ed0364d2279063afc81b6` |
| Hindsight | [0.9.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.9.2) | [`ebad478240d3171bb88201ececda5e8d9883d22d`](https://github.com/vectorize-io/hindsight/commit/ebad478240d3171bb88201ececda5e8d9883d22d) |
| Mem0 | [2.0.19](https://github.com/mem0ai/mem0/releases/tag/v2.0.19) | [`dc82354e143c2581d505d581a00286d6ef8c3605`](https://github.com/mem0ai/mem0/commit/dc82354e143c2581d505d581a00286d6ef8c3605) |
| OpenViking | [0.4.19](https://github.com/volcengine/OpenViking/releases/tag/v0.4.19) | [`f3afef11637f2d7c11e4b1f36ed2f90630737cdc`](https://github.com/volcengine/OpenViking/commit/f3afef11637f2d7c11e4b1f36ed2f90630737cdc) |

[`provider-services/versions.env`](provider-services/versions.env) は比較対象プロバイダーのイメージとソースアーカイブを固定します。OAM 開発イメージは、上記のソースコミットから以下の手順でビルドします。

### OAM 開発イメージ

記録されたコミットを含む OAM チェックアウトから v0.0.1 開発イメージをビルドします。

```bash
./provider-services/oam/build-image.sh /path/to/open-agent-memory 32f97cca5c5c1fd2546ed0364d2279063afc81b6
docker run --rm oamb/oam:0.0.1-32f97cca5c5c --version
```

ビルドはコミット済みソースと OAM ロックファイルを使用し、ローカルの未コミットファイルを含めません。作成されるローカルイメージは開発スナップショットであり、公開リリースイメージではありません。動作する OAM デプロイには、隔離された PostgreSQL データベース、適切な認証情報、設定済みのモデルサービスも必要です。[ネイティブ OAM の設定・実行ガイド](docs/native-oam.md)で、隔離された PostgreSQL/API/worker の準備、入力とモデルのバインド確認、チェックポイント、再開手順を確認してください。

`provider-services/oam/` の評価用上書き設定は、LME-500 に共有 `small` ベクトルインデックス、BEAM-10M にユーザーごとの専用 `large` インデックスを設定します。両方とも Insights と Profile を無効にし、抽出の思考強度を low にします。評価器は LME に fast 検索、BEAM に normal 検索を要求します。これらはリクエストパラメーターであり、インデックス設定ではありません。その他の設定はイメージ内の OAM の正式設定に従います。過去の実行と一致すると主張する前に、記録されたプロトコルと照合してください。

過去の評価は、32K トークンの入力上限を持つ Qwen3 埋め込みと OAM の近似トークナイザーを使用しました。現在の OAM は取り込みの分割処理に `toktok-rs` の `o200k_base` を既定で使用するため、付属のイメージと上書き設定だけでは過去の分割を再現しません。分割サイズを正確にするには、埋め込みモデル自身のトークナイザーを優先し、実際の入力・リクエスト上限を設定してください。例えば `cl100k_base` は OpenAI の `text-embedding-3` モデルに対応します。使用前に対象モデルのトークナイザー、入力形式、上限を確認してください。既定のエンコーディングがすべてのモデルに適合するわけではありません。

現在の OAM は、取り込み分割用の `tokenizer.type` として `o200k_base`、`cl100k_base`、従来の `oam-unicode-quarter-v1` カウンター、モデル固有の `tokenizer_json` アセットをサポートします。埋め込み入力の検証には別の `models.embedding.tokenizer` 設定があります。`auto` は認識された OpenAI と Qwen3-Embedding のモデル名に適合するトークナイザーを選び、`openai` は明示的に `cl100k_base`、`qwen` は同梱の Qwen3-Embedding トークナイザーを選びます。認識可能なモデル名では、適切なバリアントを選ぶため `auto` を優先してください。トークナイザーオブジェクトでカスタム `tokenizer_json` アセットを指定することもできます。

固定された開発イメージには、これらの自動埋め込みプロファイルと `openai`/`qwen` オプションが含まれます。サービスに合わせて `models.embedding.max_input_tokens`、`max_request_tokens`、`max_request_items` を設定し、埋め込み次元を確認してください。対応するカウンターがモデルに適合しなければ、近似値として余裕を持たせてください。提供元の入力上限内に収まる保証にはなりません。埋め込みモデルやトークナイザーの変更は分割、埋め込み、結果を変える可能性があるため、過去の評価と同一とせず、変更内容を報告してください。

ネイティブ BEAM ワークフローは、各会話の埋め込み済み検索単位が作成された後に専用インデックスを準備し、検索、回答、判定へ進みます。小規模な合成サンプルでは実際の取り込み、検索、回答、判定、オフライン HTML 閲覧を実施していますが、過去の結果や全データセットの結果とは別です。正確なコマンド、成果物の境界、現在の制約は[ワークフローガイド](docs/native-oam.md)を参照してください。

## LME-500 と BEAM-10M で OAM を評価する

[ネイティブ OAM 設定ガイド](docs/native-oam.md)に従って OAM イメージをビルドし、非公開の `.env.oam` にモデルエンドポイントを設定して、各データセット用に別の隔離されたデプロイを起動します。別の OAM チェックアウトが製品イメージを提供し、評価はこのリポジトリから実行します。

抽出、回答、判定には **`deepseek-v4.1-flash`**、埋め込みには **`qwen3-embedding:0.6b` と 1,024 次元ベクトル**を使用します。[.env.oam.example](.env.oam.example) を非公開の `.env.oam` にコピーし、エンドポイントとキーを設定して、両方のモデル役割を選択した YAML の `model` に合わせてください。`deepseek-flash` などのゲートウェイ別名は、対象モデルに解決される場合にこの再現目的に適しています。抽出は low、初回回答と判定は high の思考強度を使用します。失敗した質問はレビュー後に、同じコンテキストとプロンプトで 1 回だけ max 思考強度の回答再試行を行い、その後 high で判定できます。

入力と 2 つのデプロイを準備したら、以下の全データセット設定を実行します。これらのコマンドは課金対象の呼び出しを行います。クライアント環境のパスは、各デプロイが出力したものに置き換えてください。

```bash
./precheck.sh --oam configs/oam-lme500.yml \
  --env-file outputs/oam-services/lme-example/client.env \
  --output outputs/oam-lme500
./run.sh --oam outputs/oam-lme500

./precheck.sh --oam configs/oam-beam10m.yml \
  --env-file outputs/oam-services/beam-example/client.env \
  --output outputs/oam-beam10m
./run.sh --oam outputs/oam-beam10m
```

各出力ディレクトリには `report.html`、`summary.json`、保持された質問ごとのエビデンスが含まれます。中断した評価は `./run.sh --oam <output-directory> --resume` で再開します。全データセットの実行前に、対応する `*-smoke.yml` 設定と新しい出力ディレクトリで小規模な合成診断を実行できます。

初回に失敗した質問については、生成された `retry-eligibility.json` のソース、コンテキスト、正解のレビューを完了してから再開してください。明らかに誤った正解は再試行を禁止します。再試行が完了すると、スコアが下がった場合もその結果がポリシー適用後の最終結果になります。レビュー待ちは可視化され、最終評価の完了を主張できません。[レビューと再開手順](docs/native-oam.md#3-full-datasets-and-resume)を参照してください。

## クイックスタート：LME-60 比較

以下のコマンドは副次的なプロバイダー比較を開始します。主目的の全データセット OAM 評価には、[上記のネイティブ評価ワークフロー](#lme-500-と-beam-10m-で-oam-を評価する)を使用してください。保存済みの [LME-500](eval_results/lme500/README.md) と [BEAM-10M](eval_results/beam10m/README.md) 評価の閲覧にセットアップは不要です。


Git、Python 3.11 以降、`uv`、Compose 対応の Docker、`curl`、`jq`、`shasum`、および OpenAI 互換モデルエンドポイントの認証情報が必要です。準備と評価では課金対象のモデル呼び出しが発生する場合があります。処理は隔離されたプロバイダー状態を作成し、既存のプロバイダー/データベースデータを保持します。

### 1. 設定

リポジトリをクローンし、非公開設定を準備します。

```bash
git clone https://github.com/rocke2020/open-agent-memory-benchmark.git
cd open-agent-memory-benchmark
cp .env.example .env
chmod 600 .env
```

`.env` を編集し、`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_LIGHT_MODEL`、`LLM_DEEP_MODEL` を設定します。`LLM_URL_TYPE=openai_chat` はそのままにしてください。[テンプレート](.env.example)では、両方のモデルプロファイルの既定値に `deepseek-flash` を使用します。light プロファイルはプロバイダーの抽出と判定を担当し、deep プロファイルは回答を担当します。[ベンチマーク設定](configs/benchmark.yml)が役割の割り当て、思考強度、実行制御の信頼できる情報源です。

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

リクエストと検証の境界は、[プロバイダーの検索仕様](docs/designs/retrieval-top-k-provider-semantics.md)を参照してください。

1 つの履歴内のセッションは逐次処理されます。これらの制限は、プロバイダー内部の埋め込み/モデルリクエストを制限しません。新規実行は事前チェック済みプランを使用します。各 `--resume` は、未完了の処理に対して現在の YAML から履歴と質問の上限を読み取ります。この 2 つの値を変更しても完了済みの結果は無効になりません。その他のプラン設定は固定されたままで、実行中のプロセスは YAML を再読み込みしません。`resolved-plan.json` は編集しないでください。

再試行と試行ごとのタイムアウト設定は、これらの制御とともに設定ファイルに記載されています。実行全体や支出の上限ではありません。ローカルのタイムアウトは、リモート処理や課金が停止したことを証明しません。

### 3. プロバイダーサービスの停止

実行中の完全評価とそのプロバイダーコンテナを停止するには、`Ctrl-C` を 1 回押します。このリポジトリのプロバイダーサービスコンテナを直接停止するには、次を実行します。

```bash
./provider-services/bin/provider-services stop
```

停止しても、Docker ボリューム、プロバイダー/データベースデータ、保存済み結果、ログ、エビデンスは保持されます。プロバイダー処理の終了後、一時的なライフサイクルマーカーだけが消去されます。中断した完全実行を続けるには、上記の resume コマンドを使用してください。

## LME-60 の結果とトラブルシューティング

完全なレポートは `outputs/full-test/<run-label>/`、smoke レポートは `outputs/smoke-test/<run-label>/` に書き込まれます。自己完結型 HTML は自動的に開きます。ヘッドレス環境では `OAMB_NO_OPEN=1` を設定してください。レポートは引き続き生成され、パスが表示されます。

### 個別のプロバイダー結果からレポートを生成

Hindsight、Mem0、OpenViking が個別に完了し、1 つのディレクトリ配下に別々のスナップショットとして保存されている場合、プロバイダーを再実行せずにレポートを生成できます。

```bash
./run.sh --generate-report --result-dir=outputs/saved-results/lme60-provider-native
```

ディレクトリには、各プロバイダーにつき完全な 60 問のスナップショットが正確に 1 つずつ必要で、それぞれに `resolved-plan.json` と `results/<provider>.json` が含まれていなければなりません。このコマンドは、固定された判定モデルへの接続にルートの `.env` を必要とします。一致する分析キャッシュがない場合は、有界の LLM 分析呼び出しを 1 回行います。レポート入力と分析設定が変わらなければ、`<result-dir>/report-analysis-cache` のコンテンツアドレス型キャッシュを再利用し、モデルを再度呼び出しません。事前チェックやプロバイダーサービスは不要で、元のスナップショットを変更せず、新規作成専用の `comparison/` ディレクトリを書き込みます。その名前がすでに存在する場合は、タイムスタンプ付きの後続ディレクトリを作成します。すべてのソースプラン/結果ハッシュを記録し、保存済みプランから同じ比較制御を証明できない場合は、該当する首位の主張を抑制します。

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
