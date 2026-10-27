# AGENTS.md

このリポジトリでコードを書くエージェント（および人間）向けの実装方針。

## 基本方針

- **ベースとなるコードは変更しない。** 実験ごとに書き足すのはモデル・ロス・データセット・設定（YAML）・レシピだけにする。
- 新しい要素は **新しいクラス / 新しいファイル / 新しい YAML** として追加し、Hydra の `_target_` で差し替える。既存クラスを書き換えて挙動を変えない。
- ベースコードに不具合や不足があっても、勝手に直さず、内容と理由を報告して判断を仰ぐ。
- 例外として、**プロジェクト名（`exp-templete` / `exptemplete`）はベースコードを含めて変更してよい**（下の「プロジェクト名の変更」を参照）。
- 以下の説明に出てくる `exptemplete` は、改名した場合は新しいパッケージ名に読み替える。

## プロジェクト名の変更

このリポジトリは GitHub のテンプレートリポジトリとして使う想定。テンプレートから新しいリポジトリを作ったら、まずプロジェクト名を変更してよい。変更するのは **名前の置き換えだけ** で、ロジックには手を入れない。

| 名前 | 使われている場所 |
| --- | --- |
| パッケージ名 `exptemplete` | ディレクトリ `exptemplete/`、`pyproject.toml`（`name`、`[project.scripts]`、`packages.find.include`、`package-data`、`dynamic.version`）、全 Python ファイルの import、YAML の `_target_`、`recipes/_common/local/*.py` |
| CLI 名 `exptemplete-parse-run-command` | `pyproject.toml` の `[project.scripts]`、`recipes/*/train.sh`、`recipes/*/eval.sh` |
| リポジトリ名 `exp-templete` | `README.md`（タイトル・clone URL）、`configs/train/*.yaml` の `wandb.project` |

手順:

1. `git mv exptemplete <new_package>` でディレクトリを改名する。
2. 上の表の場所を `grep -rn "exptemplete\|exp-templete" . --exclude-dir=.git --exclude=uv.lock` で洗い出して置き換える。
3. `uv lock` で `uv.lock` を更新する。
4. `uv run pre-commit run --all-files` を通し、`./train.sh --epochs 1` で動作を確認する。

## 変更してはいけないもの（ベースコード）

| パス | 役割 |
| --- | --- |
| `exptemplete/amp/` | `torch.amp` の autocast / GradScaler ラッパー |
| `exptemplete/bin/` | CLI エントリポイント（`exptemplete-parse-run-command` など） |
| `exptemplete/__init__.py` | `exptemplete.main`（Hydra デコレータ）と `__version__` |
| `exptemplete/utils/__init__.py`, `utils/_torch/`, `utils/_hydra/`, `utils/_omegaconf/`, `utils/_tensorboard/`, `utils/logging/`, `utils/wandb/` | 共通ユーティリティ（seed・device・dtype・ロガー・TensorBoard・wandb） |
| `exptemplete/utils/driver/base.py` | Driver の基底クラス |
| `exptemplete/loss/__init__.py` の `Loss`, `TotalLoss` | ロスの重み付け・合算の仕組み |
| `exptemplete/configs/config.yaml`, `configs/parse_run_command.yaml`, `configs/*/default.yaml` | Hydra のルート設定と各グループの雛形 |
| `recipes/_common/` | `train.py` / `eval.py` / `parse_options.sh` |
| `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml` | ビルド・依存・lint 設定（依存追加が必要な場合は相談する） |

## 実装する場所

| 追加するもの | 置き場所 | 設定ファイル |
| --- | --- | --- |
| モデル | `exptemplete/models/`（大きくなる場合は `models/<name>.py` を作り `models/__init__.py` で re-export） | `configs/model/<name>.yaml` |
| ロス | `exptemplete/loss/`（同上） | `configs/loss/<name>.yaml` |
| データセット | `exptemplete/utils/data/` | `configs/dataloader/<name>.yaml` |
| collate 関数 | `exptemplete/utils/collator/` | `configs/dataloader/<name>.yaml` 内の `collate_fn` |
| オプティマイザ / スケジューラ | 既存の `torch.optim` を使う（自作が必要なら新規モジュール） | `configs/optimizer/<name>.yaml` |
| 学習・評価ループ（新しいタスク） | `exptemplete/utils/driver/<task>.py` に新規 Trainer / Evaluator を作り `driver/__init__.py` で export | `configs/train/<task>.yaml` |
| 実行スクリプト | `recipes/<DATASET>/train.sh`, `eval.sh`、`local -> ../_common/local` のシンボリックリンク | — |

`exptemplete/utils/driver/classifier.py` と `exptemplete/models/__init__.py` の `CNNClassifier` は MNIST 用の **実装例**。新タスクではこれを書き換えず、コピーして新しいファイルを作る。

## 実装ルール

### モデル

- `torch.nn.Module` を継承し、コンストラクタ引数は YAML から渡せる型（int / float / str / list / dict）にする。
- ハイパーパラメータはハードコードせず、コンストラクタ引数として YAML に出す。
- `forward` の入出力は、使う Driver が期待する形に合わせる（例: `ClassifierTrainer` は `logits = model(images)` で `(batch, num_classes)` を期待）。

```yaml
# configs/model/<name>.yaml
_target_: exptemplete.models.MyModel
in_channels: 1
num_classes: 10
```

### ロス

- 個々のロスは `nn.Module` として実装し、`forward(input, target) -> torch.Tensor`（スカラー）を返す。
- 重み付けと合算は自前で書かず、既存の `Loss`（重み）と `TotalLoss`（合算）で YAML 上で組み立てる。

```yaml
# configs/loss/<name>.yaml
_target_: exptemplete.loss.TotalLoss
losses:
  - _target_: exptemplete.loss.Loss
    loss_module:
      _target_: exptemplete.loss.CrossEntropyLoss
    weight: 1.0
  - _target_: exptemplete.loss.Loss
    loss_module:
      _target_: exptemplete.loss.MyAuxLoss
    weight: 0.1
```

### データセット・DataLoader

- `torch.utils.data.Dataset` を継承し、前処理（transform）はクラス内か引数で持つ。
- `configs/dataloader/<name>.yaml` は `default` を継承し、`train` / `validate` / `test` の3つを定義する（`mnist.yaml` を参照）。

### Driver（学習・評価ループ）

- `exptemplete.utils.driver.base.Driver` を継承し、`build_from_config(cls, config)` を実装する。
- AMP は必ず `exptemplete.amp`（`autocast`, `should_enable_amp`, `get_autocast_device_type`, `GradScaler`）経由で使う。`torch.amp` を直接呼ばない。
- seed・device・dtype は `exptemplete.utils` の `set_seed` / `set_device` / `convert_dtype` を使う。
- ログは `utils.logging.get_logger`、TensorBoard は `utils._tensorboard.get_writer`、wandb は `train.wandb` 設定から `hydra.utils.instantiate` で作る。
- チェックポイントには `model` / `optimizer` / `scaler` / `scheduler` / `epoch` / `iteration` / `resolved_config` / `_metadata` を保存する（`classifier.py` の `save_checkpoint` と同じ形式）。

### 設定（Hydra）

- オブジェクトはすべて `_target_` + `hydra.utils.instantiate` で生成する。Python コード内で import して直接インスタンス化しない。
- 実験条件の切り替えは YAML グループ（`model=`, `loss=` など）かコマンドラインの override で行う。コード内に `if experiment == ...` のような分岐を書かない。

## コーディング規約

- フォーマッタ / リンタは ruff（`pyproject.toml` の設定: line-length 160、ルール `E`, `F`, `I`）。
- コミット前に pre-commit で `ruff check --fix` と `ruff format` が走る。初回のみ以下でセットアップする。

```bash
uv sync --extra dev
uv run pre-commit install
```

- 全ファイルに手動でかける場合:

```bash
uv run pre-commit run --all-files
```

- 型ヒントを付ける。公開するクラス・関数は各モジュールの `__all__` に追加する。

## 実装後のチェックリスト

1. ベースコード（上の「変更してはいけないもの」）に、プロジェクト名の置き換え以外の差分が出ていないこと（`git diff --stat` で確認）。
2. 追加したクラスが `__all__` と YAML の `_target_` で正しく参照されていること。
3. `uv run pre-commit run --all-files` が通ること。
4. 少ないエポックで学習が最後まで回ること（例: `./train.sh --epochs 1`）。
5. 学習で出たチェックポイントで `./eval.sh --checkpoint ...` が動くこと。
