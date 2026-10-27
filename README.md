# exp-templete

A Hydra-based research experiment template. It includes a small MNIST image classifier as a working example.

## Layout

```
exptemplete/                 # installable Python package
  amp/                       # thin torch.amp wrappers (autocast / GradScaler)
  bin/                       # CLI entry points (exptemplete-parse-run-command, ...)
  configs/                   # Hydra YAML, split by target
    dataloader/
    train/
    model/
    optimizer/
    loss/
  loss/
  models/
  utils/
    driver/                  # Trainer / Evaluator
    data/
    wandb/
    _tensorboard/
    _hydra/
recipes/
  _common/local/             # Python entry points for train / eval
  MNIST/                     # dataset-specific run scripts
```

Hydra uses `exptemplete/configs/config.yaml` as the root. Switch `dataloader`, `train`, `model`, `optimizer`, and `loss` with YAML groups, then build objects with `hydra.utils.instantiate`.

## Use as a template

This repository is a GitHub template. Click **Use this template** → **Create a new repository** on GitHub to start a new project from it. You may then rename the project (`exp-templete` / `exptemplete`); see "プロジェクト名の変更" in [AGENTS.md](AGENTS.md).

## Setup

Python 3.10 or later is required.

```bash
git clone https://github.com/takemotoooo-eeic/exp-templete.git
cd exp-templete
uv sync
# or
pip install -e .
```

The package is importable after install:

```python
import exptemplete
from exptemplete.amp import autocast, should_enable_amp
from exptemplete.amp.grad_scaler import GradScaler
```

`exptemplete-parse-run-command` is available on PATH once the package is installed.

## Train on MNIST

```bash
cd recipes/MNIST
./train.sh
```

Common options (`parse_options.sh`):

```bash
./train.sh --tag cnn_mnist --wandb_enabled true --epochs 10 --data_dir /path/to/data
```

Hydra groups are selected by variables in the script:

- `dataloader=mnist`
- `train=classification`
- `model=cnn`
- `optimizer=adam`
- `loss=cross_entropy`

Checkpoints go to `exp/<tag>/exp/model/`. TensorBoard logs go to `exp/<tag>/tensorboard/`.

```bash
tensorboard --logdir exp/<tag>/tensorboard
```

To use wandb, run `wandb login` first, then `./train.sh --wandb_enabled true`.

## Evaluate on MNIST

```bash
./eval.sh --checkpoint exp/<tag>/exp/model/best_epoch.pth
```

## Add a new dataset

1. Add a Dataset class under `exptemplete/utils/data/`.
2. Add `exptemplete/configs/dataloader/<name>.yaml`.
3. Add `model` / `loss` / `train` YAML files if needed.
4. Create `recipes/<DATASET>/` with `train.sh` and `eval.sh`, and symlink `local` to `_common/local`.

## AMP

The training loop uses `exptemplete.amp`. For mixed precision:

```bash
./train.sh train.torch_dtype=float16
```

autocast and GradScaler are enabled only when `torch_dtype` is `float16` or `bfloat16`.
