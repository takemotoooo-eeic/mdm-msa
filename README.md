# mdm-msa

Music structure analysis experiments built on [SongFormer](https://github.com/ASLP-lab/SongFormer)
([paper](https://arxiv.org/abs/2510.02797)), on a Hydra experiment template.

The model works on **pre-extracted SSL features** (MusicFM and MuQ, layer 10, 30 s and 420 s windows).
Audio loading, feature extraction, and the Gradio app from upstream are not included.

## Layout

```
mdmsa/                       # installable Python package
  amp/                       # thin torch.amp wrappers (autocast / GradScaler)
  bin/                       # CLI entry points (mdmsa-parse-run-command, ...)
  configs/                   # Hydra YAML, one group per target
    dataloader/songformdb.yaml
    model/songformer.yaml
    loss/songformer.yaml
    optimizer/songformer.yaml
    train/songformer.yaml
  loss/songformer.py         # BoundaryBCELoss, BoundaryTVLoss, FunctionCrossEntropyLoss, FunctionFocalLoss
  models/songformer.py       # SongFormer (loads the released checkpoint unchanged)
  optim/                     # CosineWarmupLR
  utils/
    data/                    # SongFormDBDataset, SongFormBenchDataset, label tables
    collator/                # SongFormerCollator, SongCollator
    driver/                  # SongFormerTrainer / SongFormerEvaluator, EMA, loss Balancer
    postprocess/             # peak picking, segment decoding, rule-based cleanup, chunk stitching
    metrics/                 # HR.5F / HR3F / HR1F, PWF, Sf, ACC, IoU
recipes/
  _common/local/             # Python entry points for train / eval
  SongFormer/                # train.sh / eval.sh (also usable with sbatch)
```

## Setup

```bash
uv sync --extra dev
uv run pre-commit install
```

Pretrained weights (the released SongFormer checkpoint) go to `ckpts/`:

```bash
mkdir -p ckpts
curl -L -o ckpts/SongFormer.safetensors https://huggingface.co/ASLP-lab/SongFormer/resolve/main/SongFormer.safetensors
```

## Data

Both datasets are read as features laid out like this:

```
<root>/embeddings/{30s_420s,420s}/<subset>/{musicfm,muq}_hop420/layer_10/<song_id>_<start_sec>.npy   # (1, T, 1024), 25 Hz
```

- **SongFormDB** (training, `--db_root`): subsets `HX`, `Ext`, `Hook`, `Gem`, with `train_meta/labels/*.jsonl` and `train_meta/splits/*.txt`.
  `HX` validation (`hx_val.txt`) is used for validation during training.
- **SongFormBench** (evaluation, `--bench_root`): subsets `HarmonixSet`, `CN`, with references in `data/labels/<subset>/<song_id>.txt`.

| Subset | Dataset id | Supervision |
| --- | --- | --- |
| HX | `SongForm-HX-8Class` (5), x4 | boundaries + labels |
| Ext | `SongForm-Private` (3) | boundaries + labels |
| Hook | `SongForm-Hook` (6) | boundaries + labels; unlabeled gaps are ignored |
| Gem | `SongForm-Gem-Only-Label` (8) | labels only (`NO_LABEL` ignored) |

Evaluation and inference condition on `SongForm-HX-8Class` (8 classes, pre-chorus kept).

## Train

```bash
./recipes/SongFormer/train.sh --tag songformer_42
# or on Slurm (submit from inside the repository)
sbatch recipes/SongFormer/train.sh --tag songformer_42
```

Options: `--tag`, `--db_root`, `--bench_root`, `--base_exp_dir`, `--wandb_enabled true|false`, `--iterations N`, then any Hydra override,
e.g. `train.seed=8988`, `dataloader.train.batch_size=8`, `loss=...`.

Training follows upstream: Adam (lr 1e-4) with 300 warm-up steps and cosine decay over 12,010 optimizer steps, batch 4 with
2-step gradient accumulation, EMA weights, a gradient-norm balancer between the section and function losses, and validation
every 800 steps with early stopping (patience 3 on HR.5F / ACC).

Outputs in `exp/<tag>/`:

- `exp/model/iteration<N>.pth`, `best.pth`, `last.pth` (model, EMA, optimizer, scheduler, scaler, resolved config)
- `tensorboard/` (`tensorboard --logdir exp/<tag>/tensorboard`)

Resume with `train.checkpoint.resume_from=exp/<tag>/exp/model/last.pth`.

## Evaluate / infer

```bash
./recipes/SongFormer/eval.sh --checkpoint ckpts/SongFormer.safetensors --tag eval-pretrained
./recipes/SongFormer/eval.sh --checkpoint exp/<tag>/exp/model/best.pth
```

`--checkpoint` accepts checkpoints from this repository, upstream `.pt`, and `.safetensors`. EMA weights are used when present
(`train.evaluation.use_ema=false` to use the raw model).

Results go to `exp/<tag>/eval/`:

- `<subset>/estimations/<song_id>.txt`: predicted segments (`<start> <label>` lines, last line `<end> end`)
- `<subset>/<mapping>/per_song.csv`, `summary.csv`, and `summary.md` for all subsets

`<mapping>` is `normal` (8 classes), `prechorus2verse` (7 classes, as in the paper) or `prechorus2chorus`.
For inference on other features, drop `label_dir` from the `dataloader.test` subsets; only estimations are written.

Released checkpoint on SongFormBench (`prechorus2verse`):

| Subset | ACC | HR.5F | HR3F |
| --- | --- | --- | --- |
| HarmonixSet | 0.807 | 0.713 | 0.799 |
| CN | 0.895 | 0.696 | 0.864 |

## AMP

```bash
./recipes/SongFormer/train.sh train.torch_dtype=bfloat16
```

autocast and GradScaler are enabled only when `torch_dtype` is `float16` or `bfloat16`.
