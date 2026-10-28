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
    diffuser/masked_diffusion.yaml   # masked diffusion settings (with *_diffusion model / loss / train)
  loss/songformer.py         # BoundaryBCELoss, BoundaryTVLoss, FunctionCrossEntropyLoss, FunctionFocalLoss
  models/songformer.py       # SongFormer (loads the released checkpoint unchanged)
  models/songformer_diffusion.py  # SongFormerDiffusion (masked diffusion variant)
  models/diffuser.py         # MaskedDiffuser: add_mask / unmask / inference / decode
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
- **SongFormBench** (evaluation, `--bench_root`): subsets `HarmonixSet`, `CN`, `RWC-P`, with references in `data/labels/<subset>/<song_id>.txt`.
  `RWC-P` (RWC Music Database Popular, 100 songs) is added locally from the
  [AIST CHORUS annotations](https://github.com/rwc-music/rwc-annotations-archive): `<bench_root>/utils/RWC-P/prepare_labels.py`
  converts the labels (`verse A` → verse, `post-chorus` → chorus, `ending` → outro, `nothing` → silence, pre-chorus kept) and
  `extract_features.sh` extracts the features with the md-songformer scripts.

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
| RWC-P | 0.745 | 0.711 | 0.800 |

RWC-P `prechorus2verse` follows the protocol of [Wang et al. 2022](https://arxiv.org/abs/2205.14700) (AIST annotations, same label
rules), whose best RWC-Pop result is ACC 0.675 / HR.5F 0.623 (SpecTNT, cross-dataset). AIST `bridge` is mostly instrumental
interludes and is kept as `bridge` for comparability, which is the main source of RWC-P label errors.

## Masked diffusion

A masked discrete diffusion variant (after md-songformer) predicts boundary and function **tokens** instead of
regressing logits. It shares the dataset, metrics and drivers with the regression model; only these groups change:

| Group | Regression | Masked diffusion |
| --- | --- | --- |
| `model` | `songformer` | `songformer_diffusion` (adds boundary / function token embeddings) |
| `loss` | `songformer` | `songformer_diffusion` (loss on hidden frames only, no TV) |
| `train` | `songformer` | `songformer_diffusion` (adds the `diffuser` config) |

```bash
./recipes/SongFormer/train.sh --tag md_42 --train songformer_diffusion --model songformer_diffusion --loss songformer_diffusion
./recipes/SongFormer/eval.sh --checkpoint exp/md_42/exp/model/best.pth --train songformer_diffusion --model songformer_diffusion --loss songformer_diffusion
```

`mdmsa.models.MaskedDiffuser` ([`configs/diffuser/masked_diffusion.yaml`](mdmsa/configs/diffuser/masked_diffusion.yaml)) does the diffusion part:

- `add_mask`: hides a random share of the ground-truth tokens of each training batch (`function_masking`: `random`,
  `random_span`, `section`, `section_merged`; `p_all_mask` hides whole songs) and restricts the loss to hidden frames.
- `inference` / `unmask`: starts from all-MASK and reveals tokens by confidence over `num_inference_steps`
  (`unmask_strategy`: `dynamic`, `fixed_alpha`, `confidence`), keeping one boundary per `boundary_peak_window` frames.
- `decode`: every boundary EVENT starts a segment labelled by its majority function token.

Override settings with e.g. `diffuser.boundary.num_inference_steps=4 diffuser.function_masking=random_span`.

## Without the dataset id

`model=songformer_no_dataset_id` / `model=songformer_diffusion_no_dataset_id` drop the dataset (data source) prefix
embedding, so annotation-policy differences between training datasets are no longer given to the model:

```bash
./recipes/SongFormer/train.sh --tag reg_nodsid --dataloader songformdb --model songformer_no_dataset_id
./recipes/SongFormer/train.sh --tag md_nodsid --dataloader songformdb --train songformer_diffusion --model songformer_diffusion_no_dataset_id --loss songformer_diffusion
./recipes/SongFormer/eval.sh --checkpoint exp/md_nodsid/exp/model/best.pth --dataloader songformdb --train songformer_diffusion --model songformer_diffusion_no_dataset_id --loss songformer_diffusion
```

## AMP

```bash
./recipes/SongFormer/train.sh train.torch_dtype=bfloat16
```

autocast and GradScaler are enabled only when `torch_dtype` is `float16` or `bfloat16`.
