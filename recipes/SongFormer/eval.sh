#! /bin/bash

#SBATCH --job-name=songformer-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.log

set -eu
set -o pipefail

# Under sbatch the script runs from a spool copy, so search upward from the submit directory.
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
while [ ! -f "${project_root}/mdmsa/configs/config.yaml" ]; do
    if [ "${project_root}" = "/" ]; then
        echo "$0: run from inside the mdm-msa repository" 1>&2
        exit 1
    fi
    project_root="$(dirname "${project_root}")"
done
local_dir="${project_root}/recipes/_common/local"

tag=""
db_root="/home/sarulab/kengo_takemoto/data/SongFormerDB"
bench_root="/home/sarulab/kengo_takemoto/data/SongFormBench"
checkpoint=""
wandb_enabled=false

dataloader="songformdb"
train="songformer"
model="songformer"
optimizer="songformer"
loss="songformer"

help_message="Usage: $0 --checkpoint PATH (.pth / upstream .pt / .safetensors) [--tag TAG] [--bench_root DIR] [hydra overrides...]"

. "${project_root}/recipes/_common/parse_options.sh"

if [ -z "${checkpoint}" ]; then
    echo "$0: --checkpoint is required" 1>&2
    exit 1
fi
checkpoint="$(realpath "${checkpoint}")"

if [ -f "${project_root}/.env" ]; then
    set -a
    source "${project_root}/.env"
    set +a
fi

cd "${project_root}"

if [ -f "${project_root}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${project_root}/.venv/bin/activate"
fi

export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${tag}" ]; then
    tag="eval-$(date +"%Y%m%d-%H%M%S")"
fi

if command -v mdmsa-parse-run-command >/dev/null 2>&1; then
    cmd="$(mdmsa-parse-run-command)"
else
    cmd="python"
fi

exp_dir="${project_root}/exp/${tag}"

${cmd} "${local_dir}/eval.py" \
    hydra.run.dir="${exp_dir}/logs/$(date +"%Y%m%d-%H%M%S")" \
    hydra.job.chdir=false \
    dataloader="${dataloader}" \
    train="${train}" \
    model="${model}" \
    optimizer="${optimizer}" \
    loss="${loss}" \
    train.tag="${tag}" \
    train.wandb.enabled="${wandb_enabled}" \
    train.checkpoint.resume_from="${checkpoint}" \
    train.output.exp_dir="${exp_dir}" \
    dataloader.db_root="${db_root}" \
    dataloader.bench_root="${bench_root}" \
    "$@"
