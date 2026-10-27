#! /bin/bash

#SBATCH --job-name=songformer-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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
base_exp_dir="${project_root}/exp"
wandb_enabled=false
iterations=12010

exp_root="exp"
tensorboard_root="tensorboard"

dataloader="songformdb"
train="songformer"
model="songformer"
optimizer="songformer"
loss="songformer"

help_message="Usage: $0 [--tag TAG] [--db_root DIR] [--bench_root DIR] [--base_exp_dir DIR] [--wandb_enabled true|false] [--iterations N] [hydra overrides...]"

. "${project_root}/recipes/_common/parse_options.sh"

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
    tag="$(date +"%Y%m%d-%H%M%S")"
fi

if command -v mdmsa-parse-run-command >/dev/null 2>&1; then
    cmd="$(mdmsa-parse-run-command)"
else
    cmd="python"
fi

exp_dir="${base_exp_dir}/${tag}/${exp_root}"
tensorboard_dir="${base_exp_dir}/${tag}/${tensorboard_root}"

mkdir -p "${exp_dir}" "${tensorboard_dir}"

${cmd} "${local_dir}/train.py" \
    hydra.run.dir="${exp_dir}/logs/$(date +"%Y%m%d-%H%M%S")" \
    hydra.job.chdir=false \
    dataloader="${dataloader}" \
    train="${train}" \
    model="${model}" \
    optimizer="${optimizer}" \
    loss="${loss}" \
    train.tag="${tag}" \
    train.steps.iterations="${iterations}" \
    train.wandb.enabled="${wandb_enabled}" \
    dataloader.db_root="${db_root}" \
    dataloader.bench_root="${bench_root}" \
    train.output.exp_dir="${exp_dir}" \
    train.output.tensorboard_dir="${tensorboard_dir}" \
    "$@"
