#! /bin/bash

#PBS -l rt_G.small=1
#PBS -l walltime=2:00:00
#PBS -j oe

set -eu
set -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

tag=""
data_dir="${project_root}/data"
base_exp_dir="${project_root}/exp"
wandb_enabled=false
epochs=5

exp_root="exp"
tensorboard_root="tensorboard"

dataloader="mnist"
train="classification"
model="cnn"
optimizer="adam"
loss="cross_entropy"

help_message="Usage: $0 [--tag TAG] [--data_dir DIR] [--base_exp_dir DIR] [--wandb_enabled true|false] [--epochs N] [hydra overrides...]"

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

if command -v exptemplete-parse-run-command >/dev/null 2>&1; then
    cmd="$(exptemplete-parse-run-command)"
else
    cmd="python"
fi

exp_dir="${base_exp_dir}/${tag}/${exp_root}"
tensorboard_dir="${base_exp_dir}/${tag}/${tensorboard_root}"

mkdir -p "${exp_dir}" "${tensorboard_dir}"

${cmd} "${script_dir}/local/train.py" \
    hydra.run.dir="${exp_dir}/logs/$(date +"%Y%m%d-%H%M%S")" \
    hydra.job.chdir=false \
    dataloader="${dataloader}" \
    train="${train}" \
    model="${model}" \
    optimizer="${optimizer}" \
    loss="${loss}" \
    train.tag="${tag}" \
    train.steps.epochs="${epochs}" \
    train.wandb.enabled="${wandb_enabled}" \
    dataloader.train.dataset.root="${data_dir}" \
    dataloader.validate.dataset.root="${data_dir}" \
    dataloader.test.dataset.root="${data_dir}" \
    train.output.exp_dir="${exp_dir}" \
    train.output.tensorboard_dir="${tensorboard_dir}" \
    "$@"
