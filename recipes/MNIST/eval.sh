#! /bin/bash

#PBS -l rt_G.small=1
#PBS -l walltime=1:00:00
#PBS -j oe

set -eu
set -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

tag=""
data_dir="${project_root}/data"
checkpoint=""
wandb_enabled=false

dataloader="mnist"
train="classification"
model="cnn"
optimizer="adam"
loss="cross_entropy"

help_message="Usage: $0 --checkpoint PATH [--tag TAG] [--data_dir DIR] [hydra overrides...]"

. "${project_root}/recipes/_common/parse_options.sh"

if [ -z "${checkpoint}" ]; then
    echo "$0: --checkpoint is required" 1>&2
    exit 1
fi

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

if command -v exptemplete-parse-run-command >/dev/null 2>&1; then
    cmd="$(exptemplete-parse-run-command)"
else
    cmd="python"
fi

${cmd} "${script_dir}/local/eval.py" \
    hydra.run.dir="${project_root}/exp/${tag}/logs/$(date +"%Y%m%d-%H%M%S")" \
    hydra.job.chdir=false \
    dataloader="${dataloader}" \
    train="${train}" \
    model="${model}" \
    optimizer="${optimizer}" \
    loss="${loss}" \
    train.tag="${tag}" \
    train.wandb.enabled="${wandb_enabled}" \
    train.checkpoint.resume_from="${checkpoint}" \
    dataloader.train.dataset.root="${data_dir}" \
    dataloader.validate.dataset.root="${data_dir}" \
    dataloader.test.dataset.root="${data_dir}" \
    "$@"
