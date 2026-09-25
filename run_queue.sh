#!/usr/bin/env bash
# A fixed batch of foreground commands, dispatched by GNU Parallel.
# Ubuntu/Debian dependency: sudo apt install parallel
#
# Usage: bash run_queue.sh JOBS_FILE [CONCURRENT_JOBS=30] [STATE_DIR=queue_state]
#                         [THREADS_PER_JOB=1]
# Each nonempty input line is a complete Bash command, for example:
#   cd /absolute/path/run_001 && python -u newsubmittable.py
# For multiline logic, put it in a separate script and list its invocation.
# Do not use tmux, nohup, or trailing '&' inside a job: it must stay foreground.
# Each simulation needs its own output/checkpoint paths.
#
# Run from your simulation environment (activate your venv/conda environment
# first), inside one tmux session. Detach with Ctrl-b, then d.
# Repeating the SAME invocation skips jobs recorded as successful and retries
# failed/unrecorded jobs. This relaunches commands; checkpoint loading remains
# the simulation's responsibility. Keep job commands, code, and inputs fixed
# for a batch. Use a NEW state directory for a new batch.
#
# The concurrency limit applies only to this queue. It does not pin cores or
# enforce memory limits, and the environment variables do not constrain code
# that explicitly creates additional threads/processes. Budget peak RAM/job.
# This is a fixed batch, not a daemon that watches for newly submitted jobs.
#
# References:
# https://manpages.debian.org/bookworm/parallel/parallel.1.en.html

set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash run_queue.sh JOBS_FILE [CONCURRENT_JOBS=30] [STATE_DIR=queue_state] [THREADS_PER_JOB=1]' \
        'Example: bash run_queue.sh jobs.txt 30 queue_state 1' \
        'Run again with the same job list and state directory to retry/resume.'
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# < 1 || $# > 4 )); then
    usage >&2
    exit 2
fi

commands_file=$1
concurrency=${2:-30}
state_dir=${3:-queue_state}
threads_per_job=${4:-1}

[[ $concurrency =~ ^[1-9][0-9]*$ ]] || die 'CONCURRENT_JOBS must be a positive integer.'
[[ $threads_per_job =~ ^[1-9][0-9]*$ ]] || die 'THREADS_PER_JOB must be a positive integer.'
[[ -f $commands_file && -r $commands_file && -s $commands_file ]] || die 'JOBS_FILE must be a readable, nonempty file.'
command -v parallel >/dev/null || die 'Install GNU Parallel first: sudo apt install parallel'
command -v flock >/dev/null || die 'flock is required (Ubuntu/Debian package: util-linux).'
parallel_version=$(parallel --version)
[[ $parallel_version == 'GNU parallel '* ]] || die 'The parallel command must be GNU Parallel.'

commands_file=$(realpath -- "$commands_file")
mkdir -p -- "$state_dir"
state_dir=$(realpath -- "$state_dir")
working_dir=$(pwd -P)

# The lock remains held across exec and is released when the runner exits.
exec 9>"$state_dir/runner.lock"
flock -n 9 || die 'This queue is already running. Reattach to its tmux session.'

if [[ -f $state_dir/commands.txt ]]; then
    cmp -s -- "$commands_file" "$state_dir/commands.txt" ||
        die 'The job list changed. Restore the original or use a new state directory.'
    [[ -f $state_dir/working_directory.txt ]] || die 'Queue metadata is incomplete; check this state directory.'
    [[ $(cat -- "$state_dir/working_directory.txt") == "$working_dir" ]] ||
        die 'Restart from the original working directory, or use a new state directory.'
else
    [[ ! -e $state_dir/joblog.tsv ]] || die 'Existing job log has no saved job list; check this state directory.'
    printf '%s\n' "$working_dir" > "$state_dir/working_directory.txt"
    snapshot_temp=$(mktemp "$state_dir/commands.XXXXXX")
    cp -- "$commands_file" "$snapshot_temp"
    mv -- "$snapshot_temp" "$state_dir/commands.txt"
fi

# Prevent supported numerical libraries from choosing large thread pools.
# Applications can override these settings; verify application-specific pools.
export OMP_NUM_THREADS="$threads_per_job"
export MKL_NUM_THREADS="$threads_per_job"
export OPENBLAS_NUM_THREADS="$threads_per_job"
export BLIS_NUM_THREADS="$threads_per_job"
export VECLIB_MAXIMUM_THREADS="$threads_per_job"
export NUMEXPR_NUM_THREADS="$threads_per_job"
export JULIA_NUM_THREADS="$threads_per_job"
export PYTHONUNBUFFERED=1
export PARALLEL_SHELL=/bin/bash

# Separate attempts preserve the output from jobs that failed previously.
attempt_dir=$(mktemp -d "$state_dir/attempt-$(date +%Y%m%d-%H%M%S)-XXXXXX")
printf 'Concurrent jobs: %s; threads per job: %s\n' "$concurrency" "$threads_per_job"
printf 'Job status: %s/joblog.tsv\n' "$state_dir"
printf 'Output: %s/job_<number>/{stdout,stderr}\n' "$attempt_dir"

progress_options=()
if [[ -t 2 ]]; then
    progress_options=(--eta)
fi

# Input lines are full commands: no command template is necessary.
# --plain prevents a user profile from silently changing dispatch behavior.
# Failures are logged; remaining jobs continue. Skip only recorded successes.
exec parallel --plain \
    --jobs "$concurrency" \
    --joblog "$state_dir/joblog.tsv" \
    --results "$attempt_dir/job_{#}/" \
    --resume-failed \
    --halt never \
    "${progress_options[@]}" \
    < "$state_dir/commands.txt"
