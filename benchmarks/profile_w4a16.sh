#!/usr/bin/env bash
# Collect targeted Nsight Compute reports for the frozen direct W4A16 kernel.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repository_root}/.venv/bin/python}"
ncu_bin="${NCU_BIN:-ncu}"
output_dir="${PROFILE_OUTPUT_DIR:-${repository_root}/profiling-tmp/m4b-w4a16}"
warmups="${PROFILE_WARMUPS:-3}"

sections=(
    SpeedOfLight
    LaunchStats
    Occupancy
    MemoryWorkloadAnalysis
    ComputeWorkloadAnalysis
    InstructionStats
    WarpStateStats
    SchedulerStats
    SourceCounters
)

mkdir -p "${output_dir}"
"${ncu_bin}" --version > "${output_dir}/ncu-version.txt"
"${ncu_bin}" --list-sets > "${output_dir}/ncu-sets.txt"
"${ncu_bin}" --list-sections --csv > "${output_dir}/ncu-sections.csv"

available_sections="$(<"${output_dir}/ncu-sections.csv")"
section_args=()
for section in "${sections[@]}"; do
    if [[ "${available_sections}" != *"\"${section}\""* ]]; then
        echo "required local Nsight section is unavailable: ${section}" >&2
        exit 1
    fi
    section_args+=(--section "${section}")
done

profile_case() {
    local label="$1"
    local m="$2"
    local n="$3"
    local k="$4"
    local report_base="${output_dir}/${label}"
    local report_path="${report_base}.ncu-rep"

    if ! "${ncu_bin}" \
        --force-overwrite \
        --target-processes application-only \
        --clock-control none \
        --cache-control none \
        --kernel-name-base function \
        --kernel-name 'regex:.*w4a16_linear_kernel.*' \
        --launch-skip "${warmups}" \
        --launch-count 1 \
        "${section_args[@]}" \
        --export "${report_base}" \
        "${python_bin}" "${repository_root}/benchmarks/profile_w4a16.py" \
        --m "${m}" --n "${n}" --k "${k}" \
        --warmups "${warmups}" --launches 1 \
        2>&1 | tee "${report_base}-collection.log"; then
        echo "Nsight collection failed; see ${report_base}-collection.log" >&2
        return 1
    fi

    "${ncu_bin}" --import "${report_path}" \
        --page details --print-details all --print-metric-name label-name \
        --print-summary per-kernel > "${report_base}-details.txt"
    "${ncu_bin}" --import "${report_path}" \
        --page raw --csv --print-units base --print-fp \
        > "${report_base}-raw.csv"
    "${ncu_bin}" --import "${report_path}" \
        --page session > "${report_base}-session.txt"
    "${ncu_bin}" --import "${report_path}" \
        --page source --print-source sass --print-units base \
        > "${report_base}-sass.txt"
}

profile_case "canonical-q-m1-n3072-k3072" 1 3072 3072
profile_case "canonical-kv-m1-n768-k3072" 1 768 3072
profile_case "q-m8-n3072-k3072" 8 3072 3072

echo "Nsight Compute reports and text exports written to ${output_dir}"
