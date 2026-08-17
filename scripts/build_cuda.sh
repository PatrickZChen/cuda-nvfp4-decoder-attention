#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

python_executable="${CUDA_PRIMITIVES_PYTHON:-${repository_root}/.venv/bin/python}"
if [[ -n "${CUDA_PRIMITIVES_CMAKE:-}" ]]; then
    cmake_executable="${CUDA_PRIMITIVES_CMAKE}"
elif [[ -x "${repository_root}/.venv/bin/cmake" ]]; then
    cmake_executable="${repository_root}/.venv/bin/cmake"
else
    cmake_executable="cmake"
fi
build_directory="${CUDA_PRIMITIVES_BUILD_DIR:-${repository_root}/build-cuda}"

torch_cmake_prefix="$("${python_executable}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
pytorch_cuda_version="$("${python_executable}" -c 'import torch; print(torch.version.cuda or "")')"

"${cmake_executable}" \
    -S "${repository_root}" \
    -B "${build_directory}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCUDA_PRIMITIVES_PYTORCH_CUDA_VERSION="${pytorch_cuda_version}" \
    -DCMAKE_PREFIX_PATH="${torch_cmake_prefix}" \
    -DTORCH_CUDA_ARCH_LIST=8.9

"${cmake_executable}" --build "${build_directory}" --config Release --parallel
