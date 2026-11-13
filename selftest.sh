#!/bin/bash
# selftest.sh <image> [--gpu [docker --gpus value, default all]]: run selftest.py inside <image>.
# Without --gpu: CPU checks only (works on any docker host, also without a GPU). With --gpu: also the CUDA backend
# on that card. The expected OpenCV version and architecture lists are read from the Dockerfile's ARG lines (override
# with OPENCV_VERSION, CUDA_ARCH_BIN, CUDA_ARCH_PTX in the environment if the image was built with other build args).
# No network (--network none), the script mounted read-only, nothing else from the host.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
img=${1:?usage: selftest.sh <image> [--gpu [all|device=N]]}; shift
gpus=""
if [ "${1:-}" = --gpu ]; then gpus=${2:-all}; fi
arg() { awk -F= -v k="$1" '$1=="ARG "k{v=$2; gsub(/"/,"",v); print v; exit}' "$HERE/Dockerfile"; }
OPENCV_VERSION=${OPENCV_VERSION:-$(arg OPENCV_VERSION)}
CUDA_ARCH_BIN=${CUDA_ARCH_BIN:-$(arg CUDA_ARCH_BIN)}
CUDA_ARCH_PTX=${CUDA_ARCH_PTX:-$(arg CUDA_ARCH_PTX)}
bin=$(echo "$CUDA_ARCH_BIN" | tr -d . | tr ';' ' ')
ptx=$(echo "$CUDA_ARCH_PTX" | tr -d . | tr ';' ' ')
DOCKER=${DOCKER:-docker}
run=("$DOCKER" run --rm --network none -v "$HERE/selftest.py:/selftest.py:ro")
[ -n "$gpus" ] && run+=(--gpus "$gpus")
echo "+ ${run[*]} $img python3 /selftest.py --opencv $OPENCV_VERSION --arch-bin '$bin' --arch-ptx '$ptx' ${gpus:+--gpu}"
"${run[@]}" "$img" python3 /selftest.py --opencv "$OPENCV_VERSION" --arch-bin "$bin" --arch-ptx "$ptx" ${gpus:+--gpu}
# the image's own config: no entrypoint, CMD bash (a host /opt bind mount must not break the start)
ep=$("$DOCKER" image inspect "$img" --format '{{json .Config.Entrypoint}}')
cmd=$("$DOCKER" image inspect "$img" --format '{{json .Config.Cmd}}')
if [ "$ep" = null ] || [ "$ep" = "[]" ]; then echo "PASS  no entrypoint"; else echo "FAIL  entrypoint $ep"; exit 1; fi
if [ "$cmd" = '["bash"]' ]; then echo "PASS  CMD [\"bash\"]"; else echo "FAIL  CMD $cmd"; exit 1; fi
