# cudaopencv: CUDA 13 + cuDNN 9 + OpenCV (with the CUDA DNN backend) for GPU inference workers on NVIDIA GeForce
# RTX 20, 30, 40 and 50 series cards, on hosts with the NVIDIA 580 driver line or newer. README.md says why each
# version was chosen and how to bump one; build.sh builds it (on a docker host or in a throw-away VM).
#
#   CUDA 13.0.3 + cuDNN 9.24.1 (the newest pair whose minimum driver is 580), Ubuntu 24.04, Python 3.12,
#   OpenCV 4.14.0 + opencv_contrib 4.14.0 (the newest release that still has the Darknet importer and the classic DNN
#   engine with the CUDA backend), native GPU code for 7.5 8.6 8.9 12.0 + PTX 12.0, CPU baseline SSE3 with run-time
#   dispatch up to AVX512 (one image for every x86-64 CPU).
#
# Reproducible inputs: base by tag and digest; OpenCV by commit (fetched by sha, checked after checkout); Ubuntu
# packages from the Ubuntu snapshot service at a fixed time; cuDNN by exact package version from NVIDIA's signed
# repository; numpy for the build by version and sha256. OpenCV's cmake downloads a few third-party files (IPP ICV,
# ADE, contrib model files) at commits fixed in its own source, each checked by md5 by cmake itself.
# Nothing is installed under /opt (deployments may bind-mount a host /opt over it).
#
# ---- Every pinned version is in the ARG block below; change them only there (README.md, "Bumping a component"). ----

ARG BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:0230b7f243483cb15969fa3cc724a9459599604427052fc2a0d4291c7c0647dd
FROM ${BASE_IMAGE}

# Name recorded in the image's labels; build.sh also tags the image with it (README.md, "Tag naming").
ARG IMAGE_NAME=ghcr.io/mvizn/cudaopencv:13.0.3-cudnn9.24-opencv4.14.0-devel-ubuntu24.04-r3
ARG OPENCV_VERSION=4.14.0
ARG OPENCV_COMMIT=0654a42e19215ef25b1d367d822f3c630447e7c7
ARG OPENCV_CONTRIB_COMMIT=a8e9acd62cabd30419dba83007f2ac0d07de5e2c
ARG CUDA_ARCH_BIN="7.5;8.6;8.9;12.0"
ARG CUDA_ARCH_PTX="12.0"
# Ubuntu snapshot service time stamp (https://snapshot.ubuntu.com): every Ubuntu package as published at that time.
ARG UBUNTU_SNAPSHOT=20261112T000000Z
# cuDNN from NVIDIA's repository; empty keeps the base image's cuDNN (9.14.0.64).
ARG CUDNN_PKG_VERSION=9.24.1.1-1
# numpy used only to compile cv2 (headers); cv2 built against numpy 2 runs on numpy >= 1.25 and on 2.x.
ARG NUMPY_BUILD="numpy==2.5.4 --hash=sha256:fbde6962867ee75b48b0ee29b2b9372ec5d617799dbaf38e82dc0596f2f7738a"
# Parallel compile jobs; empty = min(CPUs, RAM in GB / 2.5) because nvcc for 5 targets needs ~2-3 GB per job.
ARG JOBS=

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1. Ubuntu packages (snapshot) and cuDNN (NVIDIA repo, exact version). The NVIDIA list is kept out of the Ubuntu
#    step so that nothing from it is pulled in by accident; both lists are put back as the base had them.
RUN set -eux; \
    test -f /etc/apt/sources.list.d/ubuntu.sources; mkdir /etc/apt/sources.list.d.off; \
    for f in /etc/apt/sources.list.d/*; do case "$f" in */ubuntu.sources) ;; *) mv "$f" /etc/apt/sources.list.d.off/ ;; esac; done; \
    cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/ubuntu.sources.orig; \
    sed -i "/^Types:/a Snapshot: ${UBUNTU_SNAPSHOT}" /etc/apt/sources.list.d/ubuntu.sources; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config git ca-certificates curl \
        python3 python3-dev python3-venv python3-numpy \
        libgtk-3-dev \
        libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libdc1394-dev libv4l-dev \
        libjpeg-turbo8-dev libpng-dev libtiff-dev libwebp-dev libopenjp2-7-dev libopenexr-dev \
        libopenblas-dev liblapack-dev liblapacke-dev \
        libfreetype-dev libharfbuzz-dev; \
    mv /etc/apt/ubuntu.sources.orig /etc/apt/sources.list.d/ubuntu.sources; \
    for f in /etc/apt/sources.list.d.off/*; do if [ -e "$f" ]; then mv "$f" /etc/apt/sources.list.d/; fi; done; \
    rmdir /etc/apt/sources.list.d.off; \
    if [ -n "${CUDNN_PKG_VERSION}" ]; then \
        apt-get update; \
        apt-mark unhold libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 || true; \
        apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
            "libcudnn9-cuda-13=${CUDNN_PKG_VERSION}" "libcudnn9-dev-cuda-13=${CUDNN_PKG_VERSION}" \
            "libcudnn9-headers-cuda-13=${CUDNN_PKG_VERSION}"; \
        apt-mark hold libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13; \
    fi; \
    rm -rf /var/lib/apt/lists/*; \
    dpkg-query -W -f '${Package}=${Version}\n' | sort > /usr/local/share/cudaopencv.dpkg.txt

# 2. Ubuntu 24.04 images ship a user `ubuntu` with uid/gid 1000. Workloads that create their own uid-1000 user at
#    start (`useradd -u 1000 …`) would fail, as on 18.04/22.04 images they did not. Free uid 1000 / gid 1000.
RUN set -eux; \
    u=$(getent passwd 1000 | cut -d: -f1 || true); if [ -n "$u" ]; then userdel -r "$u" 2>/dev/null || userdel "$u"; fi; \
    g=$(getent group 1000 | cut -d: -f1 || true); if [ -n "$g" ]; then groupdel "$g"; fi; \
    if getent passwd 1000 || getent group 1000; then echo "uid/gid 1000 still taken" >&2; exit 1; fi

# 3. OpenCV + contrib at fixed commits, built with CUDA/cuDNN, installed under /usr/local, sources removed.
RUN set -eux; \
    src=/usr/local/src/opencv-build; mkdir -p "$src"; cd "$src"; \
    fetch() { git init -q "$1"; git -C "$1" fetch -q --depth 1 "https://github.com/opencv/$1.git" "$2"; \
              git -C "$1" -c advice.detachedHead=false checkout -q FETCH_HEAD; \
              test "$(git -C "$1" rev-parse HEAD)" = "$2"; }; \
    fetch opencv "${OPENCV_COMMIT}"; fetch opencv_contrib "${OPENCV_CONTRIB_COMMIT}"; \
    v=$(awk '/define CV_VERSION_(MAJOR|MINOR|REVISION) /{printf "%s%s", sep, $3; sep="."}' opencv/modules/core/include/opencv2/core/version.hpp); \
    test "$v" = "${OPENCV_VERSION}"; \
    python3 -m venv /tmp/bvenv; \
    echo "${NUMPY_BUILD}" > /tmp/numpy-build.txt; \
    /tmp/bvenv/bin/pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/numpy-build.txt; \
    NPINC=$(/tmp/bvenv/bin/python3 -c 'import numpy; print(numpy.get_include())'); \
    PYVER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])'); \
    CUDNN_INC=$(dirname "$(find /usr/include -name cudnn_version.h -print -quit)"); \
    CUDNN_LIB=$(readlink -f /usr/lib/x86_64-linux-gnu/libcudnn.so); \
    cmake -S opencv -B build -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH="$src/opencv_contrib/modules" \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D WITH_CUBLAS=ON -D WITH_CUFFT=ON \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -D BUILD_opencv_cudacodec=OFF \
        -D CUDNN_INCLUDE_DIR="$CUDNN_INC" -D CUDNN_LIBRARY="$CUDNN_LIB" \
        -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" -D CUDA_ARCH_PTX="${CUDA_ARCH_PTX}" \
        -D CUDA_FAST_MATH=OFF -D ENABLE_FAST_MATH=OFF \
        -D CPU_BASELINE=SSE3 -D CPU_DISPATCH="SSE4_1;SSE4_2;AVX;FP16;AVX2;AVX512_SKX" \
        -D WITH_TBB=OFF -D WITH_OPENMP=OFF \
        -D WITH_GTK=ON -D WITH_QT=OFF -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON -D WITH_1394=ON -D WITH_V4L=ON \
        -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=OFF -D BUILD_JAVA=OFF \
        -D PYTHON3_EXECUTABLE=/tmp/bvenv/bin/python3 \
        -D PYTHON3_INCLUDE_DIR="/usr/include/python$PYVER" \
        -D PYTHON3_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython$PYVER.so" \
        -D PYTHON3_NUMPY_INCLUDE_DIRS="$NPINC" \
        -D OPENCV_PYTHON3_INSTALL_PATH="lib/python$PYVER/site-packages" \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF \
        2>&1 | tee /tmp/cmake.log; \
    grep -E "cuDNN:.*YES" /tmp/cmake.log; \
    want_bin=$(echo "${CUDA_ARCH_BIN}" | tr -d . | tr ';' ' '); want_ptx=$(echo "${CUDA_ARCH_PTX}" | tr -d . | tr ';' ' '); \
    grep -E "NVIDIA GPU arch: +${want_bin}\$" /tmp/cmake.log; \
    if [ -n "$want_ptx" ]; then grep -E "NVIDIA PTX archs: +${want_ptx}\$" /tmp/cmake.log; fi; \
    grep -E "To be built:.* dnn .*python3" /tmp/cmake.log; \
    j="${JOBS}"; if [ -z "$j" ]; then \
        c=$(nproc); m=$(awk '/MemTotal/{printf "%d", $2/2600000}' /proc/meminfo); [ "$m" -ge 1 ] || m=1; \
        j=$(( c < m ? c : m )); fi; \
    cmake --build build -j "$j"; \
    cmake --install build; \
    ldconfig; \
    mkdir -p /usr/local/share/opencv4; \
    cp build/CMakeCache.txt /usr/local/share/opencv4/CMakeCache.txt; cp /tmp/cmake.log /usr/local/share/opencv4/cmake-configure.log; \
    cd /; rm -rf "$src" /tmp/bvenv /tmp/numpy-build.txt /tmp/cmake.log /root/.cache

# 4. Self-check with the image's own Python 3 and Ubuntu's numpy (1.26.4): version, Darknet importer, DetectionModel, CUDA backend.
RUN set -eux; \
    PYVER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])'); \
    PYTHONPATH=/usr/local/lib/python$PYVER/site-packages python3 -c 'import cv2, numpy, sys; \
assert cv2.__version__ == sys.argv[1], cv2.__version__; \
assert hasattr(cv2, "dnn_DetectionModel") and hasattr(cv2.dnn, "readNet") and hasattr(cv2.dnn, "readNetFromDarknet"); \
assert cv2.dnn.DNN_BACKEND_CUDA and cv2.dnn.DNN_TARGET_CUDA; \
open("/usr/local/share/opencv4/buildinfo.txt", "w").write(cv2.getBuildInformation()); \
print("cv2", cv2.__version__, "numpy", numpy.__version__, "python", sys.version.split()[0])' "${OPENCV_VERSION}"

LABEL org.opencontainers.image.title="${IMAGE_NAME}" \
      org.opencontainers.image.source="https://github.com/mVizn/cudaopencv" \
      org.opencontainers.image.description="CUDA 13.0.3, cuDNN 9, OpenCV ${OPENCV_VERSION} with the CUDA DNN backend, Python 3.12, Ubuntu 24.04; RTX 20 to 50" \
      org.opencontainers.image.version="${OPENCV_VERSION}" \
      mvizn.opencv.commit="${OPENCV_COMMIT}" \
      mvizn.opencv_contrib.commit="${OPENCV_CONTRIB_COMMIT}" \
      mvizn.cuda_arch_bin="${CUDA_ARCH_BIN}" \
      mvizn.cuda_arch_ptx="${CUDA_ARCH_PTX}" \
      mvizn.ubuntu_snapshot="${UBUNTU_SNAPSHOT}" \
      mvizn.cudnn_package="${CUDNN_PKG_VERSION}"

WORKDIR /

# 5. No entrypoint. The NVIDIA base sets ENTRYPOINT /opt/nvidia/nvidia_entrypoint.sh; a deployment that bind-mounts
#    a host /opt over the image's /opt hides that script and the container does not start ("stat
#    /opt/nvidia/nvidia_entrypoint.sh: no such file or directory"). The script only prints NVIDIA's banner and a
#    driver check, so it is dropped. CMD is set again after ENTRYPOINT because setting ENTRYPOINT clears an inherited
#    CMD. NV_CUDNN_* and the label com.nvidia.cudnn.version are informational (inherited from the base; nothing in
#    the image or the NVIDIA container toolkit reads them) and are set to the cuDNN really installed by step 1.
ENV NV_CUDNN_VERSION=${CUDNN_PKG_VERSION:-9.14.0.64-1} \
    NV_CUDNN_PACKAGE=libcudnn9-cuda-13=${CUDNN_PKG_VERSION:-9.14.0.64-1} \
    NV_CUDNN_PACKAGE_DEV=libcudnn9-dev-cuda-13=${CUDNN_PKG_VERSION:-9.14.0.64-1} \
    NV_CUDNN_PACKAGE_DEV_HEADERS=libcudnn9-headers-cuda-13=${CUDNN_PKG_VERSION:-9.14.0.64-1}
LABEL com.nvidia.cudnn.version="${CUDNN_PKG_VERSION:-9.14.0.64-1}"
ENTRYPOINT []
CMD ["bash"]
