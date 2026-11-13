# Changelog

Published tags of `ghcr.io/mvizn/cudaopencv` built from this recipe. The package is private at the time of writing.

## Unreleased: `13.0.3-cudnn9.24-opencv4.14.0-devel-ubuntu24.04-r3` (next build from this repository)
- Same versions and content as r2, from one Dockerfile (no derived step).
- Adds `LABEL org.opencontainers.image.source=https://github.com/mVizn/cudaopencv`.
- dpkg list at `/usr/local/share/cudaopencv.dpkg.txt`; the cmake arch check is derived from `CUDA_ARCH_BIN`/`CUDA_ARCH_PTX`.
- Not built yet.

## `13.0.3-cudnn9.24-opencv4.14.0-devel-ubuntu24.04-r2` (2026-11-13)
- Registry digest / image id `sha256:1d87b61df73cd2f16f3496041522a9b9570ea0cd5bf5fb26a0648f99a44134cd`
  (one linux/amd64 manifest), config `sha256:15f0f3013287fc1e58fb8abde3abbcd4f085b1c60a6623f544356ca2a5d6ba2b`.
- `docker save` archive: 5 338 494 464 bytes, sha256 `77b16f636a0b39b47cddef65fb863e6f83b913cec5186abab385766fd695a9a5`.
- Change: **no entrypoint** (`ENTRYPOINT []`, `CMD ["bash"]`). The NVIDIA base's entrypoint lives in `/opt`; with
  a host `/opt` bind-mounted over it the container did not start. `NV_CUDNN_*` and `com.nvidia.cudnn.version` now
  name the installed cuDNN 9.24.1.1-1.
- Made from the first build by a config-only derived build (the image plus exactly step 5 of the Dockerfile):
  the same 19 layers, only the config differs. This repository's Dockerfile contains step 5, so a full build yields
  the r2 content directly.
- Checked: loads into an empty docker with the same id and layers; starts with a host `/opt` mounted over the
  image's; on an RTX 3060 (driver 580.178.04) the equivalence test against the previous image (CUDA 11.1, cuDNN 8,
  OpenCV 4.6) matched every detection, largest score difference 4.5e-6, identical boxes; smoke test PASS.

## `13.0.3-cudnn9.24-opencv4.14.0-devel-ubuntu24.04` (2026-11-13, first build, counts as r1)
- Image id (index digest) `sha256:62dae3ecd8d0ef45c9a191c7c00d8dad80cf2874185c4150937fb393d8895405`, amd64
  manifest `sha256:dd901a52628f272da29aa64335f01eeedaefe18d14090317843e8260e1eb9b5c`, 19 layers.
- Archive 5 338 499 584 bytes, sha256 `e31ee6b9db5f3eda5a9541832e483e02cc73e4e293028b5cccb947361341fbc3`.
- CUDA 13.0.3, cuDNN 9.24.1.1, OpenCV 4.14.0 + contrib, Python 3.12.3, Ubuntu 24.04.4; GPU arch 75 86 89 120 +
  PTX 120. Built in a VM (4 vCPU, 8 GB, `-j3`) in about 100 min; peak memory 3.3 GB.
- Checked on the CPU backend against the previous image: identical classes and boxes on recorded frames, largest
  score difference 6.5e-6. Kept the NVIDIA entrypoint: superseded by r2. Not published to the registry.
