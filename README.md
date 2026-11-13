# cudaopencv

A container image for GPU inference workers: **CUDA 13.0 + cuDNN 9 + OpenCV 4.14 with the CUDA DNN backend**, Python
3.12 on Ubuntu 24.04, one image for **NVIDIA GeForce RTX 20, 30, 40 and 50 series** cards on hosts with the NVIDIA
**580** driver line (or newer). mVizn uses it to run Darknet YOLO models through OpenCV's DNN module
(`cv2.dnn.readNet(weights, cfg)`, `cv2.dnn_DetectionModel`, `DNN_BACKEND_CUDA`) next to Jupyter, with the workers'
own Python environment mounted from the host.

This repository is the recipe: `Dockerfile`, `build.sh`, `selftest.sh`/`selftest.py`, `push_ghcr.py`. It contains no
model, no data and no application code. Images are published as `ghcr.io/mvizn/cudaopencv:<tag>`.

| File | What |
|---|---|
| `Dockerfile` | the image; **every pinned version is in its `ARG` block at the top** |
| `build.sh` | build on a docker host, or in a throw-away Ubuntu 26.04 VM under qemu/KVM (no docker, no root, no change on the host); writes an OCI archive + sha256, a facts file and the self-test result |
| `selftest.sh`, `selftest.py` | generic self-test inside the image: CUDA/cuBLAS/cuDNN load, OpenCV built with CUDA + cuDNN for the expected architectures, a tiny generated Darknet network forward on CPU and (with `--gpu`) on CUDA, no entrypoint |
| `push_ghcr.py` | push an archive's OCI layout to ghcr.io with the `gh` CLI's login (no secret on any command line) |
| `CHANGELOG.md` | what each published tag is |

## What is in the image (current recipe)

| Part | Version | Pinned how |
|---|---|---|
| Base | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` | tag **and** digest `sha256:0230b7f2…47dd` |
| CUDA | 13.0.3 | by the base |
| cuDNN | 9.24.1.1 (replaces the base's 9.14.0.64) | exact package version, NVIDIA's signed apt repository |
| OpenCV + opencv_contrib | 4.14.0 | commits `0654a42e…` and `a8e9acd6…`, fetched by sha and checked |
| Ubuntu packages | as of 2026-11-12 00:00 UTC | Ubuntu snapshot service (`UBUNTU_SNAPSHOT`) |
| Python | 3.12.3 (Ubuntu's), cv2 in `/usr/local/lib/python3.12/site-packages/cv2/` | by Ubuntu 24.04 |
| numpy | Ubuntu's 1.26.4 for the system Python; cv2 compiled against numpy 2.5.4 headers, so it imports with numpy >= 1.25 and 2.x | numpy wheel by version + sha256 (build only) |
| GPU code | native sm_75, sm_86, sm_89, sm_120; PTX compute_120 | `CUDA_ARCH_BIN`, `CUDA_ARCH_PTX` |
| CPU code | baseline SSE3, run-time dispatch up to AVX2/AVX512 | `CPU_BASELINE`, `CPU_DISPATCH` in the cmake line |
| Config | no entrypoint, `CMD ["bash"]`, no user with uid/gid 1000 | Dockerfile steps 2 and 5 |

Not built: `cudacodec` (with CUDA 13.0 OpenCV's FindCUDA finds no driver library at configure time; NVIDIA's Video
Codec SDK is not in the base either). Not set: `CUDA_FAST_MATH`, `-march=native` (results would depend on the build
host or differ numerically).

### Cards

| Series | Compute capability | Code | First CUDA with native support |
|---|---|---|---|
| RTX 20 (Turing) | 7.5 | native sm_75 | 10.0 |
| RTX 30 (Ampere) | 8.6 | native sm_86 | 11.1 |
| RTX 40 (Ada) | 8.9 | native sm_89 | 11.8 |
| RTX 50 (Blackwell) | 12.0 | native sm_120 | 12.8 |
| newer than RTX 50 | > 12.0 | PTX 12.0, JIT-compiled by the driver on first use | — |

PTX only helps cards *newer* than the newest native target; cuDNN and cuBLAS bring their own kernels. CUDA 13 dropped
compute capability below 7.5 (GTX 10 series, Volta and older): such cards are not supported.
Sources: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus),
[CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html),
[Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

### Why these versions (and not newer)

- **CUDA 13.0, not 13.1–13.5**: the hosts run the 580 driver line. Newer `nvidia/cuda` images declare
  `NVIDIA_REQUIRE_CUDA=cuda>=13.x` and the container toolkit refuses to start them on an older driver
  ([toolkit docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html));
  under minor version compatibility, PTX does not work on older drivers
  ([minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)).
  13.0 Update 3's toolkit driver is >= 580.126.20; 13.0 GA needs >= 580.65.06.
- **Ubuntu 24.04, not 26.04**: NVIDIA publishes `ubuntu26.04` images and a 26.04 apt repository only from CUDA 13.3.
- **cuDNN 9.24.1**: the newest cuDNN whose support matrix still says driver >= 580.65.06 for CUDA 13; 9.25 and later
  need >= 615.71.09 ([cuDNN 9.24.1 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.24.1/reference/support-matrix.html)).
- **OpenCV 4.14, not 5.x**: OpenCV 5 removed `readNetFromDarknet()` and its new DNN engine runs on CPU only
  ([4-to-5 migration guide](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration)). cuDNN 9 support is in
  OpenCV since 4.10, CUDA 13.0 support since 4.13 ([change logs](https://github.com/opencv/opencv/wiki/OpenCV-Change-Logs)).

## Tag naming

`<CUDA>-cudnn<cuDNN major.minor>-opencv<OpenCV>-devel-ubuntu<Ubuntu>-r<N>`, for example
`13.0.3-cudnn9.24-opencv4.14.0-devel-ubuntu24.04-r2`.

- Change of any version in the name: a new name, starting at `-r1`.
- Change of the recipe with the same versions (a fix, a label, a package): same name, next `-rN`.
- A published tag is never overwritten. The first build of this recipe was published without a suffix (it counts
  as r1).
- The next build from this repository is **`-r3`** (`ARG IMAGE_NAME` in the Dockerfile): the same content as r2
  plus the `org.opencontainers.image.source` label, the dpkg list renamed to `/usr/local/share/cudaopencv.dpkg.txt`,
  and the arch check in the cmake step derived from the `ARG`s.

## Build

What it needs: network access to Docker Hub, GitHub (OpenCV sources and its cmake downloads from
raw.githubusercontent.com), snapshot.ubuntu.com, developer.download.nvidia.com and PyPI (one numpy wheel).

```
# A. on a docker host (tested in the VM mode with docker 29.1.3 + buildx; mode A itself not run yet)
./build.sh docker out/                      # -> out/cudaopencv-oci.tar (+ .sha256), build.log, image-facts.txt, selftest.txt

# B. no docker on the host: a throw-away VM (qemu/KVM, ssh, xorriso, socat, curl; no root)
VM_CPUS=4 VM_RAM_MB=8192 JOBS=3 ./build.sh vm-all /some/big/work-dir    # not on a RAM-backed /tmp
```

Measured for the first build (4 vCPU, 8 GB VM, `-j3`, VM disk on NFS): apt through the snapshot service ~22 min
(425 packages, the snapshot service is slow), OpenCV configure + compile + install 75 min (1668 ninja steps), export
~1 min: **about 100 min**. Peak guest memory in use **3.3 GB** (nvcc for 4 SASS + 1 PTX target needs ~1–2.5 GB per
job; `JOBS` empty means min(CPUs, RAM in GB / 2.5)). Disk: ~30 GB inside docker during the build (base ~4.3 GB
compressed + ~12 GB unpacked, build cache ~16 GB, result); the archive is ~5.3 GB, the image ~10.7 GB unpacked.

The VM mode also has guards for a shared host: start only with `START_MIN_GB` (18) GB available, pause the VM (QMP
`stop`) under `PAUSE_UNDER_GB` (6) and resume above `RESUME_OVER_GB` (10), stop if free disk falls under
`MIN_FREE_GB` (50). Steps can be run one by one (`vm-up`, `vm-build`, `vm-save`, `vm-facts`, `vm-selftest`,
`vm-down`, `vm-clean`); see the header of `build.sh`.

Reproducibility: the content is pinned (see the table above); layer digests still differ between builds (file
times), so a rebuild is equivalent, not bit-identical. A fixed Ubuntu snapshot also means no later security fixes:
move `UBUNTU_SNAPSHOT` on purpose.

## Test

```
./selftest.sh ghcr.io/mvizn/cudaopencv:<tag>              # CPU checks, any docker host
./selftest.sh ghcr.io/mvizn/cudaopencv:<tag> --gpu        # + CUDA backend on the host's GPU (NVIDIA container toolkit)
./selftest.sh ghcr.io/mvizn/cudaopencv:<tag> --gpu device=1
```
It ends with `SELFTEST: PASS` and checks that the image has no entrypoint and `CMD ["bash"]`. Run `--gpu` on one
card of each generation you deploy to.

This self-test does not prove that a real model gives the same detections as before. For each tag mVizn also runs an
equivalence test with its own models and recorded frames (same frames, same GPU, old image vs new image, every
detection compared). That test and its results are kept in mVizn's internal project repository, because they need
data that is not public.

## Push a new revision, step by step

1. Change the Dockerfile (versions only in the `ARG` block); set `ARG IMAGE_NAME` to the new tag (rule above).
2. `./build.sh docker out/` (or `vm-all`). Check `out/selftest.txt` ends with `SELFTEST: PASS`, and
   `out/image-facts.txt` (versions, `cv2.getBuildInformation()`, arch list).
3. `./selftest.sh <image> --gpu` on real cards; run the internal equivalence test.
4. Unpack and push (needs `gh auth login` with `write:packages`):
   ```
   mkdir layout && tar -xf out/cudaopencv-oci.tar -C layout
   ./push_ghcr.py layout mvizn/cudaopencv <tag>
   ```
   The script prints each blob, the manifest PUT status (201) and the registry's digest; that digest must equal
   the image id printed by `build.sh`.
5. Add the tag, its digest, archive sha256 and what was checked to `CHANGELOG.md`; commit.

The `org.opencontainers.image.source` label links a pushed package to this repository. Pushing never changes a
package's visibility.

## Bumping one component safely

Change one thing at a time, rebuild, run the self-test and the equivalence test, then publish under the next name.

| Component | Where | Watch out |
|---|---|---|
| cuDNN | `CUDNN_PKG_VERSION` | the cuDNN support matrix's minimum driver for CUDA 13 (9.25+ needs 615) |
| OpenCV | `OPENCV_VERSION`, `OPENCV_COMMIT`, `OPENCV_CONTRIB_COMMIT` (commits of the release tags: `gh api repos/opencv/opencv/git/refs/tags/<v>`, then dereference the tag object) | 5.x has no Darknet importer; read the change log for DNN/CUDA changes |
| CUDA / base image | `BASE_IMAGE` (tag + digest from the registry) | the host driver must be at least the toolkit's; cuDNN package names (`libcudnn9-*-cuda-13`) |
| GPU targets | `CUDA_ARCH_BIN`, `CUDA_ARCH_PTX` | each target adds compile time and size; the cmake check follows the `ARG`s |
| Ubuntu packages | `UBUNTU_SNAPSHOT` | a later snapshot brings security fixes and also new package versions |
| numpy (build headers) | `NUMPY_BUILD` (version + sha256 of the cp312 manylinux wheel from PyPI) | the default target API decides the oldest numpy cv2 imports with |

## Proposal (not added): a manual GitHub Actions build

A `workflow_dispatch` workflow could run `build.sh docker` on a GitHub-hosted runner and push with the workflow's
`GITHUB_TOKEN` (`packages: write`). Limits: standard Linux runners have 4 CPUs / 16 GB in public repositories but
**14 GB of SSD** ([GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)),
far less than the ~30 GB the build needs, so the job would have to delete preinstalled software first or use a
larger (paid) runner; there is no GPU (CPU self-test only); and pushing from a workflow publishes without the GPU
and equivalence tests. Not added: decide first.

## Licence

The build recipe and scripts in this repository are under the MIT licence (`LICENSE`). It does not cover the
software in the built image: NVIDIA CUDA and cuDNN (from the `nvidia/cuda` base image) are under NVIDIA's licence
agreements (the base image's terms, the CUDA EULA and the cuDNN software licence agreement), OpenCV and
opencv_contrib under Apache 2.0 (with the non-free contrib algorithms enabled: check their own terms before
commercial use), numpy under BSD, and the Ubuntu packages under their own licences.
