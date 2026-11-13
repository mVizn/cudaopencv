#!/bin/bash
# build.sh: build the cudaopencv image (Dockerfile in this folder), save it as an OCI archive, write a facts file
# and run the self-test (selftest.sh, CPU only unless a GPU is there).
#
# Two ways, same Dockerfile, same result:
#   A. On a host that has docker:
#        build.sh docker <out-dir> [docker build args ...]
#      -> <out-dir>/cudaopencv-oci.tar (+ .sha256), build.log, image-facts.txt, selftest.txt. Nothing is pushed anywhere.
#   B. On a host with only qemu/KVM (no docker, no root, no host changes), in a throw-away Ubuntu 26.04 cloud VM
#      (the VM pulls the base image and the sources itself; the host only runs qemu and ssh):
#        build.sh vm-up    <work-dir>            # cloud image + seed + VM, waits for docker.io in the VM
#        build.sh vm-build <work-dir> [args...]  # copies the Dockerfile in, docker build in the VM, watched from the host
#        build.sh vm-wait  <work-dir>            # (re)attach to a running VM build and wait for it
#        build.sh vm-save  <work-dir>            # docker save in the VM, streamed to <work-dir>/cudaopencv-oci.tar
#        build.sh vm-facts <work-dir>            # digest, size, Python, cv2.getBuildInformation() -> <work-dir>/facts/
#        build.sh vm-selftest <work-dir>         # selftest.sh in the VM (CPU checks) -> <work-dir>/facts/selftest.txt
#        build.sh vm-down  <work-dir>            # power the VM off
#        build.sh vm-clean <work-dir>            # delete the VM disk and the cloud image (keeps archive, logs, facts)
#        build.sh vm-all   <work-dir>            # vm-up, vm-build, vm-save, vm-facts, vm-selftest, vm-down, vm-clean
#
# Environment (B): VM_CPUS (4), VM_RAM_MB (8192), VM_DISK (200G), PORT (2245), VM_CACHE (writeback; "unsafe" is
# acceptable for a build VM whose disk is thrown away), JOBS (3: parallel compile jobs in the guest, passed as
# --build-arg JOBS), host memory rules for a shared host: START_MIN_GB (18: vm-up starts the VM only when the host's
# MemAvailable is at least that; WAIT_MIN (0) minutes of waiting, checked every WAIT_STEP_MIN (10) minutes),
# PAUSE_UNDER_GB (6: the VM is paused with QMP `stop` when MemAvailable falls under it) and RESUME_OVER_GB (10: QMP
# `cont` when it is back above), MIN_FREE_GB (50: the build is stopped and the VM disk deleted if free space where
# <work-dir> lives falls under it). QMP_DIR (the QMP socket's folder; default $XDG_RUNTIME_DIR or /tmp, not NFS).
# Image name: IMAGE_NAME (default from the Dockerfile's ARG). The VM pulls the base image from Docker Hub anonymously.
# Builds run with BUILDX_NO_DEFAULT_ATTESTATIONS=1, so the result is one linux/amd64 manifest (no attestation index).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE_NAME=${IMAGE_NAME:-$(awk -F= '/^ARG IMAGE_NAME=/{print $2; exit}' "$HERE/Dockerfile")}
CLOUD=https://cloud-images.ubuntu.com/releases/resolute/release
CIMG=ubuntu-26.04-server-cloudimg-amd64.img
PORT=${PORT:-2245}
VM_CPUS=${VM_CPUS:-4}
VM_RAM_MB=${VM_RAM_MB:-8192}
VM_DISK=${VM_DISK:-200G}
VM_CACHE=${VM_CACHE:-writeback}
START_MIN_GB=${START_MIN_GB:-18}
PAUSE_UNDER_GB=${PAUSE_UNDER_GB:-6}
RESUME_OVER_GB=${RESUME_OVER_GB:-10}
MIN_FREE_GB=${MIN_FREE_GB:-50}
JOBS=${JOBS:-3}
QMP_DIR=${QMP_DIR:-${XDG_RUNTIME_DIR:-/tmp}}
QMP="$QMP_DIR/cudaopencv-build.qmp"
OCI=cudaopencv-oci.tar

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }
vssh() { local w=$1; shift
  ssh -i "$w/vmkey" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR -o ConnectTimeout=5 -o ServerAliveInterval=30 ubuntu@127.0.0.1 "$@"; }
vscp() { local w=$1; shift
  scp -q -i "$w/vmkey" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$@"; }

# ---------- facts about a built image (runs where docker is: host for A, VM for B) ----------
image_facts() {
  local name=$1
  docker image inspect "$name" --format 'Id={{.Id}}
RepoTags={{json .RepoTags}}
Created={{.Created}}
Size={{.Size}}
Os/Arch={{.Os}}/{{.Architecture}}
Layers={{len .RootFS.Layers}}
Labels={{json .Config.Labels}}'
  docker run --rm --network none --entrypoint /bin/bash "$name" -c '
    . /etc/os-release; echo "os: $PRETTY_NAME"
    echo "python3: $(python3 --version 2>&1)"
    echo "nvcc: $(/usr/local/cuda/bin/nvcc --version | tail -n 1)"
    echo "cudnn: $(dpkg-query -W -f "\${Version}" libcudnn9-cuda-13)"
    echo "uid1000: $(getent passwd 1000 || echo none)"
    PYVER=$(python3 -c "import sys;print(\"%d.%d\"%sys.version_info[:2])")
    PYTHONPATH=/usr/local/lib/python$PYVER/site-packages python3 -c "import cv2,numpy;print(\"cv2\",cv2.__version__,\"numpy\",numpy.__version__,\"cv2 file\",cv2.__file__)"
    echo "---- cv2.getBuildInformation()"; cat /usr/local/share/opencv4/buildinfo.txt'
}

save_and_hash() {   # docker save to a file, then hash it
  local name=$1 out=$2
  docker save "$name" -o "$out.part"; sync "$out.part"; mv "$out.part" "$out"
  (cd "$(dirname "$out")" && sha256sum "$(basename "$out")" | tee "$(basename "$out").sha256")
  ls -l "$out"
}

cmd_docker() {
  local out=$1; shift
  mkdir -p "$out"
  log "docker build -t $IMAGE_NAME $HERE $*"
  BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker build --progress=plain -t "$IMAGE_NAME" --build-arg "IMAGE_NAME=$IMAGE_NAME" "$@" "$HERE" 2>&1 | tee "$out/build.log"
  image_facts "$IMAGE_NAME" | tee "$out/image-facts.txt"
  save_and_hash "$IMAGE_NAME" "$out/$OCI"
  "$HERE/selftest.sh" "$IMAGE_NAME" 2>&1 | tee "$out/selftest.txt"   # CPU checks; with a GPU: selftest.sh <image> --gpu
  log "image digest (Id): $(docker image inspect "$IMAGE_NAME" --format '{{.Id}}')"
}

# ---------- B: helper VM ----------
host_avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
ram_ok() {      # enough host RAM to start the VM?
  local avail; avail=$(host_avail_gb)
  [ "$avail" -ge "$START_MIN_GB" ] || { echo "host MemAvailable ${avail} GB < ${START_MIN_GB} GB needed to start the VM"; return 1; }
}
qmp() { printf '{"execute":"qmp_capabilities"}\n{"execute":"%s"}\n' "$1" | socat - "UNIX-CONNECT:$QMP" >/dev/null; }

vm_up() {
  local w=$1 why waited=0
  # WAIT_MIN (default 0): wait up to that many minutes for enough host RAM (other VMs on a shared host come and go)
  while ! why=$(ram_ok); do
    [ "$waited" -lt "${WAIT_MIN:-0}" ] || die "$why"
    log "waiting: $why"; sleep $(( ${WAIT_STEP_MIN:-10} * 60 )); waited=$((waited + ${WAIT_STEP_MIN:-10}))
  done
  mkdir -p "$w/dl" "$w/seed" "$w/logs"
  if [ ! -f "$w/dl/$CIMG" ]; then
    (cd "$w/dl" && curl -sSfLO "$CLOUD/SHA256SUMS" && curl -sSfL -o "$CIMG.part" "$CLOUD/$CIMG" && mv "$CIMG.part" "$CIMG" \
       && sha256sum -c --ignore-missing SHA256SUMS) || { rm -f "$w/dl/$CIMG"; die "cloud image download/check failed"; }
  fi
  [ -f "$w/vmkey" ] || ssh-keygen -q -t ed25519 -N '' -C helper-vm -f "$w/vmkey"
  [ -f "$w/vm.qcow2" ] || qemu-img create -q -f qcow2 -F qcow2 -b "dl/$CIMG" "$w/vm.qcow2" "$VM_DISK"
  printf 'instance-id: cudaopencv-build\nlocal-hostname: cudaopencv-build\n' > "$w/seed/meta-data"
  cat > "$w/seed/user-data" <<EOF
#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "$w/vmkey.pub")
ssh_pwauth: false
package_update: true
packages: [docker.io, docker-buildx, rsync]
growpart: {mode: auto, devices: ['/']}
EOF
  xorriso -as mkisofs -quiet -V cidata -J -r -o "$w/seed.iso" "$w/seed/user-data" "$w/seed/meta-data"
  qemu-system-x86_64 -name cudaopencv-build -enable-kvm -cpu host -smp "$VM_CPUS" -m "$VM_RAM_MB" \
    -drive "file=$w/vm.qcow2,format=qcow2,if=virtio,discard=unmap,cache=$VM_CACHE,file.locking=off" \
    -drive "file=$w/seed.iso,format=raw,if=virtio,readonly=on,file.locking=off" \
    -netdev "user,id=n0,hostfwd=tcp:127.0.0.1:$PORT-:22" -device virtio-net-pci,netdev=n0 \
    -qmp "unix:$QMP,server=on,wait=off" \
    -display none -serial "file:$w/logs/serial.log" -pidfile "$w/qemu.pid" -daemonize
  local i; for i in $(seq 120); do vssh "$w" true 2>/dev/null && break; sleep 5; done
  vssh "$w" 'cloud-init status --wait >/dev/null; cloud-init status; docker --version; nproc; free -g | head -2; df -h /' \
    | tee "$w/logs/vm-up.txt"
}

host_guard() {   # 0 = fine; 1 = stop (prints why): free space where the work dir lives
  local w=$1 free
  free=$(df -P -BG "$w" | awk 'NR==2{gsub("G","",$4); print $4}')
  if [ "$free" -lt "$MIN_FREE_GB" ]; then echo "free space at $w ${free} GB < ${MIN_FREE_GB} GB"; return 1; fi
  return 0
}

BLOG=/var/tmp/cudaopencv-build.log   # in the VM, written by root
BRC=/var/tmp/cudaopencv-build.rc

vm_build() {   # start the build in the VM (detached), then wait for it
  local w=$1; shift
  local args="" a; for a in "$@"; do args+=" $(printf '%q' "$a")"; done
  vssh "$w" 'rm -rf ~/ctx && mkdir -p ~/ctx'
  vscp "$w" "$HERE/Dockerfile" ubuntu@127.0.0.1:ctx/
  local inner="cd /home/ubuntu/ctx && BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker build --progress=plain -t $(printf '%q' "$IMAGE_NAME") \
--build-arg IMAGE_NAME=$(printf '%q' "$IMAGE_NAME") --build-arg JOBS=$JOBS$args . > $BLOG 2>&1; echo \$? > $BRC"
  vssh "$w" "sudo rm -f $BLOG $BRC; sudo setsid nohup bash -o pipefail -c $(printf '%q' "$inner") > /dev/null 2>&1 < /dev/null &"
  log "build started in the VM"
  vm_wait "$w"
}

vm_wait() {    # poll the VM build every minute; copy its log; pause/resume on host RAM; stop on disk space
  local w=$1 why paused=0 avail used peak=0
  mkdir -p "$w/logs"
  while :; do
    avail=$(host_avail_gb)
    if [ "$paused" = 1 ]; then
      if [ "$avail" -gt "$RESUME_OVER_GB" ]; then qmp cont && paused=0 && log "RESUMED: host MemAvailable ${avail} GB"; fi
      sleep 30; continue
    fi
    if [ "$avail" -lt "$PAUSE_UNDER_GB" ]; then
      qmp stop && paused=1 && log "PAUSED (QMP stop): host MemAvailable ${avail} GB < ${PAUSE_UNDER_GB} GB"; continue
    fi
    vssh "$w" "sudo cat $BLOG" > "$w/logs/build.log" 2>/dev/null || true
    used=$(vssh "$w" "free -m | awk '/^Mem:/{print \$3}'" 2>/dev/null || echo 0)
    [ -n "$used" ] && [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used
    echo "$(date '+%F %T') guest_used_mb=$used peak_mb=$peak host_avail_gb=$avail" >> "$w/logs/mem.log"
    if vssh "$w" "sudo test -f $BRC" 2>/dev/null; then break; fi
    if ! why=$(host_guard "$w"); then
      log "STOP: $why; killing the VM and deleting its disk"
      kill "$(cat "$w/qemu.pid")" 2>/dev/null || true; sleep 5; rm -f "$w/vm.qcow2"
      die "build stopped by guard: $why"
    fi
    log "building (guest RAM used ${used} MB, peak ${peak} MB, host avail ${avail} GB): $(tail -n 1 "$w/logs/build.log" | cut -c1-120)"
    sleep 60
  done
  local rc; rc=$(vssh "$w" "sudo cat $BRC"); vssh "$w" "sudo cat $BLOG" > "$w/logs/build.log"
  log "guest RAM peak (used, sampled every minute): ${peak} MB"
  [ "$rc" = 0 ] || die "docker build failed (rc $rc); see $w/logs/build.log"
  log "build finished OK"
}

vm_save() {
  local w=$1 out="$1/$OCI"
  # stream the archive out of the VM; written with a normal (synced) write on the host, then hashed on both sides
  vssh "$w" "sudo docker save $(printf '%q' "$IMAGE_NAME")" > "$out.part"
  sync "$out.part"; mv "$out.part" "$out"
  local hsum vsum
  hsum=$(sha256sum "$out" | cut -d' ' -f1)
  vsum=$(vssh "$w" "sudo docker save $(printf '%q' "$IMAGE_NAME") | sha256sum" | cut -d' ' -f1)
  echo "$hsum  $OCI" > "$out.sha256"
  log "archive $out $(stat -c %s "$out") bytes sha256 $hsum (re-saved in VM: $vsum)"
  [ "$hsum" = "$vsum" ] || log "WARNING: docker save is not byte-stable here or the copy differs; check"
  tar -tf "$out" | grep -E '^(index.json|manifest.json|oci-layout)$' | tr '\n' ' '; echo
}

vm_facts() {
  local w=$1; mkdir -p "$w/facts"
  { declare -f image_facts; printf 'image_facts %q\n' "$IMAGE_NAME"; } | vssh "$w" sudo bash -euo pipefail -s \
    | tee "$w/facts/image-facts.txt"
  vssh "$w" "sudo docker image inspect $(printf '%q' "$IMAGE_NAME") --format '{{.Id}}'" > "$w/facts/image-id.txt"
  vssh "$w" "sudo docker run --rm --network none --entrypoint cat $(printf '%q' "$IMAGE_NAME") /usr/local/share/cudaopencv.dpkg.txt" \
    > "$w/facts/dpkg.txt"
  vssh "$w" "sudo docker run --rm --network none --entrypoint cat $(printf '%q' "$IMAGE_NAME") /usr/local/share/opencv4/CMakeCache.txt" \
    > "$w/facts/CMakeCache.txt"
  log "image id: $(cat "$w/facts/image-id.txt")"
}

vm_selftest() {  # CPU self-test in the VM (it has no GPU) -> <work-dir>/facts/selftest.txt
  local w=$1; mkdir -p "$w/facts"
  vssh "$w" 'mkdir -p ~/ctx'
  vscp "$w" "$HERE/Dockerfile" "$HERE/selftest.py" "$HERE/selftest.sh" ubuntu@127.0.0.1:ctx/
  vssh "$w" "sudo bash ctx/selftest.sh $(printf '%q' "$IMAGE_NAME")" 2>&1 | tee "$w/facts/selftest.txt"
}

vm_down() { vssh "$1" sudo poweroff || true; sleep 10; }
vm_clean() {
  local w=$1
  if [ -f "$w/qemu.pid" ] && kill -0 "$(cat "$w/qemu.pid")" 2>/dev/null; then die "VM still running (vm-down first)"; fi
  rm -f "$w/vm.qcow2" "$w/dl/$CIMG" "$w/seed.iso"; ls -la "$w"
}

cmd=${1:-}; shift || true
case "$cmd" in
  docker)   cmd_docker "$@" ;;
  vm-up)    vm_up "$1" ;;
  vm-build) vm_build "$@" ;;
  vm-wait)  vm_wait "$1" ;;
  vm-save)  vm_save "$1" ;;
  vm-facts) vm_facts "$1" ;;
  vm-selftest) vm_selftest "$1" ;;
  vm-down)  vm_down "$1" ;;
  vm-clean) vm_clean "$1" ;;
  vm-all)   w=$1; shift; vm_up "$w"; vm_build "$w" "$@"; vm_save "$w"; vm_facts "$w"; vm_selftest "$w"; vm_down "$w"; vm_clean "$w" ;;
  *) sed -n '2,31p' "$0"; exit 2 ;;
esac
