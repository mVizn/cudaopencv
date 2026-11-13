#!/usr/bin/env python3
"""Generic self-test of a cudaopencv image. Runs INSIDE the image (selftest.sh starts it); needs no model or data.

Checks: CUDA runtime, cuBLAS and cuDNN libraries load (and report their versions); OpenCV is the expected version
and was built with CUDA + cuDNN for the expected GPU architectures and PTX; the Darknet importer, DetectionModel and
the CUDA backend exist; a tiny Darknet network (generated here with fixed random weights: conv, maxpool, conv, yolo)
loads and runs a forward pass and DetectionModel.detect() on the CPU. With --gpu: a CUDA device is visible, the same
network runs on DNN_BACKEND_CUDA, its raw outputs agree with the CPU's, and the time per forward pass is printed.
Exit code 0 = all checks passed.
"""
import argparse, ctypes, glob, json, os, struct, sys, tempfile, time

v = "%d.%d" % sys.version_info[:2]
for p in ("/usr/local/lib/python%s/site-packages" % v, "/usr/local/lib/python%s/dist-packages" % v):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

FAILS = []


def check(ok, what, detail=""):
    print("%s  %s%s" % ("PASS" if ok else "FAIL", what, (": " + str(detail)) if detail != "" else ""), flush=True)
    if not ok:
        FAILS.append(what)
    return ok


def lib_version(names, func):
    for n in names:
        try:
            lib = ctypes.CDLL(n)
        except OSError:
            continue
        if func is None:
            return n, "loaded"
        f = getattr(lib, func)
        if func == "cudnnGetVersion":
            f.restype = ctypes.c_size_t
            return n, f()
        out = ctypes.c_int(0)
        f(ctypes.byref(out))
        return n, out.value
    return None, None


CFG = """[net]
width=64
height=64
channels=3

[convolutional]
filters=16
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
filters=18
size=1
stride=1
pad=0
activation=linear

[yolo]
mask=0,1,2
anchors=8,8, 16,16, 32,32
classes=1
num=3
"""


def write_darknet(d, rng):
    """cfg + weights file in Darknet's format: int32 major, minor, revision, uint64 seen; per convolutional layer
    (no batch norm) float32 biases[filters] then weights[filters * channels * size * size]."""
    import numpy as np
    cfg = os.path.join(d, "tiny.cfg"); wts = os.path.join(d, "tiny.weights")
    open(cfg, "w").write(CFG)
    with open(wts, "wb") as fh:
        fh.write(struct.pack("<iiiQ", 0, 2, 0, 0))
        for n, c, k in ((16, 3, 3), (18, 16, 1)):
            fh.write(rng.uniform(-0.1, 0.1, n).astype("<f4").tobytes())
            fh.write(rng.uniform(-0.3, 0.3, n * c * k * k).astype("<f4").tobytes())
    return cfg, wts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opencv", required=True, help="expected cv2.__version__")
    ap.add_argument("--arch-bin", required=True, help='expected "NVIDIA GPU arch" line, e.g. "75 86 89 120"')
    ap.add_argument("--arch-ptx", default="", help='expected "NVIDIA PTX archs" line, e.g. "120"')
    ap.add_argument("--gpu", action="store_true", help="a GPU is passed in (docker --gpus): test the CUDA backend")
    ap.add_argument("--repeat", type=int, default=50)
    a = ap.parse_args()

    print("python", sys.version.split()[0])
    try:
        print("cuda toolkit", json.load(open("/usr/local/cuda/version.json"))["cuda"]["version"])
    except Exception as e:
        check(False, "CUDA toolkit present (/usr/local/cuda/version.json)", repr(e))
    n, ver = lib_version(["libcudart.so.13", "libcudart.so"], "cudaRuntimeGetVersion")
    check(n is not None, "CUDA runtime library loads", "%s version %s" % (n, ver))
    n, ver = lib_version(["libcublas.so.13", "libcublas.so"], None)
    check(n is not None, "cuBLAS library loads", n)
    n, ver = lib_version(["libcudnn.so.9", "libcudnn.so"], "cudnnGetVersion")
    check(n is not None, "cuDNN library loads", "%s version %s" % (n, ver))

    import numpy as np
    import cv2
    print("numpy", np.__version__, "cv2", cv2.__version__, cv2.__file__)
    check(cv2.__version__ == a.opencv, "OpenCV version", "%s (expected %s)" % (cv2.__version__, a.opencv))
    bi = {}
    for line in cv2.getBuildInformation().split("\n"):
        s = line.strip()
        for k in ("NVIDIA CUDA:", "NVIDIA GPU arch:", "NVIDIA PTX archs:", "cuDNN:", "Baseline:"):
            if s.startswith(k):
                bi[k[:-1]] = s[len(k):].strip()
    print("build info", bi)
    check(bi.get("NVIDIA CUDA", "").startswith("YES"), "OpenCV built with CUDA", bi.get("NVIDIA CUDA"))
    check(bi.get("cuDNN", "").startswith("YES"), "OpenCV built with cuDNN", bi.get("cuDNN"))
    check(bi.get("NVIDIA GPU arch", "") == a.arch_bin, "GPU architectures", bi.get("NVIDIA GPU arch"))
    check(bi.get("NVIDIA PTX archs", "") == a.arch_ptx, "PTX architectures", bi.get("NVIDIA PTX archs"))
    check(hasattr(cv2, "dnn_DetectionModel") and hasattr(cv2.dnn, "readNetFromDarknet")
          and hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and hasattr(cv2, "cuda"),
          "API present (dnn_DetectionModel, readNetFromDarknet, DNN_BACKEND_CUDA, cv2.cuda)")

    rng = np.random.RandomState(1234)
    img = rng.randint(0, 256, (96, 128, 3)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        cfg, wts = write_darknet(d, rng)

        def run(backend, target):
            net = cv2.dnn.readNet(wts, cfg)
            net.setPreferableBackend(backend); net.setPreferableTarget(target)
            net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), swapRB=True, crop=False))
            out = net.forward(net.getUnconnectedOutLayersNames())
            T = time.time()
            for _ in range(a.repeat):
                net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), swapRB=True, crop=False))
                net.forward(net.getUnconnectedOutLayersNames())
            ms = (time.time() - T) * 1000.0 / a.repeat
            model = cv2.dnn_DetectionModel(cv2.dnn.readNet(wts, cfg))
            model.setPreferableBackend(backend); model.setPreferableTarget(target)
            model.setInputParams(size=(64, 64), scale=1 / 255)
            det = model.detect(img, 0.2, 0.4)
            return np.concatenate([o.ravel() for o in out]), ms, det

        try:
            cpu, ms, det = run(cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
            check(cpu.size > 0 and np.isfinite(cpu).all(), "Darknet net, forward pass on CPU",
                  "%d outputs, %.2f ms per pass, detect() -> %d boxes" % (cpu.size, ms, len(det[0])))
        except Exception as e:
            check(False, "Darknet net, forward pass on CPU", repr(e)); cpu = None

        if a.gpu:
            try:
                ndev = cv2.cuda.getCudaEnabledDeviceCount()
            except Exception as e:
                ndev = repr(e)
            if check(isinstance(ndev, int) and ndev >= 1, "CUDA device visible", ndev):
                di = cv2.cuda.DeviceInfo(0)
                print("device 0:", di.name(), "compute capability %d.%d" % (di.majorVersion(), di.minorVersion()),
                      "memory %d MiB" % (di.totalMemory() // 1048576))
                try:
                    gpu, ms, det = run(cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
                    diff = float(np.abs(gpu - cpu).max()) if cpu is not None else float("nan")
                    check(gpu.size == (cpu.size if cpu is not None else gpu.size) and diff < 1e-3,
                          "Darknet net, forward pass on CUDA, agrees with CPU",
                          "max |CUDA - CPU| %.2e, %.2f ms per pass, detect() -> %d boxes" % (diff, ms, len(det[0])))
                except Exception as e:
                    check(False, "Darknet net, forward pass on CUDA", repr(e))
        else:
            print("SKIP  CUDA checks (no --gpu)")

    print("SELFTEST: %s" % ("PASS" if not FAILS else "FAIL (%s)" % "; ".join(FAILS)))
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
