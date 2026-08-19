"""
compare_nara_static.py
======================
Valida el WPE funcional de memoria estatica (src/cpp/wpe_float.cpp) contra la
referencia nara_wpe, en el dominio STFT.

Pipeline:
  1. genera una escena reverberante sintetica de M=8 canales (reproducible),
  2. calcula su STFT -> Y (T, F, M),
  3. la pasa por el binario C++ (wpe_float_harness, que maneja wpe_step frame a
     frame con estado static),
  4. la pasa por una re-implementacion en Python de EXACTAMENTE el mismo loop
     online usando nara_wpe.wpe.online_wpe_step (misma init identidad, mismo
     warmup/bypass de los primeros K+delay frames),
  5. reporta  ||C++ - nara|| / ||nara||  en el dominio STFT.

Como el C++ es float32 y el port respeta la matematica de nara (n_iter=1, sin
Hermitiana, sin normalizacion), un port correcto debe dar error relativo ~1e-5
(solo redondeo float32). Un error grande (>1e-2) delata un bug.

USO:
  g++ -O2 -o wpe_float_harness wpe_float_harness.cpp
  /home/matias/miniconda3/envs/tesis_beam/bin/python compare_nara_static.py
"""

import os
import sys
import subprocess

import numpy as np

# nara_wpe (paquete de referencia)
sys.path.insert(0, "/home/matias/Documents/Tesis/nara/nara_wpe")
from nara_wpe.wpe import online_wpe_step, get_power_online
from nara_wpe.utils import stft

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "wpe_float_harness")
Y_BIN = os.path.join(HERE, "Y_static.bin")
X_BIN = os.path.join(HERE, "X_hat_static.bin")

# --- Parametros: DEBEN coincidir con las constexpr de wpe_float.cpp -----------
K = 5          # taps
DELAY = 1
ALPHA = 0.9999
M = 8          # canales
SIZE = 512     # -> F = 257
SHIFT = 128
F = SIZE // 2 + 1


def make_scene(fs=16000, dur=4.0, seed=0):
    """Fuente coloreada convolucionada con IRs cortas por microfono + cola."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    src = rng.standard_normal(n)
    src = np.cumsum(src) - np.cumsum(np.concatenate([[0], src[:-1]]))  # 1/f-ish
    src = src / (np.std(src) + 1e-9)
    x = np.zeros((M, n))
    for m in range(M):
        h = np.zeros(1600)
        h[10 + m * 3] = 1.0                                    # camino directo
        tail = rng.standard_normal(1600) * np.exp(-np.arange(1600) / 300.0) * 0.3
        h += tail                                              # cola reverberante
        x[m] = np.convolve(src, h)[:n]
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x


def write_bin(path, Y):
    """Header int32 T,F,M + Y como complex64 en orden-C [t][f][m]."""
    T = Y.shape[0]
    with open(path, "wb") as f:
        np.array([T, F, M], dtype=np.int32).tofile(f)
        np.ascontiguousarray(Y, dtype=np.complex64).tofile(f)


def read_bin(path):
    with open(path, "rb") as f:
        T, Ff, Mm = np.fromfile(f, dtype=np.int32, count=3)
        n = int(T) * int(Ff) * int(Mm)
        data = np.fromfile(f, dtype=np.complex64, count=n)
    return data.reshape(int(T), int(Ff), int(Mm))


def run_nara_reference(Y):
    """Replica EXACTA del loop de wpe_step (warmup + online) usando nara.

    Y: (T, F, M) -> Z: (T, F, M).
    """
    T, Fb, Mb = Y.shape
    taps = K
    inv_cov = np.stack(
        [np.identity(Mb * taps) for _ in range(Fb)]
    ).astype(np.complex128)
    filter_taps = np.zeros((Fb, Mb * taps, Mb), dtype=np.complex128)
    Z = np.zeros_like(Y)

    # Warmup: primeros taps+delay frames -> bypass (salida = entrada).
    for t in range(taps + DELAY):
        Z[t] = Y[t]
    buffer = list(Y[:taps + DELAY])

    # Online: desde t = taps+delay.
    for t in range(taps + DELAY, T):
        buffer.append(Y[t])
        Y_step = np.array(buffer)                          # (taps+delay+1, F, M)
        power = get_power_online(Y_step.transpose(1, 2, 0))
        Z_frame, inv_cov, filter_taps = online_wpe_step(
            Y_step, power, inv_cov, filter_taps, ALPHA, taps, DELAY)
        Z[t] = Z_frame
        buffer.pop(0)
    return Z


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    if not os.path.exists(HARNESS):
        sys.exit(
            f"No existe el binario {HARNESS}. Compilalo primero:\n"
            f"  g++ -O2 -o {HARNESS} "
            f"{os.path.join(HERE, 'wpe_float_harness.cpp')}")

    x = make_scene()
    Y = stft(x, size=SIZE, shift=SHIFT).transpose(1, 2, 0)   # (T, F, M)
    assert Y.shape[1] == F and Y.shape[2] == M, Y.shape
    print(f"escena: {x.shape} (M,N)   STFT Y: {Y.shape} (T,F,M)")

    # C++
    write_bin(Y_BIN, Y)
    subprocess.run([HARNESS, Y_BIN, X_BIN], check=True)
    X_cpp = read_bin(X_BIN)

    # nara (mismo loop online, mismo warmup)
    print(f"corriendo referencia nara (taps={K}, delay={DELAY}, alpha={ALPHA})...")
    X_nara = run_nara_reference(Y)

    tw = K + DELAY
    print("\n=== C++ (float32, memoria estatica)  vs  nara ===")
    print(f"  ||Y||                  = {np.linalg.norm(Y):.4e}")
    print(f"  rel_err total          = {relerr(X_cpp, X_nara):.3e}")
    print(f"  rel_err post-warmup    = {relerr(X_cpp[tw:], X_nara[tw:]):.3e}")
    print(f"  reduccion de energia   = "
          f"{np.linalg.norm(X_nara[tw:]) / (np.linalg.norm(Y[tw:]) + 1e-12):.3f} "
          f"(||X||/||Y|| post-warmup; <1 = dereverbera)")
    print("  esperado ~1e-5..1e-6 si el port es correcto (solo float32).")


if __name__ == "__main__":
    main()
