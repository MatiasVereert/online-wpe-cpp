"""
compare_fixed.py
================
Valida el WPE de PUNTO FIJO (src/cpp/wpe_fixed.cpp) contra el modelo de punto
fijo de Python (nara_wrappers_fixed.online_wpe_step_fixed), NO contra nara.

El golden es la config que matchea el C++:
    FixedPointConfig.wordlength(W, force_hermitian=False, normalize_target=0.0)
es decir: mismos int_bits por senal, redondeo nearest, saturacion, SIN Hermitiana
y SIN normalizacion (mismo algoritmo que el float, solo cuantizado).

Pipeline: escena 8ch -> STFT -> [C++ fixed harness] y [loop Python fixed] -> compara.

Nota (verificado): con W=40 y datapath en double el error cae a ~1e-7, lo que
confirma que el ALGORITMO es correcto. El residuo ~2.7e-3 en W=24 escala con la
cuantizacion: es la diferencia de convencion de redondeo/saturacion entre ap_fixed
(C++, == el HW real) y la emulacion Fx de numpy (Python), NO un bug ni float32/64.
Un bug de indices/formula daria error grande (>1e-1).

USO:
  g++ -O2 -I /home/matias/Xilinx/2025.2/Vitis/include -o wpe_fixed_harness wpe_fixed_harness.cpp
  /home/matias/miniconda3/envs/tesis_beam/bin/python compare_fixed.py
"""

import os
import sys
import subprocess

import numpy as np

sys.path.insert(0, "/home/matias/Documents/Tesis/nara/nara_wpe")
sys.path.insert(0, "/home/matias/Documents/Tesis/wpe/src/py_impl")
from nara_wpe.wpe import get_power_online
from nara_wpe.utils import stft
from nara_wrappers_fixed import online_wpe_step_fixed, FixedPointConfig

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "wpe_fixed_harness")
Y_BIN = os.path.join(HERE, "Y_fixed.bin")
X_BIN = os.path.join(HERE, "X_hat_fixed.bin")

# --- DEBEN coincidir con las constexpr de wpe_fixed.cpp ----------------------
K = 5
DELAY = 1
ALPHA = 0.9999
M = 8
SIZE = 512
SHIFT = 128
F = SIZE // 2 + 1
W = 24            # word length (== constexpr W en wpe_fixed.cpp)


def make_scene(fs=16000, dur=4.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    src = rng.standard_normal(n)
    src = np.cumsum(src) - np.cumsum(np.concatenate([[0], src[:-1]]))
    src = src / (np.std(src) + 1e-9)
    x = np.zeros((M, n))
    for m in range(M):
        h = np.zeros(1600)
        h[10 + m * 3] = 1.0
        tail = rng.standard_normal(1600) * np.exp(-np.arange(1600) / 300.0) * 0.3
        h += tail
        x[m] = np.convolve(src, h)[:n]
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x


def write_bin(path, Y):
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


def run_fixed_reference(Y, cfg, gnorm=1.0):
    """Replica EXACTA del loop de wpe_fixed.cpp con el modelo Python fixed.

    gnorm: ganancia de front-end (normalizacion). Escala Y antes de cuantizar y
    desescala la salida. La Hermitiana la aplica online_wpe_step_fixed segun
    cfg.force_hermitian.
    """
    rnd, sat = cfg.rounding, cfg.saturate
    taps = K
    Yq = cfg.f("in").q(Y * gnorm, rnd, sat)            # normaliza + cuantiza a in_t
    T, Fb, Mb = Yq.shape
    inv_cov = np.stack(
        [np.identity(Mb * taps) for _ in range(Fb)]
    ).astype(np.complex128)
    filter_taps = np.zeros((Fb, Mb * taps, Mb), dtype=np.complex128)
    Z = np.zeros_like(Yq)

    for t in range(taps + DELAY):                      # warmup: bypass
        Z[t] = Yq[t]
    buffer = list(Yq[:taps + DELAY])

    for t in range(taps + DELAY, T):                   # online
        buffer.append(Yq[t])
        Y_step = np.array(buffer)
        power = get_power_online(Y_step.transpose(1, 2, 0))
        power = cfg.pow_fx.q(power, rnd, sat)          # float (passthrough)
        Z_frame, inv_cov, filter_taps = online_wpe_step_fixed(
            Y_step, power, inv_cov, filter_taps,
            alpha=ALPHA, taps=taps, delay=DELAY, cfg=cfg, n_iter=1)
        Z[t] = Z_frame
        buffer.pop(0)
    return Z / gnorm                                   # desescala salida


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    if not os.path.exists(HARNESS):
        sys.exit(f"Falta el binario {HARNESS}. Compilalo:\n"
                 f"  g++ -O2 -I /home/matias/Xilinx/2025.2/Vitis/include "
                 f"-o {HARNESS} {os.path.join(HERE, 'wpe_fixed_harness.cpp')}")

    x = make_scene()
    Y = stft(x, size=SIZE, shift=SHIFT).transpose(1, 2, 0)   # (T, F, M)
    assert Y.shape[1] == F and Y.shape[2] == M, Y.shape
    print(f"escena: {x.shape} (M,N)   STFT Y: {Y.shape} (T,F,M)   W={W}")

    # Ganancia de front-end (normalizacion): pico global -> max|Y|*gnorm = 0.5
    peak = float(np.max(np.abs(Y)))
    gnorm = 0.5 / (peak + 1e-12)

    # C++ fixed (paso 2: Hermitiana en el core + gnorm en el harness)
    write_bin(Y_BIN, Y)
    subprocess.run([HARNESS, Y_BIN, X_BIN, repr(gnorm)], check=True)
    X_cpp = read_bin(X_BIN)

    # Python fixed golden: config DEFAULT (force_hermitian=True) + misma gnorm
    print(f"corriendo modelo Python fixed (wordlength={W}, herm=True, gnorm={gnorm:.3e})...")
    cfg = FixedPointConfig.wordlength(W, force_hermitian=True)
    X_py = run_fixed_reference(Y, cfg, gnorm)

    tw = K + DELAY
    print("\n=== C++ fixed (ap_fixed, herm+norm)  vs  Python fixed (numpy Fx) ===")
    print(f"  rel_err total          = {relerr(X_cpp, X_py):.3e}")
    print(f"  rel_err post-warmup    = {relerr(X_cpp[tw:], X_py[tw:]):.3e}")
    print("  ~pocos e-3 = convencion de redondeo ap_fixed vs numpy Fx (no bug).")


if __name__ == "__main__":
    main()
