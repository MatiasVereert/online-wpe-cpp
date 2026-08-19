"""
sweep_W.py
==========
Barrido de word-length del WPE fijo. Para cada W:
  * recompila wpe_fixed_harness con -DWLEN=W,
  * corre el C++ (ap_fixed) y el modelo Python fixed (numpy Fx) sobre el MISMO Y,
  * mide la degradacion de cada uno respecto del ideal float (W=40), su acuerdo
    mutuo, y el max|X| (indicador de divergencia).

Responde: a partir de cuantos bits se degrada, y si ap_fixed es mas o menos
estable que la emulacion Python.

USO:
  /home/matias/miniconda3/envs/tesis_beam/bin/python sweep_W.py
"""

import os
import subprocess

import numpy as np

import compare_fixed as cf   # reusa make_scene, run_fixed_reference, read/write_bin, stft, cfg

VITIS = "/home/matias/Xilinx/2025.2/Vitis/include"
HARNESS_SRC = os.path.join(cf.HERE, "wpe_fixed_harness.cpp")

W_LIST = [32, 28, 24, 20, 18, 16, 14, 12, 10]


def build(W):
    subprocess.run(
        ["g++", "-O2", f"-DWLEN={W}", "-I", VITIS, "-o", cf.HARNESS, HARNESS_SRC],
        check=True)


def run_cpp(Y, gnorm):
    cf.write_bin(cf.Y_BIN, Y)
    subprocess.run([cf.HARNESS, cf.Y_BIN, cf.X_BIN, repr(gnorm)], check=True)
    return cf.read_bin(cf.X_BIN)


def make_cfg(W):
    # Paso 2: config default (Hermitiana ON); la normalizacion va por gnorm.
    return cf.FixedPointConfig.wordlength(W, force_hermitian=True)


def main():
    x = cf.make_scene()
    Y = cf.stft(x, size=cf.SIZE, shift=cf.SHIFT).transpose(1, 2, 0)
    tw = cf.K + cf.DELAY
    peak = float(np.max(np.abs(Y)))
    gnorm = 0.5 / (peak + 1e-12)
    print(f"escena {x.shape}, Y {Y.shape}, gnorm={gnorm:.3e}  (paso 2: herm+norm)")

    # Ideal float: modelo fixed con W=40 (cuantizacion despreciable).
    X_float = cf.run_fixed_reference(Y, make_cfg(40), gnorm)[tw:]
    ref = np.linalg.norm(X_float) + 1e-12

    hdr = (f"{'W':>3} | {'cpp vs float':>12} | {'py vs float':>12} | "
           f"{'cpp vs py':>10} | {'max|Xcpp|':>10} | {'max|Xpy|':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for W in W_LIST:
        build(W)
        Xc = run_cpp(Y, gnorm)[tw:]
        Xp = cf.run_fixed_reference(Y, make_cfg(W), gnorm)[tw:]
        e_cf = np.linalg.norm(Xc - X_float) / ref
        e_pf = np.linalg.norm(Xp - X_float) / ref
        e_cp = np.linalg.norm(Xc - Xp) / (np.linalg.norm(Xp) + 1e-12)
        print(f"{W:>3} | {e_cf:>12.3e} | {e_pf:>12.3e} | {e_cp:>10.3e} | "
              f"{np.abs(Xc).max():>10.2e} | {np.abs(Xp).max():>10.2e}")

    print("\nLecturas:")
    print("  * 'cpp/py vs float': degradacion respecto del ideal (mas chico = mejor).")
    print("  * 'cpp vs py': acuerdo entre las dos emulaciones fixed.")
    print("  * max|X| que explota (>> el de W alto) = divergencia de la recursion.")
    print("  * quien mantiene error chico hasta W mas bajo = mas estable.")


if __name__ == "__main__":
    main()
