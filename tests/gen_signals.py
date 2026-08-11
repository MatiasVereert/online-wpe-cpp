"""
gen_signals.py
==============
Genera la escena reverberante (RIRs reales MIRD, T60=610 ms) igual que
`Vision-Aided-Beamformer/tests/run_wpe_online_mird.py`, calcula la STFT de la
mezcla multicanal y la vuelca a `Y.bin` para que el wrapper en C++
(`test_wpe_online.cpp`) la procese con la clase OnlineWPE.

Tambien guarda `meta.npz` con los parametros de STFT y las senales de entrada y
referencia (early), que usara `eval_output.py` despues del C++.

USO:
    /home/matias/miniconda3/envs/tesis_beam/bin/python gen_signals.py
"""

import os
import sys

import numpy as np
import scipy.signal as signal

# --- Rutas: reutilizamos la infraestructura del repo Vision-Aided-Beamformer -
VISION_REPO = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(VISION_REPO, "src"))

from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array

# --- Configuracion (igual que el script de referencia) -----------------------
FS = 16000
DURATION = 8
ISIR_DB = 0

MIRD_ROOT = os.path.join(VISION_REPO, "tools", "data", "rirs", "mird")
SOURCE_WAV = os.path.join(VISION_REPO, "tools", "data", "signals",
                          "p002_emo_adoration_sentences.wav")

TARGET_T60 = 0.610
SPACING_CFG = "4-4-4-8-4-4-4"
ARRAY_CENTER = np.array([3.0, 3.0, 1.2])

NPERSEG = 512
NOVERLAP = 384
REF_MIC = 0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # wpe/tests


def build_reverberant_scene():
    """Escena de una fuente con RIRs reales MIRD a T60=610 ms."""
    mics = generate_mird_linear_array() + ARRAY_CENTER
    scene = SimAcoustic(mics, array_mismatch=0.0, duration=DURATION, fs=FS, seed=0)

    source_pos = ARRAY_CENTER + np.array([1.0, 0.0, 0.0])
    scene.set_source(SOURCE_WAV, gain=1.0, position=source_pos.reshape(1, 3))

    scene.import_rirs(MirdDatasetProvider(MIRD_ROOT), target_t60=TARGET_T60,
                      array_center=ARRAY_CENTER, spacing_cfg=SPACING_CFG)
    scene.convolve_signals(t_early=0.050)
    data = scene.mix_and_normalize(iSIR_dB=ISIR_DB, inter_normalization=False)
    return data


def write_stft_bin(path, Y_fmt):
    """Vuelca la STFT a binario.

    Y_fmt tiene forma (F, M, T). En disco se escribe el header (T, M, F) como
    tres int32 seguido de los datos complex128 en orden-C (t, m, f), que es el
    layout que espera el driver en C++.
    """
    F, M, T = Y_fmt.shape
    arr = np.ascontiguousarray(Y_fmt.transpose(2, 1, 0)).astype(np.complex128)
    with open(path, "wb") as fo:
        np.array([T, M, F], dtype=np.int32).tofile(fo)
        arr.tofile(fo)


def main():
    print(f"[1/2] Generando escena reverberante MIRD (T60={TARGET_T60*1000:.0f} ms)...")
    data = build_reverberant_scene()
    mic_signals = data["mic_signals"]      # (M, N)
    target_early = data["target_early"]    # (M, N)
    print(f"      mezcla: {mic_signals.shape}  (M canales x N muestras)")

    # SciPy devuelve (M, F, T); el WPE espera (F, M, T).
    _, _, Zxx = signal.stft(mic_signals, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Y_in = Zxx.transpose(1, 0, 2)

    y_bin = os.path.join(OUT_DIR, "Y.bin")
    write_stft_bin(y_bin, Y_in)
    np.savez(os.path.join(OUT_DIR, "meta.npz"),
             fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, ref_mic=REF_MIC,
             mic_signals=mic_signals, target_early=target_early)

    print(f"[2/2] {os.path.basename(y_bin)} escrito. Y_in (F,M,T)={Y_in.shape}")
    print("      meta.npz escrito (fs, params STFT, senales de entrada y referencia)")


if __name__ == "__main__":
    main()
