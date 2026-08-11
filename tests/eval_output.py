"""
eval_output.py
==============
Lee `X_hat.bin` (salida del wrapper en C++), hace la iSTFT y guarda tres WAVs
mono con escala compartida para comparacion auditiva:

    1_input_reverberant.wav     mezcla reverberante (entrada)
    2_wpe_cpp_output.wav        salida del OnlineWPE en C++
    3_target_early_reference.wav  referencia "ideal" (early)

Reutiliza `meta.npz` generado por `gen_signals.py`.

USO:
    /home/matias/miniconda3/envs/tesis_beam/bin/python eval_output.py
"""

import os

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # wpe/tests


def read_stft_bin(path):
    """Lee el binario (header T,M,F + complex128 orden-C) y devuelve (F, M, T)."""
    with open(path, "rb") as fi:
        T, M, F = np.fromfile(fi, dtype=np.int32, count=3)
        n = int(T) * int(M) * int(F)
        data = np.fromfile(fi, dtype=np.complex128, count=n)
    arr = data.reshape(int(T), int(M), int(F))  # (T, M, F)
    return arr.transpose(2, 1, 0)               # (F, M, T)


def save_wav_shared_scale(path, fs, sig, scale):
    """Guarda un WAV mono aplicando una escala compartida (comparacion justa)."""
    x = np.clip(np.real(sig) * scale, -1.0, 1.0)
    wavfile.write(path, fs, (x * 32767).astype(np.int16))
    print(f"  -> {os.path.basename(path)}")


def main():
    meta = np.load(os.path.join(OUT_DIR, "meta.npz"))
    fs = int(meta["fs"])
    nperseg = int(meta["nperseg"])
    noverlap = int(meta["noverlap"])
    ref = int(meta["ref_mic"])
    mic_signals = meta["mic_signals"]      # (M, N)
    target_early = meta["target_early"]    # (M, N)

    X_hat = read_stft_bin(os.path.join(OUT_DIR, "X_hat.bin"))  # (F, M, T)
    Zxx_out = X_hat.transpose(1, 0, 2)                          # (M, F, T)
    _, wpe_out = signal.istft(Zxx_out, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Alinea longitudes (la iSTFT puede devolver un largo distinto).
    n = min(mic_signals.shape[1], wpe_out.shape[1], target_early.shape[1])
    x_in = mic_signals[ref, :n]
    y_out = wpe_out[ref, :n]
    e_ref = target_early[ref, :n]

    peak = max(np.max(np.abs(x_in)), np.max(np.abs(y_out)), np.max(np.abs(e_ref)))
    scale = 0.9 / (peak + 1e-12)

    print(f"Guardando WAVs en {OUT_DIR} (mic {ref})...")
    save_wav_shared_scale(os.path.join(OUT_DIR, "1_input_reverberant.wav"), fs, x_in, scale)
    save_wav_shared_scale(os.path.join(OUT_DIR, "2_wpe_cpp_output.wav"), fs, y_out, scale)
    save_wav_shared_scale(os.path.join(OUT_DIR, "3_target_early_reference.wav"), fs, e_ref, scale)
    print("\nListo. Compara 1_input vs 2_wpe_cpp_output.")


if __name__ == "__main__":
    main()
