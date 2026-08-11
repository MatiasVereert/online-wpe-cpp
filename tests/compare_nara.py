"""
compare_nara.py
===============
Corre el OnlineWPE de referencia (paquete nara_wpe) sobre EXACTAMENTE el mismo
tensor STFT que consume el wrapper en C++ (`Y.bin`), y exporta WAVs con escala
compartida para comparar a oido:

    1_input_reverberant.wav        entrada reverberante
    2_wpe_cpp_output.wav           salida del OnlineWPE en C++  (leida de X_hat.bin)
    4_nara_online_output.wav       salida del OnlineWPE de nara (referencia)
    3_target_early_reference.wav   referencia "ideal" (early)

La unica diferencia entre 2 y 4 es la implementacion (tu port vs nara): mismo
STFT, misma escena, mismos parametros.

USO (despues de correr ./test_wpe_online Y.bin X_hat.bin):
    /home/matias/miniconda3/envs/tesis_beam/bin/python compare_nara.py
"""

import os
import sys

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

# nara_wpe (paquete de referencia)
sys.path.insert(0, "/home/matias/Documents/Tesis/nara/nara_wpe")
from nara_wpe.wpe import OnlineWPE

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Parametros WPE: MANTENER EN SINCRONIA con test_wpe_online.cpp -----------
TAPS = 7
DELAY = 1
ALPHA = 0.99999


def read_stft_bin(path):
    """Lee el binario (header T,M,F + complex128 orden-C) y devuelve (F, M, T)."""
    with open(path, "rb") as fi:
        T, M, F = np.fromfile(fi, dtype=np.int32, count=3)
        n = int(T) * int(M) * int(F)
        data = np.fromfile(fi, dtype=np.complex128, count=n)
    return data.reshape(int(T), int(M), int(F)).transpose(2, 1, 0)  # (F, M, T)


def run_nara_online(Y):
    """Corre nara OnlineWPE frame a frame. Y: (F, M, T) -> X_hat: (F, M, T)."""
    F, M, T = Y.shape
    wpe = OnlineWPE(taps=TAPS, delay=DELAY, alpha=ALPHA,
                    channel=M, frequency_bins=F)
    X = np.zeros_like(Y)
    for t in range(T):
        # nara step_frame espera un frame de forma (F, D).
        X[:, :, t] = wpe.step_frame(Y[:, :, t])
    return X


def istft_ref(X_fmt, fs, nperseg, noverlap):
    """X_fmt: (F, M, T) -> senal (M, N)."""
    _, x = signal.istft(X_fmt.transpose(1, 0, 2), fs=fs,
                        nperseg=nperseg, noverlap=noverlap)
    return x


def save_wav(path, fs, sig, scale):
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

    Y = read_stft_bin(os.path.join(OUT_DIR, "Y.bin"))          # (F, M, T)
    X_cpp = read_stft_bin(os.path.join(OUT_DIR, "X_hat.bin"))  # (F, M, T)

    print(f"Corriendo nara OnlineWPE (taps={TAPS}, delay={DELAY}, alpha={ALPHA})...")
    X_nara = run_nara_online(Y)

    cpp_out = istft_ref(X_cpp, fs, nperseg, noverlap)
    nara_out = istft_ref(X_nara, fs, nperseg, noverlap)

    # Alinea longitudes.
    n = min(mic_signals.shape[1], cpp_out.shape[1],
            nara_out.shape[1], target_early.shape[1])
    x_in = mic_signals[ref, :n]
    y_cpp = cpp_out[ref, :n]
    y_nara = nara_out[ref, :n]
    e_ref = target_early[ref, :n]

    # Escala compartida entre las cuatro senales -> comparacion de nivel justa.
    peak = max(np.max(np.abs(x_in)), np.max(np.abs(y_cpp)),
               np.max(np.abs(y_nara)), np.max(np.abs(e_ref)))
    scale = 0.9 / (peak + 1e-12)

    print(f"Guardando WAVs en {OUT_DIR} (mic {ref})...")
    save_wav(os.path.join(OUT_DIR, "1_input_reverberant.wav"), fs, x_in, scale)
    save_wav(os.path.join(OUT_DIR, "2_wpe_cpp_output.wav"), fs, y_cpp, scale)
    save_wav(os.path.join(OUT_DIR, "4_nara_online_output.wav"), fs, y_nara, scale)
    save_wav(os.path.join(OUT_DIR, "3_target_early_reference.wav"), fs, e_ref, scale)

    # Metrica rapida: energia media |.| por frame en el mic de referencia.
    def band_energy(Xf):
        return np.abs(Xf[:, ref, :]).mean()
    print("\nEnergia media |STFT| (mic ref):")
    print(f"  entrada Y : {band_energy(Y):.4e}")
    print(f"  C++       : {band_energy(X_cpp):.4e}")
    print(f"  nara      : {band_energy(X_nara):.4e}")
    # Diferencia relativa C++ vs nara en el dominio STFT.
    num = np.linalg.norm(X_cpp - X_nara)
    den = np.linalg.norm(X_nara) + 1e-12
    print(f"  ||C++ - nara|| / ||nara|| = {num/den:.4f}")


if __name__ == "__main__":
    main()
