// wpe_fixed_harness.cpp
// =====================
// Harness de test para wpe_fixed.cpp. Lee un tensor STFT Y.bin (complex64/float),
// lo cuantiza a la entrada (in_t), maneja wpe_step frame a frame, y escribe la
// prediccion X_hat.bin (de vuelta como complex64/float).
//
// Formato binario (little-endian): int32 T,F,M luego T*F*M complex64 [t][f][m].
//
// Compilar (ajustar el path de Vitis si hace falta):
//   g++ -O2 -I /home/matias/Xilinx/2025.2/Vitis/include \
//       -o wpe_fixed_harness wpe_fixed_harness.cpp
// Usar:
//   ./wpe_fixed_harness Y.bin X_hat.bin

#include "../src/cpp/wpe_fixed.cpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "uso: %s Y.bin X_hat.bin [gnorm]\n", argv[0]);
        return 1;
    }
    // Ganancia de front-end (normalizacion): escala la entrada antes de cuantizar
    // a in_t, y desescala la salida. gnorm=1 => sin normalizacion.
    const double gnorm = (argc > 3) ? std::atof(argv[3]) : 1.0;

    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) { std::perror("fopen Y.bin"); return 1; }

    int32_t T = 0, Ff = 0, Mm = 0;
    if (std::fread(&T,  sizeof(int32_t), 1, fin) != 1 ||
        std::fread(&Ff, sizeof(int32_t), 1, fin) != 1 ||
        std::fread(&Mm, sizeof(int32_t), 1, fin) != 1) {
        std::fprintf(stderr, "header invalido\n"); std::fclose(fin); return 1;
    }
    if (Ff != F || Mm != M) {
        std::fprintf(stderr,
            "dimensiones no coinciden: Y.bin F=%d M=%d, binario F=%d M=%d\n",
            Ff, Mm, F, M);
        std::fclose(fin); return 2;
    }

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::perror("fopen X_hat.bin"); std::fclose(fin); return 1; }
    std::fwrite(&T,  sizeof(int32_t), 1, fout);
    std::fwrite(&Ff, sizeof(int32_t), 1, fout);
    std::fwrite(&Mm, sizeof(int32_t), 1, fout);

    static cpx_in   frame_new[F][M];
    static cpx_pred frame_pred[F][M];
    std::vector<float> io(F * M * 2);   // buffer float intercalado re,im

    for (int t = 0; t < T; ++t) {
        if (std::fread(io.data(), sizeof(float), F * M * 2, fin)
                != (size_t)(F * M * 2)) {
            std::fprintf(stderr, "lectura incompleta en frame %d\n", t);
            std::fclose(fin); std::fclose(fout);
            return 3;
        }
        // (Y * gnorm) -> in_t (normaliza + cuantiza, == wrapper Python)
        for (int f = 0; f < F; ++f) {
            for (int m = 0; m < M; ++m) {
                int idx = (f * M + m) * 2;
                frame_new[f][m].re = io[idx]     * gnorm;
                frame_new[f][m].im = io[idx + 1] * gnorm;
            }
        }

        wpe_step(frame_new, frame_pred);

        // pred_t -> float, desescalado por gnorm
        for (int f = 0; f < F; ++f) {
            for (int m = 0; m < M; ++m) {
                int idx = (f * M + m) * 2;
                io[idx]     = (float)((double)frame_pred[f][m].re / gnorm);
                io[idx + 1] = (float)((double)frame_pred[f][m].im / gnorm);
            }
        }
        std::fwrite(io.data(), sizeof(float), F * M * 2, fout);
    }

    std::fclose(fin);
    std::fclose(fout);
    return 0;
}
