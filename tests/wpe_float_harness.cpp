// wpe_float_harness.cpp
// =====================
// Harness de test para el WPE funcional de memoria estatica (wpe_float.cpp).
// Lee un tensor STFT Y.bin, maneja wpe_step UNA vez por frame (el estado static
// persiste entre llamadas) y escribe el tensor dereverberado X_hat.bin.
//
// Formato binario (little-endian):
//   int32 T, int32 F, int32 M,
//   luego T*F*M complex64 (float32 re,im intercalados) en orden [t][f][m].
// complex64 de numpy == struct complex { float re, im; }  (8 bytes, sin padding).
//
// Compilar:
//   g++ -O2 -o wpe_float_harness wpe_float_harness.cpp
// Usar:
//   ./wpe_float_harness Y.bin X_hat.bin

#include "../src/cpp/wpe_float.cpp"

#include <cstdio>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "uso: %s Y.bin X_hat.bin\n", argv[0]);
        return 1;
    }

    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) { std::perror("fopen Y.bin"); return 1; }

    int32_t T = 0, Ff = 0, Mm = 0;
    std::fread(&T,  sizeof(int32_t), 1, fin);
    std::fread(&Ff, sizeof(int32_t), 1, fin);
    std::fread(&Mm, sizeof(int32_t), 1, fin);

    if (Ff != F || Mm != M) {
        std::fprintf(stderr,
            "dimensiones no coinciden: Y.bin tiene F=%d M=%d, "
            "binario compilado con F=%d M=%d\n", Ff, Mm, F, M);
        std::fclose(fin);
        return 2;
    }

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::perror("fopen X_hat.bin"); std::fclose(fin); return 1; }
    std::fwrite(&T,  sizeof(int32_t), 1, fout);
    std::fwrite(&Ff, sizeof(int32_t), 1, fout);
    std::fwrite(&Mm, sizeof(int32_t), 1, fout);

    static complex frame_new[F][M];
    static complex frame_pred[F][M];

    for (int t = 0; t < T; ++t) {
        if (std::fread(frame_new, sizeof(complex), F * M, fin)
                != (size_t)(F * M)) {
            std::fprintf(stderr, "lectura incompleta en frame %d\n", t);
            std::fclose(fin); std::fclose(fout);
            return 3;
        }
        wpe_step(frame_new, frame_pred);
        std::fwrite(frame_pred, sizeof(complex), F * M, fout);
    }

    std::fclose(fin);
    std::fclose(fout);
    return 0;
}
