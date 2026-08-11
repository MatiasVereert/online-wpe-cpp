// test_wpe_online.cpp
// ---------------------------------------------------------------------------
// Standalone test driver for the OnlineWPE C++ class.
//
// It reads an STFT tensor Y produced by gen_signals.py, runs the online WPE
// frame-by-frame, and writes the dereverberated STFT tensor X_hat back to
// disk so that eval_output.py can perform the iSTFT and save the WAVs.
//
// Binary exchange format (little-endian, matches numpy on x86):
//   header : 3 x int32     -> T, M, F
//   data   : T*M*F x complex128 in C-order (t, m, f)
//
// NOTE: OnlineWPE is a template, so its member definitions must be visible at
// the point of instantiation. That is why we include the .cpp directly here
// (a normal .h/.cpp split would fail to link). This is fine for a test driver.
// ---------------------------------------------------------------------------

#include <array>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/cpp/wpe.cpp"

// Must match the number of microphones in the scene (MIRD linear array = 8).
constexpr size_t NUM_CHANNELS = 8;

int main(int argc, char** argv) {
    // --- CLI ---------------------------------------------------------------
    std::string in_path  = "Y.bin";
    std::string out_path = "X_hat.bin";
    if (argc >= 2) in_path  = argv[1];
    if (argc >= 3) out_path = argv[2];

    // --- WPE parameters (kept in sync with the python reference) -----------
    const double alpha = 0.99999;
    const int    taps  = 7;
    const int    delay = 1;

    // --- Read Y.bin --------------------------------------------------------
    std::ifstream fin(in_path, std::ios::binary);
    if (!fin) { std::cerr << "Cannot open " << in_path << "\n"; return 1; }

    int32_t T = 0, M = 0, F = 0;
    fin.read(reinterpret_cast<char*>(&T), sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(&M), sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(&F), sizeof(int32_t));

    if (static_cast<size_t>(M) != NUM_CHANNELS) {
        std::cerr << "Channel mismatch: file has M=" << M
                  << " but the driver was built with NUM_CHANNELS="
                  << NUM_CHANNELS << ".\n";
        return 1;
    }

    const size_t n_total = static_cast<size_t>(T) * M * F;
    std::vector<Complex> Y(n_total);
    fin.read(reinterpret_cast<char*>(Y.data()),
             static_cast<std::streamsize>(n_total * sizeof(Complex)));
    if (!fin) { std::cerr << "Failed reading Y data\n"; return 1; }
    fin.close();

    std::cout << "Loaded Y: T=" << T << " M=" << M << " F=" << F << "\n";

    // --- Rough initial power estimate (mean |Y|^2 over the whole tensor) ---
    double power_estimate = 0.0;
    for (const auto& z : Y) power_estimate += std::norm(z);
    power_estimate /= static_cast<double>(n_total);

    // --- Instantiate and run frame-by-frame --------------------------------
    OnlineWPE<NUM_CHANNELS> wpe(
        static_cast<float>(alpha), taps, delay, M, F, power_estimate);

    std::vector<Complex> X_hat(n_total, Complex(0.0, 0.0));

    // Per-timestep frame: one complex vector of length F per channel.
    std::array<std::vector<Complex>, NUM_CHANNELS> frame;
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) frame[ch].resize(F);

    for (int t = 0; t < T; ++t) {
        // Gather Y[t, :, :] into the channel-major frame layout.
        for (int m = 0; m < M; ++m) {
            const size_t base = (static_cast<size_t>(t) * M + m) * F;
            for (int f = 0; f < F; ++f) frame[m][f] = Y[base + f];
        }

        const std::vector<Complex>& pred = wpe.step_frame(frame);

        // prediction is indexed as f*M + m -> scatter into X_hat[t, m, f].
        for (int m = 0; m < M; ++m) {
            const size_t base = (static_cast<size_t>(t) * M + m) * F;
            for (int f = 0; f < F; ++f) X_hat[base + f] = pred[f * M + m];
        }
    }

    // --- Write X_hat.bin ---------------------------------------------------
    std::ofstream fout(out_path, std::ios::binary);
    if (!fout) { std::cerr << "Cannot open " << out_path << " for writing\n"; return 1; }
    fout.write(reinterpret_cast<const char*>(&T), sizeof(int32_t));
    fout.write(reinterpret_cast<const char*>(&M), sizeof(int32_t));
    fout.write(reinterpret_cast<const char*>(&F), sizeof(int32_t));
    fout.write(reinterpret_cast<const char*>(X_hat.data()),
               static_cast<std::streamsize>(n_total * sizeof(Complex)));
    fout.close();

    std::cout << "Wrote " << out_path << "\n";
    return 0;
}
