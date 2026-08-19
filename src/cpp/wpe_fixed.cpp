// wpe_fixed.cpp
// =============
// Version PUNTO FIJO del WPE online funcional (port de wpe_float.cpp).
//
// Modelo (== config por defecto de nara_wrappers_fixed.py):
//   * ALMACENAMIENTO en fijo (ap_fixed): buffer (in), inv_cov (p), filter_taps (g),
//     y la salida (pred). Son lo que en la FPGA vive en BRAM/URAM.
//   * ARITMETICA en float (block-float): power, nominator, kalman, denom, reciproco.
//     Se cuantiza SOLO al escribir en un arreglo de estado (el resto es float).
//   * Redondeo AP_RND_CONV (banker's = np.round) + saturacion AP_SAT
//     (== rounding="nearest", saturate=True del modelo Python).
//   * Mismo algoritmo que el float: n_iter=1, SIN Hermitiana, SIN normalizacion.
//     (Golden de validacion: wordlength(W, force_hermitian=False, normalize_target=0).)
//
// W es el word length a barrer (16/18/24/32). I = int_bits + 1 (bit de signo):
//   in/pred -> int_bits=1 -> I=2 ;  g/p -> int_bits=4 -> I=5.

#include <ap_fixed.h>

// ---- CONSTANTES -------------------------------------------------------------
constexpr int   K     = 5;        // taps
constexpr int   M     = 8;        // canales
constexpr int   F     = 257;      // bins
constexpr int   KM    = K * M;
constexpr int   delay = 1;
constexpr float alpha = 0.9999f;

// ---- TIPOS DE PUNTO FIJO ----------------------------------------------------
#ifndef WLEN
#define WLEN 24                   // override con -DWLEN=N para el barrido
#endif
constexpr int W = WLEN;           // word length (barrer)

typedef ap_fixed<W, 2, AP_RND_CONV, AP_SAT> in_t;    // int_bits=1
typedef ap_fixed<W, 2, AP_RND_CONV, AP_SAT> pred_t;  // int_bits=1
typedef ap_fixed<W, 5, AP_RND_CONV, AP_SAT> g_t;     // int_bits=4
typedef ap_fixed<W, 5, AP_RND_CONV, AP_SAT> p_t;     // int_bits=4

// ---- COMPLEJO (templateado): almacenamiento fijo, aritmetica float ----------
template<typename T>
struct cpx { T re; T im; };

// Tipo del datapath block-float (aritmetica y transitorios). En HW: float.
// Ponerlo en double sirve solo para diagnostico (igualar la precision de Python).
typedef float real_t;

typedef cpx<in_t>   cpx_in;
typedef cpx<pred_t> cpx_pred;
typedef cpx<g_t>    cpx_g;
typedef cpx<p_t>    cpx_p;
typedef cpx<real_t> cpx_f;         // block-float

// Convierte cualquier cpx almacenado a float para operar.
template<typename T>
inline cpx_f to_f(const cpx<T>& z) {
    cpx_f r;
    r.re = (real_t)z.re;
    r.im = (real_t)z.im;
    return r;
}

// ---- INDEX HELPERS ----------------------------------------------------------
inline int get_window_idx(int m, int k) {
    return m * K + k;   // layout per-bin (M, K), channel-major
}

// ---- OPERACIONES COMPLEJAS (en float) ---------------------------------------
inline real_t cpx_abs2(cpx_f z) {
    return z.re * z.re + z.im * z.im;
}

inline cpx_f cpx_mul(cpx_f a, cpx_f b) {
    cpx_f r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

inline cpx_f cpx_sum(cpx_f a, cpx_f b) {
    cpx_f r;
    r.re = a.re + b.re;
    r.im = a.im + b.im;
    return r;
}

inline cpx_f cpx_sub(cpx_f a, cpx_f b) {
    cpx_f r;
    r.re = a.re - b.re;
    r.im = a.im - b.im;
    return r;
}

// ---- FUNCIONES --------------------------------------------------------------

// G <- G + kalman (x) conj(pred).  Se calcula en float, se cuantiza a g_t al guardar.
void update_filter_taps(const cpx_f (&kalman_gain_bin)[KM],
                        const cpx_f (&pred_bin)[M],
                        cpx_g (&filter_taps)[KM][M]) {
    for (int r = 0; r < KM; ++r) {
        for (int c = 0; c < M; ++c) {
            cpx_f pred_conj = { pred_bin[c].re, -pred_bin[c].im };
            cpx_f g_new = cpx_sum(to_f(filter_taps[r][c]),
                                  cpx_mul(kalman_gain_bin[r], pred_conj));
            filter_taps[r][c].re = g_new.re;   // float -> g_t (cuantiza)
            filter_taps[r][c].im = g_new.im;
        }
    }
}

// P <- (P - kalman (x) (w^H P)) / alpha.  Float; se cuantiza a p_t al guardar.
void update_inv_cov(const cpx_in (&window_bin)[KM],
                    const cpx_f (&kalman_gain_bin)[KM],
                    cpx_p (&inv_cov)[KM][KM]) {
    cpx_f wH_R[KM];
    // wH_R[c] = sum_r conj(w[r]) * P[r][c]
    for (int c = 0; c < KM; ++c) {
        cpx_f acc = { 0.f, 0.f };
        for (int r = 0; r < KM; ++r) {
            cpx_f w = to_f(window_bin[r]);
            cpx_f w_conj = { w.re, -w.im };
            acc = cpx_sum(acc, cpx_mul(w_conj, to_f(inv_cov[r][c])));
        }
        wH_R[c] = acc;
    }
    const real_t inv_alpha = 1.0f / alpha;
    // Update (float) + simetrizacion Hermitiana  P <- 1/2 (P + P^H),  luego cuantiza.
    // Se recorre por pares (i,j)/(j,i) leyendo el P VIEJO; cada par es disjunto,
    // asi que actualizar in place no aliasa.
    for (int i = 0; i < KM; ++i) {
        for (int j = i; j < KM; ++j) {
            // updates float de (i,j) y (j,i) a partir del P viejo
            cpx_f u_ij = cpx_sub(to_f(inv_cov[i][j]),
                                 cpx_mul(kalman_gain_bin[i], wH_R[j]));
            cpx_f u_ji = cpx_sub(to_f(inv_cov[j][i]),
                                 cpx_mul(kalman_gain_bin[j], wH_R[i]));
            u_ij.re *= inv_alpha; u_ij.im *= inv_alpha;
            u_ji.re *= inv_alpha; u_ji.im *= inv_alpha;
            // Hermitiana: h_ij = 1/2 (u_ij + conj(u_ji)),  h_ji = conj(h_ij)
            real_t h_re = 0.5f * (u_ij.re + u_ji.re);
            real_t h_im = 0.5f * (u_ij.im - u_ji.im);
            inv_cov[i][j].re = h_re;      // float -> p_t
            inv_cov[i][j].im = h_im;
            inv_cov[j][i].re = h_re;
            inv_cov[j][i].im = -h_im;     // conjugado (en la diagonal i==j da im=0)
        }
    }
}

// kalman = (P . window) / (alpha*power + w^H P w).  Todo float (block-float).
void update_kalman_gain(const real_t power,
                        const cpx_in (&window_bin)[KM],
                        cpx_f (&kalman_gain_bin)[KM],
                        const cpx_p (&inv_cov)[KM][KM]) {
    // Nominator = P . window
    for (int r = 0; r < KM; ++r) {
        cpx_f acc = { 0.f, 0.f };
        for (int c = 0; c < KM; ++c) {
            acc = cpx_sum(acc, cpx_mul(to_f(inv_cov[r][c]), to_f(window_bin[c])));
        }
        kalman_gain_bin[r] = acc;
    }
    // Denominador = alpha*power + Re(w^H . nominator)
    real_t denom_right = 0.f;
    for (int r = 0; r < KM; ++r) {
        cpx_f w = to_f(window_bin[r]);
        denom_right += w.re * kalman_gain_bin[r].re
                     + w.im * kalman_gain_bin[r].im;
    }
    const real_t eps = 1e-20f;
    real_t denom = power * alpha + denom_right;
    real_t inv_denom = 1.0f / (denom > eps ? denom : eps);
    // Escalado por el reciproco (complejo * real)
    for (int r = 0; r < KM; ++r) {
        kalman_gain_bin[r].re *= inv_denom;
        kalman_gain_bin[r].im *= inv_denom;
    }
}

// power = media de |Y|^2 sobre canales y todo el buffer.  Float.
real_t compute_power_bin(const cpx_in (&buffer_bin)[M][K + delay + 1]) {
    real_t power = 0.f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < K + delay + 1; ++n) {
            power += cpx_abs2(to_f(buffer_bin[m][n]));
        }
    }
    return power / (M * (delay + K + 1));
}

// window (in_t) = snapshot de los K frames viejos del buffer (in_t). Copia exacta.
void build_window_bin(cpx_in (&window_bin)[KM],
                      const cpx_in (&buffer_bin)[M][K + delay + 1]) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            window_bin[get_window_idx(m, k)] = buffer_bin[m][k];
        }
    }
}

// Shift-register: descarta el mas viejo, mete el frame nuevo en el indice alto.
void update_buffer_bin(const cpx_in (&frame_new_bin)[M],
                       cpx_in (&buffer_bin)[M][K + delay + 1]) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < K + delay; ++n) {
            buffer_bin[m][n] = buffer_bin[m][n + 1];
        }
        buffer_bin[m][K + delay] = frame_new_bin[m];
    }
}

// pred = Y_t - G^H . window.  Usa el filtro VIEJO. Transitorio en float.
void get_prediction(const cpx_in (&frame_new_bin)[M],
                    const cpx_in (&window_bin)[KM],
                    const cpx_g (&filter_taps)[KM][M],
                    cpx_f (&pred_bin)[M]) {
    for (int m = 0; m < M; ++m) {
        cpx_f acc = { 0.f, 0.f };
        for (int j = 0; j < KM; ++j) {
            cpx_f g = to_f(filter_taps[j][m]);
            cpx_f g_conj = { g.re, -g.im };
            acc = cpx_sum(acc, cpx_mul(g_conj, to_f(window_bin[j])));
        }
        pred_bin[m] = cpx_sub(to_f(frame_new_bin[m]), acc);
    }
}

// ---- TOP: un frame (F, M) -> prediccion (F, M) ------------------------------
void wpe_step(const cpx_in (&frame_new)[F][M], cpx_pred (&frame_pred)[F][M]) {
    static cpx_in buffer[F][M][K + delay + 1] = {};
    static cpx_g  filter_taps[F][KM][M] = {};
    static cpx_p  inv_cov[F][KM][KM];              // off-diag por static zero-init
    static bool   init_done = false;
    static int    frame_count = 0;

    // Init inv_cov = identidad (una sola vez).
    if (!init_done) {
        for (int f = 0; f < F; ++f)
            for (int d = 0; d < KM; ++d) {
                inv_cov[f][d][d].re = 1;
                inv_cov[f][d][d].im = 0;
            }
        init_done = true;
    }

    for (int f = 0; f < F; ++f) {
        update_buffer_bin(frame_new[f], buffer[f]);

        if (frame_count < K + delay) {
            // Warmup: bypass (salida = entrada), sin actualizar P/G.
            for (int m = 0; m < M; ++m) {
                frame_pred[f][m].re = frame_new[f][m].re;   // in_t -> pred_t (mismo formato)
                frame_pred[f][m].im = frame_new[f][m].im;
            }
        } else {
            cpx_in window_bin[KM];
            cpx_f  kalman_gain_bin[KM];
            cpx_f  pred_bin[M];

            build_window_bin(window_bin, buffer[f]);
            get_prediction(frame_new[f], window_bin, filter_taps[f], pred_bin);

            // Salida: pred (float transitorio) -> pred_t
            for (int m = 0; m < M; ++m) {
                frame_pred[f][m].re = pred_bin[m].re;
                frame_pred[f][m].im = pred_bin[m].im;
            }

            real_t power = compute_power_bin(buffer[f]);
            update_kalman_gain(power, window_bin, kalman_gain_bin, inv_cov[f]);
            update_inv_cov(window_bin, kalman_gain_bin, inv_cov[f]);
            // update_taps usa el pred FLOAT transitorio (no el cuantizado de salida).
            update_filter_taps(kalman_gain_bin, pred_bin, filter_taps[f]);
        }
    }

    frame_count++;
}
