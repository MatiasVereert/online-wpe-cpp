#include <array>
#include <complex>

// CONSTANTS 
constexpr int K = 5;
constexpr int M = 8;
constexpr int F = 257; 
constexpr int KM = K * M; 
constexpr int delay = 1; 
constexpr float power_estimate = 0.0;
constexpr float alpha = 0.9999f;

// DATA TYPES  
typedef float data_t; 

struct complex{
  data_t re;
  data_t im;
};

// INDEX HELPERS 
inline size_t get_window_idx( int m, int k)  {
  // Snapshot of the regression window, layout (F, M, K)
  return  m * K + k;
}

inline size_t get_kalman_gain_idx( int m_row, int k_row){
  return  m_row * K+ k_row;
}

// Complex operations

inline data_t cpx_abs2(complex z) {
    return z.re * z.re + z.im * z.im;   // |z|^2
}

inline complex cpx_mul(complex a, complex b) {
    complex r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}


inline complex cpx_sum(complex a, complex b) {
    complex r;
    r.re = a.re +b.re;
    r.im =  b.im + a.im;
    return r;
}


inline complex cpx_sub(complex a, complex b) {
    complex r;
    r.re = a.re -b.re;
    r.im =  a.im - b.im;
    return r;
}

// FUNCTIONS 

void update_filter_taps(const complex (&kalman_gain_bin)[KM],
                        const complex (&frame_pred)[M],
                        complex (&filter_taps)[KM][M]){
                          
  for (int r=0; r<KM; r++){
    for (int c=0; c<M; c++){
        filter_taps[r][c] =cpx_sum(filter_taps[r][c],
                          cpx_mul(kalman_gain_bin[r],
                          {frame_pred[c].re, -frame_pred[c].im }));
    }
  }
}


void update_inv_cov(const complex (&window_bin)[KM],
                 const complex (&kalman_gain_bin)[KM], 
                 complex (&inv_cov)[KM][KM]){

  complex acc = {0};
  complex wH_R[KM] = {0};
  float inv_alpha = 1/alpha; 
  
  for (int c =0; c<KM; ++c){
      for (int r =0; r<KM; ++r){
        complex w_conj = { window_bin[r].re , - window_bin[r].im};
        acc = cpx_sum( acc, cpx_mul(w_conj,inv_cov[r][c] ));
      }
    wH_R[c] = acc;
    acc = { 0, 0}; 
    } 
    for (int c =0; c<KM; ++c){
      for (int r =0; r<KM; ++r){
        inv_cov[c][r] = cpx_sub( inv_cov[c][r], cpx_mul(kalman_gain_bin[c], wH_R[r])) ;
        inv_cov[c][r].re = inv_cov[c][r].re * inv_alpha;
        inv_cov[c][r].im = inv_cov[c][r].im * inv_alpha;
      }
    }
}



void update_kalman_gain(const float power, 
                        const complex (&window_bin)[KM],
                        complex (&kalman_gain_bin)[KM], 
                        const complex (&inv_cov)[KM][KM]){
  complex acc = {0};

  float inv_denom = power * alpha;
  float denom_right_term = 0.0;

  // Nominator 
  for (int r=0; r<KM; ++r){
    for(int c=0; c<KM; ++c){
      acc = cpx_sum(acc , cpx_mul( inv_cov[r][c], window_bin[c] ));
    }
    kalman_gain_bin[r] = acc; 
    acc.re = 0; 
    acc.im = 0;
  }
  // Denominator
  for (int r=0; r<KM; ++r){

    denom_right_term +=  window_bin[r].re * kalman_gain_bin[r].re
         + window_bin[r].im * kalman_gain_bin[r].im;
  }
  // Use acc as denominator to faciltate operation
  const float eps = 1e-20f;                         // piso
  float denom = inv_denom + denom_right_term;       // denom completo = alpha*power + wHn
  inv_denom = 1.0f / (denom > eps ? denom : eps);   // recíproco con piso

  for (int r=0; r<KM; ++r){

    kalman_gain_bin[r].re = kalman_gain_bin[r].re * inv_denom;
    kalman_gain_bin[r].im = kalman_gain_bin[r].im * inv_denom;
  }
}

float compute_power_bin( complex (&buffer_bin)[M][ K + delay + 1] ){
  float power = 0;
  for ( int m = 0; m<M; ++m){
    for (int n = 0; n< K + delay+1; ++n){
      power += cpx_abs2(buffer_bin[m][n] ); 
    }
  }
  return power/ (M*(delay+ K + 1));
}

void build_window_bin(complex (&window_bin)[KM], 
                      complex (&buffer_bin)[M][ K + delay + 1] ){
    for (int m =0; m<M; ++m ){
      for (int k = 0; k < K ; ++k){
      window_bin[get_window_idx(m, k)] = buffer_bin[m][k];
    }
  }
}

void update_buffer_bin(const complex (&frame_new_bin)[M],
                      complex (&buffer_bin)[M][K + delay + 1]) {
  
  // shift-register acá
  for (int m =0; m<M; ++m ){
    for (int n =0; n<K + delay ; ++n){
      buffer_bin[m][n] = buffer_bin[m][n+1];
    }
  buffer_bin[m][ K + delay ] = frame_new_bin[m];
  }
}


void get_prediction(const complex (&frame_new_bin)[M],
                    const complex (&window_bin)[KM],
                    const complex (&filter_taps)[KM][M],
                          complex (&frame_pred)[M]) {
  
  complex acc = {0};
  complex taps_conj;
  
  for (int m =0; m<M; m++){
    for (int j =0; j<KM; j++){
      taps_conj = {filter_taps[j][m].re, -filter_taps[j][m].im};
      acc = cpx_sum( acc, cpx_mul( taps_conj ,window_bin[j]));
    }
    
    frame_pred[m] = cpx_sub(frame_new_bin[m], acc);
    acc = {0,0};
  }
}
void wpe_step(const complex (&frame_new)[F][M], complex (&frame_pred)[F][M]) {
    static complex buffer[F][M][K + delay + 1] = {};
    static complex filter_taps[F][KM][M] = {};
    static complex inv_cov[F][KM][KM];
    static bool init_done = false;
    static int  frame_count = 0;

    // Init inv_cov = identidad (una sola vez)
    if (!init_done) {
        for (int f = 0; f < F; ++f)
            for (int d = 0; d < KM; ++d)
                inv_cov[f][d][d] = {1, 0};
        init_done = true;
    }

    // UN solo loop sobre bins, con warmup vs step
    for (int f = 0; f < F; ++f) {
        update_buffer_bin(frame_new[f], buffer[f]);        // siempre, UNA vez

        if (frame_count < K + delay) {
            // Warmup: bypass (salida = entrada), sin actualizar P/G
            for (int m = 0; m < M; ++m)
                frame_pred[f][m] = frame_new[f][m];
        } else {
            // Step completo
            complex window_bin[KM];
            complex kalman_gain_bin[KM];

            build_window_bin(window_bin, buffer[f]);
            get_prediction(frame_new[f], window_bin, filter_taps[f], frame_pred[f]);
            float power = compute_power_bin(buffer[f]);
            update_kalman_gain(power, window_bin, kalman_gain_bin, inv_cov[f]);
            update_inv_cov(window_bin, kalman_gain_bin, inv_cov[f]);
            update_filter_taps(kalman_gain_bin, frame_pred[f], filter_taps[f]);
        }
    }

    frame_count++;   // una vez por frame (por llamada), FUERA del loop de bins
}