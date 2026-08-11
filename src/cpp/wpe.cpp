#include <vector> 
#include <complex> 
#include <array> 
#include <algorithm>

using Complex = std::complex<double>;

template <size_t num_channels>

class OnlineWPE {
  private:
    // Helper Methods for Indexation
    inline size_t get_cov_idx(int f, int m_row, int k_row, int m_col, int k_col ) const {
      return f * (M * K * M * K) 
            + m_row * (K * M * K) 
            + k_row * (M * K)
            + m_col * K
            + k_col; 
    }
    inline size_t get_cov_right_term_idx(int m_row, int k_row, int m_col, int k_col ) const {
      return  m_row * (K * M * K) 
            + k_row * (M * K)
            + m_col * K
            + k_col; 
    }
    inline size_t get_filter_taps_idx(int f, int m_row, int k_row, int m_col) const {
      // Added frequency stride calculation
      return f * (M * K * M) 
           + m_row * (K * M) 
           + k_row * M
           + m_col;
    }

    inline size_t get_kalman_gain_idx( int f, int m_row, int k_row) const {
      return f * (M*K) + m_row * K+ k_row;
    }

    inline size_t get_window_idx( int f, int m, int k) const {
      // Snapshot of the regression window, layout (F, M, K)
      return f * (M*K) + m * K + k;
    }

    inline size_t get_buffer_idx( int f, int k_tap) const {
      // taps axis is mirrored to align with math
      // buffer has (past samples)<(present)
      // this idx returns (past samples)>(present)
      return f * (K + delay +1) + K - 1 -k_tap; 
    }



  public:
    double alpha; 
    int K;  // taps 
    int delay;
    int M ; // Mics dimension 
    int F ; // Frecuency bins

    std::vector<Complex> inv_cov;
    std::vector<Complex> filter_taps;
    std::vector<double> power;
    std::vector<Complex> kalman_gain;
    std::vector<Complex> prediction;
    std::vector<Complex> window;  // regression window snapshot (F, M, K)
    // Aux

    std::array<std::vector<Complex>, num_channels> buffer;

    // Define the constructor

    OnlineWPE( float a, int b, int c, int d, int e, double power_estimate); 

    void update_buffer(const std::array<std::vector<Complex>, num_channels> & new_frame);
    void update_power_block();
    void update_kalman_gain();
    void update_inv_cov();
    void update_taps();
    const std::vector<Complex>& step_frame(const std::array<std::vector<Complex>, num_channels> & new_frame);
    void get_prediction(const std::array<std::vector<Complex>, num_channels> & new_frame);

};

// CONSTRUCTOR
template <size_t num_channels>
OnlineWPE<num_channels>::OnlineWPE( float a, int b, int c, int d, int e, double power_estimate) {
  alpha = a;
  K = b;
  delay = c;
  M = d;
  F = e;
  
  filter_taps.resize(F * M * K * M, Complex(0.0,0.0));

  // Init. cov_matrix as Identity
  int inv_cov_len = F*M*K*M*K;
  int cov_matrix_size = M*K;
  int diag_idx_f;
  int diag_idx;

  // Variables
  inv_cov.resize(inv_cov_len, Complex(0.0,0.0)); //shape (F, MK, MK)
  power.resize(F , power_estimate); 
  kalman_gain.resize( F* K* M, Complex(0.0,0.0));
  prediction.resize(F* M, Complex(0.0,0.0) );
  window.resize( F* M* K, Complex(0.0,0.0));

  // Auxiliar
  for( int f = 0; f <F; f++ ) {
  diag_idx_f = f * cov_matrix_size * cov_matrix_size;

    for (int i = 0; i<cov_matrix_size; i++){
      diag_idx = diag_idx_f + (cov_matrix_size + 1) * i;
      inv_cov[diag_idx] = Complex( 1.0, 0.0 );
    }
  }                             

  // Init buffer
  for (size_t ch = 0; ch < num_channels; ++ch){
    buffer[ch].resize( F * ( K + delay + 1));
  }
}
template <size_t num_channels>
const std::vector<Complex>& OnlineWPE<num_channels>::step_frame(const std::array<std::vector<Complex>, num_channels> & new_frame) {

    get_prediction(new_frame);
    update_buffer(new_frame);
    update_power_block();
    update_kalman_gain();
    update_inv_cov();
    update_taps();

    // Return by constant reference to avoid expensive memory copies during runtime
    return prediction;
}

template <size_t num_channels>
  void OnlineWPE<num_channels>::get_prediction(const std::array<std::vector<Complex>, num_channels> & new_frame){

  int  taps_idx;
  int  window_idx;
  int pred_idx;
  std::fill( prediction.begin(), prediction.end(), Complex(0.0, 0.0));

  // Snapshot the regression window from the CURRENT buffer, before the new
  // frame is pushed by update_buffer. The RLS updates (kalman gain, inv_cov)
  // must reuse exactly this window, mirroring nara's OnlineWPE where `window`
  // is computed once and passed to every update.
  for (int f=0; f<F; ++f){
    for (int m1=0; m1<M; ++m1){
      for (int k=0; k<K; ++k){
        window[get_window_idx(f, m1, k)] = buffer[m1][get_buffer_idx(f, k)];
      }
    }
  }

  for (int f=0;f<F; ++f){
    //brodcast f1


    for (int m0=0; m0<M; ++m0){
      Complex inner_summ(0.0,0.0);
      for (int k=0; k<K; ++k){
        for (int m1=0; m1<M; ++m1){
          taps_idx = get_filter_taps_idx(f, m1, k, m0);
          window_idx = get_window_idx(f, m1, k);

          inner_summ += std::conj(filter_taps[taps_idx]) * window[window_idx];
        }
      }
      pred_idx = f * M + m0;
      prediction[pred_idx] = new_frame[m0][f] - inner_summ;
    }
  }
}



template <size_t num_channels>
void OnlineWPE<num_channels>::update_taps(){
  int kalman_idx; 
  int taps_idx;
  // x_hat has dimensin M


  //(DK,1)(D)-> (DK,D)
  for (int f=0;f<F; ++f){
    //brodcast f
    for (int m0=0; m0<M; ++m0){
      for (int k=0; k<K; ++k){
        for (int m1=0; m1<M; ++m1){
          // indices
          kalman_idx = get_kalman_gain_idx(f, m0, k );
          taps_idx = get_filter_taps_idx( f, m0, k, m1);

          filter_taps[taps_idx] += kalman_gain[kalman_idx] * std::conj(prediction[ f* M+ m1]); 
        }
      } 
    }
  }
}

template <size_t num_channels>
void OnlineWPE<num_channels>::update_inv_cov(){

  // Define stridles
  int kalman_idx;
  int inv_cov_idx;

  for (int f= 0; f<F; ++f){

    for (int m0=0; m0<M; ++m0){
      for (int k0=0; k0<K; ++k0){
        Complex inner_summ(0.0,0.0);

        for (int m1=0; m1<M; ++m1){
          for (int k1=0; k1<K; ++k1){
            // first operation is y_buffer dot R_inv (1, MK)(MK,MK) (1, MK)
            inv_cov_idx = get_cov_idx(f,m1,k1,m0,k0 );

            inner_summ += std::conj(window[get_window_idx(f, m1, k1)]) * inv_cov[inv_cov_idx];
          }
        }
      for (int m1=0; m1<M; ++m1){
          for (int k1=0; k1<K; ++k1){ 
            inv_cov_idx = get_cov_idx(f, m1, k1, m0, k0 );
            kalman_idx = get_kalman_gain_idx(f, m1,k1);

            inv_cov[inv_cov_idx] = (inv_cov[inv_cov_idx] - 
                                inner_summ * kalman_gain[kalman_idx])/ alpha;

          }
        }
      }
    }
  }
}



template <size_t num_channels>
void OnlineWPE<num_channels>::update_kalman_gain(){
  int len_MK = M * K;
  int len_KMK = K * M * K;
  int m_left_idx = 0;
  int k_left_idx = 0;
  int m_right_idx = 0;
  int f_idx_vec  =0;
  int m_left_vec = 0;
  int idx_m =0;

  // Clean past kalman gain 
  std::fill(kalman_gain.begin(), kalman_gain.end(), Complex(0.0, 0.0));

  // Nominator: R_inv dot y_taps 
  // Shape F, DKx1 

  // y_taps is buffer in between idx =0 and idx = K 
  // as buffer[ch][0] = tap_K and buffer[ch][K] = tap_0
  // it must be mirrored 
  int f_idx =0; 
  for (int f= 0; f<F; ++f ){
    f_idx = f * len_MK * len_MK;
    f_idx_vec = f * len_MK;

    // Aux vector
   
  
    // Tensor shape ( m_left, K_left,m_right, K_right, )
    // Using row major standard of the matrix is define as
    // m_left * K_left * m_right * K_right + k_left * m_right * K_right + m_right * K_right + k_right

    for ( int m_left = 0; m_left < M; ++m_left){
      m_left_idx = m_left * len_KMK;
      m_left_vec = m_left * K;

      for ( int k_left = 0; k_left < K; ++k_left){
        k_left_idx =m_left_idx + k_left * len_MK ;

        for ( int m_right = 0; m_right < M; ++m_right){
          m_right_idx = k_left_idx + m_right * K;

          for ( int k_right = 0; k_right < K; ++k_right){

            kalman_gain[f_idx_vec   + m_left_vec + k_left ] +=  inv_cov[f_idx + m_right_idx + k_right]
                                                              * window[get_window_idx(f, m_right, k_right)];
          }       
        }
      }
    }

  Complex denominator(alpha * power[f], 0.0);
  // Vector product of denom
  for (int m = 0; m<M; ++m){
    idx_m = m * K;
    for (int k= 0; k<K; ++k){
      denominator += std::conj(window[get_window_idx(f, m, k)]) * kalman_gain[f_idx_vec + idx_m + k  ];
    }
  }
  // Scalar divition
    for (int m = 0; m<M; m++){
      idx_m = m * K;
      for (int k=0 ; k<K; ++k){
        kalman_gain[f_idx_vec + idx_m + k ] = kalman_gain[f_idx_vec + idx_m + k ] / denominator;
      }
    }
  } 
}

template <size_t num_channels>
  void OnlineWPE<num_channels>::update_power_block( ){
    int num_frames = K + delay +1;
    // Shape of buffer: (M, F (K+delay+1))
    int f_indx = 0;
    double power_sum = 0.0;

    for (int f =0; f < F; ++f){
      f_indx = f * num_frames;
      power_sum = 0.0;
      for (int ch=0; ch< num_channels; ++ch){
        for (int n=0; n<num_frames; ++n){

          power_sum += std::norm(buffer[ch][f_indx+ n]);
        }
      }
      power[f] = power_sum / (num_channels * num_frames);
    }
  }
  
template <size_t num_channels>
void OnlineWPE<num_channels>::update_buffer(const std::array<std::vector<Complex>, num_channels> & new_frame){
  int num_frames = K + delay +1;
  int idx_f = 0;
  for (size_t ch = 0; ch < num_channels; ++ch){

    for (int f=0; f< F; ++f){

      idx_f = f * num_frames; 
      for( int n=0; n< num_frames-1; ++n){ 
        // shift 
        buffer[ch][idx_f + n] = buffer[ch][idx_f + n + 1];
      }
      // Refresh new frame in last index
      buffer[ch][idx_f + num_frames - 1] = new_frame[ch][f]; 
    }

  }    


}