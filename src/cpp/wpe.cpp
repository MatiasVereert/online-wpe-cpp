#include <vector> 
#include <complex> 
#include <array> 
#include <algorithm>

using Complex = std::complex<double>;

template <size_t num_channels>

class OnlineWPE {
public:
  float alpha; 
  int K;  // taps 
  int delay;
  int M ; // Mics dimension 
  int F ; // Frecuency bins

  std::vector<Complex> inv_cov;
  std::vector<Complex> filter_taps;
  std::vector<double> power;
  std::vector<Complex> kalman_gain;
  // Aux
  std::vector<Complex> inv_cov_right_term;

  std::array<std::vector<Complex>, num_channels> buffer;

  // Define the constructor

  OnlineWPE( float a, int b, int c, int d, int e, double power_estimate); 

  void update_buffer(const std::array<std::vector<Complex>, num_channels> & new_frame);
  void update_power_block();
  void update_kalman_gain();
  void update_inv_cov();

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

  // Auxiliar
  inv_cov_right_term.resize( M*K*M*K, Complex(0.0,0.0) )

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


template <size_t numb_channels>
void OnlineWPE<numb_channels>::update_inv_cov(){
  // Clean aux vector
  std::fill(inv_cov_right_term.begin();inv_cov_right_term.end(), Complex(0.0,0.0));

  // Define stridles



}

template <size_t num_channels>
void OnlineWPE<num_channels>::update_kalman_gain(){
  int num_frames = K + delay +1;
  int len_MK = M * K;
  int len_KMK = K * M * K;
  int m_left_idx = 0;
  int k_left_idx = 0;
  int m_right_idx = 0;
  int f_idx_buff = 0;
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
    f_idx_buff  = f * num_frames;
    
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
                                                              * buffer[m_right][ f_idx_buff + K - 1 -k_right];
          }       
        }
      }
    }
  Complex denominator(alpha * power[f], 0.0);
  // Vector product of denom
  for (int m = 0; m<M; ++m){
    idx_m = m * K;
    for (int k= 0; k<K; ++k){
      denominator += std::conj(buffer[m][f_idx_buff + K - 1 - k]) * kalman_gain[f_idx_vec + idx_m + k  ];
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