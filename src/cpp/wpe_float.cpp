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

    inline size_t get_pred_idx( int f, int m) const {
      // Prediction / output, layout (F, M)
      return f * M + m;
    }

    inline size_t get_buffer_raw_idx( int f, int n) const {
      // Raw buffer position, layout (F, K + delay + 1). n counts from the
      // oldest frame (n=0) to the newest (n=K+delay). Used by the buffer
      // shift and the power estimate.
      return f * (K + delay + 1) + n;
    }

    inline size_t get_buffer_idx( int f, int k_tap) const {
      // taps axis is mirrored to align with math
      // buffer has (past samples)<(present)
      // this idx returns (past samples)>(present)
      return get_buffer_raw_idx(f, K - 1 - k_tap);
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

    OnlineWPE( double a, int b, int c, int d, int e, double power_estimate);

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
OnlineWPE<num_channels>::OnlineWPE( double a, int b, int c, int d, int e, double power_estimate) {
  alpha = a;
  K = b;
  delay = c;
  M = d;
  F = e;
  
  filter_taps.resize(F * M * K * M, Complex(0.0,0.0));

  // Init. cov_matrix as Identity
  int inv_cov_len = F*M*K*M*K;

  // Variables
  inv_cov.resize(inv_cov_len, Complex(0.0,0.0)); //shape (F, MK, MK)
  power.resize(F , power_estimate); 
  kalman_gain.resize( F* K* M, Complex(0.0,0.0));
  prediction.resize(F* M, Complex(0.0,0.0) );
  window.resize( F* M* K, Complex(0.0,0.0));

  // Init inv_cov as the identity per frequency bin (diagonal = 1).
  for (int f = 0; f < F; ++f){
    for (int m = 0; m < M; ++m){
      for (int k = 0; k < K; ++k){
        inv_cov[get_cov_idx(f, m, k, m, k)] = Complex(1.0, 0.0);
      }
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
      prediction[get_pred_idx(f, m0)] = new_frame[m0][f] - inner_summ;
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

          filter_taps[taps_idx] += kalman_gain[kalman_idx] * std::conj(prediction[get_pred_idx(f, m1)]);
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

  // Clean past kalman gain
  std::fill(kalman_gain.begin(), kalman_gain.end(), Complex(0.0, 0.0));

  for (int f = 0; f < F; ++f){

    // Nominator: kalman_gain = R_inv dot window   (per bin, shape M*K)
    // window is the mirrored buffer snapshot (see get_window_idx / get_buffer_idx).
    for (int m_left = 0; m_left < M; ++m_left){
      for (int k_left = 0; k_left < K; ++k_left){
        for (int m_right = 0; m_right < M; ++m_right){
          for (int k_right = 0; k_right < K; ++k_right){
            kalman_gain[get_kalman_gain_idx(f, m_left, k_left)] +=
                  inv_cov[get_cov_idx(f, m_left, k_left, m_right, k_right)]
                * window[get_window_idx(f, m_right, k_right)];
          }
        }
      }
    }

    // Denominator: alpha * power + window^H dot nominator
    Complex denominator(alpha * power[f], 0.0);
    for (int m = 0; m < M; ++m){
      for (int k = 0; k < K; ++k){
        denominator += std::conj(window[get_window_idx(f, m, k)])
                     * kalman_gain[get_kalman_gain_idx(f, m, k)];
      }
    }

    // Scalar division
    for (int m = 0; m < M; ++m){
      for (int k = 0; k < K; ++k){
        kalman_gain[get_kalman_gain_idx(f, m, k)] =
            kalman_gain[get_kalman_gain_idx(f, m, k)] / denominator;
      }
    }
  }
}

template <size_t num_channels>
  void OnlineWPE<num_channels>::update_power_block( ){
    int num_frames = K + delay +1;
    // Shape of buffer: (M, F, (K+delay+1))
    double power_sum = 0.0;

    for (int f =0; f < F; ++f){
      power_sum = 0.0;
      for (size_t ch=0; ch< num_channels; ++ch){
        for (int n=0; n<num_frames; ++n){
          power_sum += std::norm(buffer[ch][get_buffer_raw_idx(f, n)]);
        }
      }
      power[f] = power_sum / (num_channels * num_frames);
    }
  }
  
template <size_t num_channels>
void OnlineWPE<num_channels>::update_buffer(const std::array<std::vector<Complex>, num_channels> & new_frame){
  int num_frames = K + delay +1;
  for (size_t ch = 0; ch < num_channels; ++ch){

    for (int f=0; f< F; ++f){

      for( int n=0; n< num_frames-1; ++n){
        // shift
        buffer[ch][get_buffer_raw_idx(f, n)] = buffer[ch][get_buffer_raw_idx(f, n + 1)];
      }
      // Refresh new frame in last index
      buffer[ch][get_buffer_raw_idx(f, num_frames - 1)] = new_frame[ch][f];
    }

  }


}