#include <vector> 
#include <complex> 
#include <array> 

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

  std::array<std::vector<Complex>, num_channels> buffer;

  // Define the constructor
  OnlineWPE( float a, int b, int c, int d, int e, double power_estimate) {
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

    inv_cov.resize(inv_cov_len, Complex(0.0,0.0)); //shape (F, MK, MK)
    power.resize(F , power_estimate); 

    for( int f = 0; f <F; f++ ) {
    diag_idx_f = f * cov_matrix_size * cov_matrix_size;

      for (int i = 0; i<cov_matrix_size; i++){
        diag_idx = diag_idx_f + (cov_matrix_size + 1) * i;
        inv_cov[diag_idx] = Complex( 1.0, 0.0 );
      }
    }                             

    // Init buffer
    for (size_t ch = 0; ch < num_channels; ++ch){
      buffer[c].resize( F * ( K + delay + 1));
    }

    void _update_buffer(const std::array<std::vector<Complex>, num_channels> new_frame : ){
      for (size_t ch = 0, ch < ch; ch++)





    }
  }




};




int main(){





}