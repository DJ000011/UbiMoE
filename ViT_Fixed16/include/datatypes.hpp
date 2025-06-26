#ifndef __DATATYPES_HPP__
#define __DATATYPES_HPP__

#include <ap_fixed.h>
#include <hls_vector.h>

#include "model.hpp"

typedef ap_fixed<16, 6> fm_t;
typedef ap_fixed<16, 3> wt_linear_t;
typedef ap_fixed<16, 6> wt_attn_bias_t;
typedef ap_fixed<16, 6> wt_bias_t;
typedef ap_fixed<16, 5> wt_norm_t;
typedef ap_fixed<32, 10> fm32_t;

#endif
