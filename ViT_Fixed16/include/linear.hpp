#ifndef __LINEAR_HPP__
#define __LINEAR_HPP__

#include "dcl.hpp"
#include "hls_stream.h"
#include "gelu.hpp"

typedef hls::vector<hls::vector<wt_linear_t, LINEAR_IN_SIZE>, LINEAR_OUT_SIZE> wt_linear_block_t;
typedef hls::vector<fm_t,LINEAR_OUT_SIZE> wt_bias_block_t;


void load_linear_weights(
    wt_linear_block_t weights_dst[],
    wt_linear_t weights_src[],
    unsigned int out_dim,
    unsigned int in_dim
);
template<typename T>
void load_linear_bias(
    wt_bias_block_t bias_dst[],
    T bias_src[],
    unsigned int out_dim
);

void compute_linear_single(
    fm_block_t dst[],
    fm_block_t src[],
    const wt_linear_block_t weights[],
    const wt_bias_block_t bias[],
    unsigned int out_dim,
    unsigned int in_dim
);
#include "../src/linear.cpp"
#endif
