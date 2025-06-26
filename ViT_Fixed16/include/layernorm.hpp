#ifndef __LAYERNORM_HPP__
#define __LAYERNORM_HPP__

#include "dcl.hpp"

extern fm_t norm_eps;
extern wt_norm_t norm1_weights[FEATURE_DIM];
extern wt_norm_t norm2_weights[FEATURE_DIM];
extern wt_bias_t norm1_bias[FEATURE_DIM];
extern wt_bias_t norm2_bias[FEATURE_DIM];

void load_norms(
    wt_norm_t norm_weights[FEATURE_DIM],
    wt_bias_t norm_bias[FEATURE_DIM],
    wt_norm_t weights[FEATURE_DIM],
    wt_bias_t bias[FEATURE_DIM]
);
void compute_norm(
    patch_blocks_t x, 
    patch_blocks_t out, 
    wt_norm_t weights[FEATURE_DIM], 
    wt_bias_t bias[FEATURE_DIM]
);

#include "../src/layernorm.cpp"
#endif
