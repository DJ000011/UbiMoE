#ifndef __LAYERNORM_HPP__
#define __LAYERNORM_HPP__

#include "dcl.hpp"

enum LayerNorm {
    NORM_1 = 0,
    NORM_2 = 1,
    NUM_LAYER_NORMS
};

extern fm_t norm_eps;
extern wt_norm_t norm1_weights[FEATURE_DIM];
extern wt_norm_t norm2_weights[FEATURE_DIM];
extern wt_bias_t norm1_bias[FEATURE_DIM];
extern wt_bias_t norm2_bias[FEATURE_DIM];

void load_norms(
    wt_norm_t norm_weights[NUM_LAYER_NORMS][FEATURE_DIM],
    wt_bias_t norm_bias[NUM_LAYER_NORMS][FEATURE_DIM],
    wt_norm_t weights[FEATURE_DIM],
    wt_bias_t bias[FEATURE_DIM]
);
void compute_norm(
    patch_blocks_t x, 
    patch_blocks_t out, 
    wt_norm_t weights[FEATURE_DIM], 
    wt_bias_t bias[FEATURE_DIM]
);
#include "layernorm.cpp"
#endif
