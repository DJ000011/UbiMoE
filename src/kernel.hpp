#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__


#include "dcl.hpp"
#include "attention.hpp"
#include "layernorm.hpp"

extern "C"
{
void ViT_compute(
    unsigned int num_images,
	unsigned int layer,
    patch_blocks_t x[],
    wt_linear_t attn_weights[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM][FEATURE_DIM],
    wt_attn_bias_t attn_bias[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM],
    wt_norm_t norm_weights[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
    wt_bias_t norm_bias[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
	patch_blocks_t norm2_x[]
);
}

#endif
