#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include "dcl.hpp"
#include "attention.hpp"
#include "layernorm.hpp"
#include "linear.hpp"
#include "add.hpp"

enum AttentionLinear {
    ATTN_Q = 0,
    ATTN_K = 1,
    ATTN_V = 2,
    NUM_ATTN_LINEAR
};

extern "C"
{
void ViT_compute(
	    unsigned int num_images,
		unsigned int layer,
	    patch_blocks_t x[],
		patch_blocks_t output[],
		patch_blocks_t x_norm,
		patch_blocks_t Q_linear_ping,
		patch_blocks_t Q_linear_pong,
		patch_blocks_t K_linear_ping,
		patch_blocks_t K_linear_pong,
		patch_blocks_t V_linear_ping,
		patch_blocks_t V_linear_pong,
		patch_blocks_t attn,
		patch_blocks_t PROJ_linear,
	    wt_linear_t attn_weights[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM][FEATURE_DIM],
	    wt_attn_bias_t attn_bias[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM],
		wt_linear_t proj_weights[NUM_LAYERS][FEATURE_DIM][FEATURE_DIM],
		wt_attn_bias_t proj_bias[NUM_LAYERS][FEATURE_DIM],
	    wt_norm_t norm_weights_l1[NUM_LAYERS][FEATURE_DIM],
	    wt_bias_t norm_bias_l1[NUM_LAYERS][FEATURE_DIM]
);
}

#endif
