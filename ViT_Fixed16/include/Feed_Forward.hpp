#ifndef __FF_HPP__
#define __FF_HPP__

#include "dcl.hpp"
#include "layernorm.hpp"
#include "linear.hpp"
#include "add.hpp"

extern "C"
{
	void fullconnect(
		unsigned int num_images,
		unsigned int layer,
		patch_blocks_t input[],
		patch_blocks_t output[],
		patch_blocks_t x_norm2,
		patch_blocks_t tmp,
		fm_block_t tmp_hidden_ping[NUM_PATCHES * ceildiv(VIT_HIDDEN_DIM, FEATURE_BLOCK_SIZE)],
		fm_block_t tmp_hidden_pong[NUM_PATCHES * ceildiv(VIT_HIDDEN_DIM, FEATURE_BLOCK_SIZE)],
		wt_linear_t vit_weights_l1[NUM_LAYERS][VIT_HIDDEN_DIM][FEATURE_DIM],
		wt_bias_t vit_bias_l1[NUM_LAYERS][VIT_HIDDEN_DIM],
		wt_linear_t vit_weights_l2[NUM_LAYERS][FEATURE_DIM][VIT_HIDDEN_DIM],
		wt_bias_t vit_bias_l2[NUM_LAYERS][FEATURE_DIM],
		wt_norm_t norm_weights_l2[NUM_LAYERS][FEATURE_DIM],
		wt_bias_t norm_bias_l2[NUM_LAYERS][FEATURE_DIM]
	);
}

#endif
