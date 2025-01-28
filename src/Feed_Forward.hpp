#ifndef __FF_HPP__
#define __FF_HPP__

#include "dcl.hpp"
#include "moe.hpp"
#include "layernorm.hpp"

extern "C"
{
	void fullconnect(
		unsigned int num_images,
		unsigned int layer,
		patch_blocks_t input[],
		patch_blocks_t output[],
		wt_linear_t vit_weights_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM][FEATURE_DIM],
		wt_bias_t vit_bias_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM],
		wt_linear_t vit_weights_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM][VIT_HIDDEN_DIM],
		wt_bias_t vit_bias_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM],
		wt_linear_t moe_w_gate[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM],
		wt_linear_t moe_weights_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM][FEATURE_DIM],
		wt_bias_t moe_bias_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM],
		wt_linear_t moe_weights_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM][EXPERT_HIDDEN_DIM],
		wt_bias_t moe_bias_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM],
		wt_norm_t norm_weights[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
		wt_bias_t norm_bias[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM]

	);
}

#endif
