
#include "Feed_Forward.hpp"

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
	)
	{
	norm_eps = 1e-6;

#pragma HLS interface m_axi depth = 1 port = input offset = slave bundle = in max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = output offset = slave bundle = out max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = moe_w_gate offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = moe_weights_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = moe_bias_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = moe_weights_l2 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = moe_bias_l2 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = vit_weights_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = vit_bias_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = vit_weights_l2 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = vit_bias_l2 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth=1 port=norm_weights offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth=1 port=norm_bias offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH

		load_norms(norm_weights[layer][NORM_2], norm_bias[layer][NORM_2],norm2_weights, norm2_bias);

		if (layer % 2 == 0)
		{
			load_linear_weights(linear_weights_ping, reinterpret_cast<wt_linear_t *>(vit_weights_l1[layer / 2]), VIT_HIDDEN_DIM, FEATURE_DIM);
			load_linear_bias(linear_bias_ping, reinterpret_cast<wt_bias_t *>(vit_bias_l1[layer / 2]), VIT_HIDDEN_DIM);
			load_linear_weights(linear_weights_pong, reinterpret_cast<wt_linear_t *>(vit_weights_l2[layer / 2]), FEATURE_DIM, VIT_HIDDEN_DIM);
			load_linear_bias(linear_bias_pong, reinterpret_cast<wt_bias_t *>(vit_bias_l2[layer / 2]), FEATURE_DIM);
		}
		else
		{
			load_w_gate(moe_w_gate[layer / 2]);
		}
		
		
		FOR_EACH(image, num_images)
		{
			fm_block_t tmp_hidden[NUM_PATCHES * ceildiv(max(VIT_HIDDEN_DIM, EXPERT_HIDDEN_DIM), FEATURE_BLOCK_SIZE)];
			patch_blocks_t tmp, x_norm2;

			compute_norm(input[image], x_norm2,norm2_weights, norm2_bias);

			if (layer % 2 == 0)
			{
				compute_linear(tmp_hidden, reinterpret_cast<fm_block_t *>(x_norm2), linear_weights_ping, linear_bias_ping, VIT_HIDDEN_DIM, FEATURE_DIM, 0, true, false, false);
				compute_linear(reinterpret_cast<fm_block_t *>(tmp), tmp_hidden, linear_weights_pong, linear_bias_pong, FEATURE_DIM, VIT_HIDDEN_DIM, 0, false, false, false);
			}

			else
			{
				compute_moe(
					x_norm2,
					tmp,
					tmp_hidden,
					moe_weights_l1[layer / 2],
					moe_bias_l1[layer / 2],
					moe_weights_l2[layer / 2],
					moe_bias_l2[layer / 2]);
			}
			compute_add(input[image], tmp, output[image]);
		}
	}
}
