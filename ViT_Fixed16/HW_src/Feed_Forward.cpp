#include "../include/Feed_Forward.hpp"



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
	)
	{
#pragma HLS interface m_axi depth = 1 port = input offset = slave bundle = in max_widen_bitwidth = AXI_XFER_BIT_WIDTH

#pragma HLS interface m_axi depth = 1 port = output offset = slave bundle = out max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = x_norm2 offset = slave bundle = out max_widen_bitwidth = AXI_XFER_BIT_WIDTH

#pragma HLS interface m_axi depth = 1 port = tmp offset = slave bundle = temp max_widen_bitwidth = AXI_XFER_BIT_WIDTH

#pragma HLS interface m_axi depth = 1 port = tmp_hidden_ping offset = slave bundle = ping max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = tmp_hidden_pong offset = slave bundle = pong max_widen_bitwidth = AXI_XFER_BIT_WIDTH

#pragma HLS interface m_axi depth = 1 port = vit_weights_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = vit_bias_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = vit_weights_l2 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth = 1 port = vit_bias_l2 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth=1 port=norm_weights_l2 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
#pragma HLS interface m_axi depth=1 port=norm_bias_l2 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH

		wt_linear_block_t linear_weights_ping[ceildiv(MAX_LINEAR_DIM_PRODUCT, LINEAR_IN_SIZE * LINEAR_OUT_SIZE)];
	 	wt_bias_block_t linear_bias_ping[ceildiv(MAX_LINEAR_OUT_DIM, LINEAR_OUT_SIZE)];
		wt_linear_block_t linear_weights_pong[ceildiv(MAX_LINEAR_DIM_PRODUCT, LINEAR_IN_SIZE * LINEAR_OUT_SIZE)];
	 	wt_bias_block_t linear_bias_pong[ceildiv(MAX_LINEAR_OUT_DIM, LINEAR_OUT_SIZE)];
		wt_norm_t norm2_weights[FEATURE_DIM];
		wt_bias_t norm2_bias[FEATURE_DIM];

		norm_eps = 1e-6;

		bool flag = true;

		load_norms(norm_weights_l2[layer], norm_bias_l2[layer],norm2_weights, norm2_bias);

		load_linear_weights(linear_weights_ping, reinterpret_cast<wt_linear_t *>(vit_weights_l1[layer]), VIT_HIDDEN_DIM, FEATURE_DIM);
		load_linear_bias(linear_bias_ping, reinterpret_cast<wt_bias_t *>(vit_bias_l1[layer]), VIT_HIDDEN_DIM);
		load_linear_weights(linear_weights_pong, reinterpret_cast<wt_linear_t *>(vit_weights_l2[layer]), FEATURE_DIM, VIT_HIDDEN_DIM);
		load_linear_bias(linear_bias_pong, reinterpret_cast<wt_bias_t *>(vit_bias_l2[layer]), FEATURE_DIM);
		
		//initial compute
		compute_norm(input[0], x_norm2, norm2_weights, norm2_bias);
		compute_linear_single(tmp_hidden_ping, reinterpret_cast<fm_block_t *>(x_norm2), linear_weights_ping, linear_bias_ping, VIT_HIDDEN_DIM, FEATURE_DIM,true);

		for (unsigned int image = 1; image < num_images ; image++)
		{
			compute_norm(input[image], x_norm2, norm2_weights, norm2_bias);

			if(flag){
				flag = !flag;
				compute_linear_single(reinterpret_cast<fm_block_t *>(tmp), tmp_hidden_ping, linear_weights_pong, linear_bias_pong, FEATURE_DIM, VIT_HIDDEN_DIM, false);
				compute_linear_single(tmp_hidden_pong, reinterpret_cast<fm_block_t *>(x_norm2), linear_weights_ping, linear_bias_ping, VIT_HIDDEN_DIM, FEATURE_DIM, true);
			}
			else
			{
				flag = !flag;
				compute_linear_single(reinterpret_cast<fm_block_t *>(tmp), tmp_hidden_pong , linear_weights_pong, linear_bias_pong, FEATURE_DIM, VIT_HIDDEN_DIM, false);
				compute_linear_single(tmp_hidden_ping, reinterpret_cast<fm_block_t *>(x_norm2), linear_weights_ping, linear_bias_ping, VIT_HIDDEN_DIM, FEATURE_DIM, true);
			}
			compute_add(input[image - 1], tmp, output[image - 1]);
		}
		if(flag){
			compute_linear_single(reinterpret_cast<fm_block_t *>(tmp), tmp_hidden_ping, linear_weights_pong, linear_bias_pong, FEATURE_DIM, VIT_HIDDEN_DIM,false);
		}
		else{
			compute_linear_single(reinterpret_cast<fm_block_t *>(tmp), tmp_hidden_pong, linear_weights_pong, linear_bias_pong, FEATURE_DIM, VIT_HIDDEN_DIM,false);
		}
		compute_add(input[num_images - 1], tmp, output[num_images - 1]);
	}
}


//void compute_unit1(
//    patch_blocks_t input,
//    patch_blocks_t x_norm2,
//    fm_block_t* tmp_hidden,
//    wt_norm_t* norm2_weights,
//    wt_bias_t* norm2_bias,
//    wt_linear_block_t* linear_weights,
//    wt_bias_block_t* linear_bias,
//    unsigned int hidden_dim,
//    unsigned int feature_dim)
//{
//#pragma HLS inline off
//    compute_norm(input, x_norm2, norm2_weights, norm2_bias);
//    compute_linear_single(tmp_hidden, reinterpret_cast<fm_block_t*>(x_norm2),linear_weights, linear_bias, hidden_dim, feature_dim);
//}
//
//void compute_unit2(
//    patch_blocks_t input,
//    fm_block_t* tmp_hidden,
//    patch_blocks_t tmp,
//    patch_blocks_t output,
//    wt_linear_block_t* linear_weights,
//    wt_bias_block_t* linear_bias,
//    unsigned int feature_dim,
//    unsigned int hidden_dim)
//{
//#pragma HLS inline off
//    compute_linear_single(reinterpret_cast<fm_block_t*>(tmp), tmp_hidden, linear_weights, linear_bias, feature_dim, hidden_dim);
//    compute_add(input, tmp, output);
//}
//		compute_unit1(input[0], x_norm2, tmp_hidden_ping, norm2_weights, norm2_bias,linear_weights_ping, linear_bias_ping,VIT_HIDDEN_DIM, FEATURE_DIM);
//
//		for (unsigned int image = 0; image < num_images - 1; image++)
//		{
//			if(flag){
//				compute_unit2(input[image],tmp_hidden_ping,tmp,output[image],linear_weights_pong,linear_bias_pong,FEATURE_DIM,VIT_HIDDEN_DIM);
//				compute_unit1(input[image + 1], x_norm2, tmp_hidden_pong, norm2_weights, norm2_bias,linear_weights_ping, linear_bias_ping,VIT_HIDDEN_DIM, FEATURE_DIM);
//			}
//			else
//			{
//				compute_unit2(input[image],tmp_hidden_pong,tmp,output[image],linear_weights_pong,linear_bias_pong,FEATURE_DIM,VIT_HIDDEN_DIM);
//				compute_unit1(input[image + 1], x_norm2, tmp_hidden_ping, norm2_weights, norm2_bias,linear_weights_ping, linear_bias_ping,VIT_HIDDEN_DIM, FEATURE_DIM);
//			}
//			flag = !flag;
//		}
//
//		if(flag){
//			compute_unit2(input[num_images - 1],tmp_hidden_ping,tmp,output[num_images - 1],linear_weights_pong,linear_bias_pong,FEATURE_DIM,VIT_HIDDEN_DIM);
//		}
//		else{
//			compute_unit2(input[num_images - 1],tmp_hidden_pong,tmp,output[num_images - 1],linear_weights_pong,linear_bias_pong,FEATURE_DIM,VIT_HIDDEN_DIM);
//		}
