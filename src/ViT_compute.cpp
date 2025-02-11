
#include "add.hpp"
#include "conv.hpp"
#include "kernel.hpp"

extern "C" {
void ViT_compute(
    unsigned int num_images,
	unsigned int layer,
    patch_blocks_t x[],
    wt_linear_t attn_weights[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM][FEATURE_DIM],
    wt_attn_bias_t attn_bias[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM],
    wt_norm_t norm_weights[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
    wt_bias_t norm_bias[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
	patch_blocks_t norm2_x[]
)
{
	attn_scale = 0.125;
    norm_eps = 1e-6;
    patch_blocks_t Q_linear,K_linear,V_linear;
    patch_blocks_t x_norm,attn,PROJ_linear;
    
    #pragma HLS interface m_axi depth=1 port=x offset=slave bundle=in max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=norm2_x offset=slave bundle=out max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=attn_weights offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=attn_bias offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=norm_weights offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=norm_bias offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH


    load_norms(norm_weights[layer][NORM_1], norm_bias[layer][NORM_1],norm1_weights, norm1_bias);
   
    for(int i=0; i< NUM_ATTN_LINEAR;i++){
           load_linear_weights(linear_weights_attn[i], reinterpret_cast<wt_linear_t*>(attn_weights[layer][i]), FEATURE_DIM, FEATURE_DIM);
           load_linear_bias(linear_bias_attn[i], reinterpret_cast<wt_attn_bias_t*>(attn_bias[layer][i]), FEATURE_DIM);
    	}

        FOR_EACH(image, num_images)
        {
//the following code we present is running sequentially for simplicity, \
and can be parallelized by using Double Buffering between QKV generation and attention computation
           compute_norm(x[image], x_norm, norm1_weights, norm1_bias);
           compute_linear_single(reinterpret_cast<fm_block_t*>(Q_linear), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_Q], linear_bias_attn[ATTN_Q], FEATURE_DIM, FEATURE_DIM);
           compute_linear_single(reinterpret_cast<fm_block_t*>(K_linear), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_K], linear_bias_attn[ATTN_K], FEATURE_DIM, FEATURE_DIM);
           compute_linear_single(reinterpret_cast<fm_block_t*>(V_linear), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_V], linear_bias_attn[ATTN_V], FEATURE_DIM, FEATURE_DIM);
           compute_attn(Q_linear, K_linear,V_linear,attn);
           compute_linear_single(reinterpret_cast<fm_block_t*>(PROJ_linear), reinterpret_cast<fm_block_t*>(attn), linear_weights_attn[ATTN_PROJ], linear_bias_attn[ATTN_PROJ], FEATURE_DIM, FEATURE_DIM);
           compute_add(x[image], PROJ_linear, norm2_x[image]);
        }
    }
}

