#include "../include/ViT_compute.hpp"

void QKV_gen(
    patch_blocks_t input,
    patch_blocks_t x_norm,
	patch_blocks_t Q_linear,
	patch_blocks_t K_linear,
	patch_blocks_t V_linear,
    wt_norm_t* norm1_weights,
    wt_bias_t* norm1_bias,
    wt_linear_block_t linear_weights_attn[NUM_ATTN_LINEAR][ceildiv(QKV_LINEAR_DIM_PRODUCT, LINEAR_IN_SIZE * LINEAR_OUT_SIZE)],
    wt_bias_block_t linear_bias_attn[NUM_ATTN_LINEAR][ceildiv(FEATURE_DIM, LINEAR_OUT_SIZE)],
    unsigned int feature_dim)
{
#pragma HLS inline off

	compute_norm(input, x_norm, norm1_weights, norm1_bias);
    compute_linear_single(reinterpret_cast<fm_block_t*>(Q_linear), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_Q], linear_bias_attn[ATTN_Q], feature_dim, feature_dim,false);
    compute_linear_single(reinterpret_cast<fm_block_t*>(K_linear), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_K], linear_bias_attn[ATTN_K], feature_dim, feature_dim,false);
    compute_linear_single(reinterpret_cast<fm_block_t*>(V_linear), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_V], linear_bias_attn[ATTN_V], feature_dim, feature_dim,false);

}

void attn_compute(
	patch_blocks_t Q_linear,
	patch_blocks_t K_linear,
	patch_blocks_t V_linear,
	patch_blocks_t PROJ_linear,
	patch_blocks_t attn,
    wt_linear_block_t* proj_weights,
    wt_bias_block_t* proj_bias,
    unsigned int feature_dim)
{
#pragma HLS inline off
	compute_attn(Q_linear, K_linear, V_linear, attn);
	compute_linear_single(reinterpret_cast<fm_block_t*>(PROJ_linear), reinterpret_cast<fm_block_t*>(attn), proj_weights, proj_bias, feature_dim, feature_dim,false);
}



extern "C" {
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
)
{
    #pragma HLS interface m_axi depth=1 port=x offset=slave bundle=in max_widen_bitwidth=AXI_XFER_BIT_WIDTH
	#pragma HLS interface m_axi depth=1 port=attn offset=slave bundle=in1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH

    #pragma HLS interface m_axi depth=1 port=output offset=slave bundle=out max_widen_bitwidth=AXI_XFER_BIT_WIDTH
	#pragma HLS interface m_axi depth=1 port=x_norm offset=slave bundle=out1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH

	#pragma HLS interface m_axi depth=1 port=PROJ_linear offset=slave bundle=temp max_widen_bitwidth=AXI_XFER_BIT_WIDTH

	#pragma HLS interface m_axi depth=1 port=Q_linear_ping offset=slave bundle=attn_in max_widen_bitwidth=AXI_XFER_BIT_WIDTH
	#pragma HLS interface m_axi depth=1 port=Q_linear_pong offset=slave bundle=attn_out max_widen_bitwidth=AXI_XFER_BIT_WIDTH

    #pragma HLS interface m_axi depth=1 port=K_linear_ping offset=slave bundle=attn_in1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
	#pragma HLS interface m_axi depth=1 port=K_linear_pong offset=slave bundle=attn_out1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH

	#pragma HLS interface m_axi depth=1 port=V_linear_ping offset=slave bundle=attn_in2 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
	#pragma HLS interface m_axi depth=1 port=V_linear_pong offset=slave bundle=attn_out2 max_widen_bitwidth=AXI_XFER_BIT_WIDTH

    #pragma HLS interface m_axi depth=1 port=attn_weights offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=attn_bias offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
	#pragma HLS interface m_axi depth=1 port=proj_weights offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=proj_bias offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=norm_weights_l1 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=norm_bias_l1 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH

//    patch_blocks_t Q_linear,K_linear,V_linear;
//    patch_blocks_t x_norm,attn,PROJ_linear;

    wt_linear_block_t linear_weights_attn[NUM_ATTN_LINEAR][ceildiv(QKV_LINEAR_DIM_PRODUCT, LINEAR_IN_SIZE * LINEAR_OUT_SIZE)];
    wt_bias_block_t linear_bias_attn[NUM_ATTN_LINEAR][ceildiv(FEATURE_DIM, LINEAR_OUT_SIZE)];
    wt_linear_block_t linear_weights_proj[ceildiv(QKV_LINEAR_DIM_PRODUCT, LINEAR_IN_SIZE * LINEAR_OUT_SIZE)];
    wt_bias_block_t linear_bias_proj[ceildiv(FEATURE_DIM, LINEAR_OUT_SIZE)];

    wt_norm_t norm1_weights[FEATURE_DIM];
    wt_bias_t norm1_bias[FEATURE_DIM];

    attn_scale = 0.125;
    norm_eps = 1e-12;

    bool flag = true;

    load_norms(norm_weights_l1[layer], norm_bias_l1[layer], norm1_weights, norm1_bias);
   
    for(int i = 0 ; i < NUM_ATTN_LINEAR; i++)
    {
       load_linear_weights(linear_weights_attn[i], reinterpret_cast<wt_linear_t*>(attn_weights[layer][i]), FEATURE_DIM, FEATURE_DIM);
       load_linear_bias(linear_bias_attn[i], reinterpret_cast<wt_attn_bias_t*>(attn_bias[layer][i]), FEATURE_DIM);
    }
    load_linear_weights(linear_weights_proj, reinterpret_cast<wt_linear_t*>(proj_weights[layer]), FEATURE_DIM, FEATURE_DIM);
    load_linear_bias(linear_bias_proj, reinterpret_cast<wt_attn_bias_t*>(proj_bias[layer]), FEATURE_DIM);

    QKV_gen(x[0],x_norm,Q_linear_ping,K_linear_ping,V_linear_ping,norm1_weights,norm1_bias,linear_weights_attn,linear_bias_attn,FEATURE_DIM);

    for (unsigned int image = 1; image < num_images; image++)
    {
    	if(flag)
    	{
    		flag = !flag;
    		attn_compute(Q_linear_ping, K_linear_ping, V_linear_ping, PROJ_linear, attn, linear_weights_proj, linear_bias_proj, FEATURE_DIM);
    		QKV_gen(x[image], x_norm, Q_linear_pong, K_linear_pong, V_linear_pong, norm1_weights, norm1_bias, linear_weights_attn, linear_bias_attn, FEATURE_DIM);

    	}
    	else
    	{
    		flag = !flag;
    		attn_compute(Q_linear_pong,K_linear_pong,V_linear_pong, PROJ_linear, attn, linear_weights_proj,linear_bias_proj,FEATURE_DIM);
    		QKV_gen(x[image],x_norm,Q_linear_ping,K_linear_ping,V_linear_ping,norm1_weights,norm1_bias,linear_weights_attn,linear_bias_attn,FEATURE_DIM);

    	}
    	compute_add(x[image - 1], PROJ_linear, output[image - 1]);
    }

    if(flag){
    	attn_compute(Q_linear_ping, K_linear_ping, V_linear_ping, PROJ_linear, attn, linear_weights_proj, linear_bias_proj, FEATURE_DIM);
    }
    else{
    	attn_compute(Q_linear_pong, K_linear_pong, V_linear_pong, PROJ_linear, attn, linear_weights_proj, linear_bias_proj, FEATURE_DIM);
    }
    compute_add(x[num_images - 1], PROJ_linear, output[num_images - 1]);
  }
//    FOR_EACH(image, num_images)
//        {
//           compute_norm(x[image], x_norm, norm1_weights, norm1_bias);
//           if(flag){
//        	   compute_attn(Q_linear_pong, K_linear_pong,V_linear_pong,attn);
//
//        	   compute_linear_single(reinterpret_cast<fm_block_t*>(Q_linear_ping), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_Q], linear_bias_attn[ATTN_Q], FEATURE_DIM, FEATURE_DIM);
//        	   compute_linear_single(reinterpret_cast<fm_block_t*>(K_linear_ping), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_K], linear_bias_attn[ATTN_K], FEATURE_DIM, FEATURE_DIM);
//        	   compute_linear_single(reinterpret_cast<fm_block_t*>(V_linear_ping), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_V], linear_bias_attn[ATTN_V], FEATURE_DIM, FEATURE_DIM);
//           }
//           else{
//        	   compute_attn(Q_linear_ping, K_linear_ping, V_linear_ping, attn);
//
//               compute_linear_single(reinterpret_cast<fm_block_t*>(Q_linear_pong), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_Q], linear_bias_attn[ATTN_Q], FEATURE_DIM, FEATURE_DIM);
//               compute_linear_single(reinterpret_cast<fm_block_t*>(K_linear_pong), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_K], linear_bias_attn[ATTN_K], FEATURE_DIM, FEATURE_DIM);
//               compute_linear_single(reinterpret_cast<fm_block_t*>(V_linear_pong), reinterpret_cast<fm_block_t*>(x_norm), linear_weights_attn[ATTN_V], linear_bias_attn[ATTN_V], FEATURE_DIM, FEATURE_DIM);
//
//           }
//           flag = !flag;
//           compute_linear_single(reinterpret_cast<fm_block_t*>(PROJ_linear), reinterpret_cast<fm_block_t*>(attn), linear_weights_attn[ATTN_PROJ], linear_bias_attn[ATTN_PROJ], FEATURE_DIM, FEATURE_DIM);
//           compute_add(x[image], PROJ_linear, output[image]);
//        }
}

