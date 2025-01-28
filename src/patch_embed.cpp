# include "patch_embed.hpp"

extern "C"
{
    void patch_embed(
        unsigned int num_images,
        image_t images[],
        patch_blocks_t x[],
        wt_patch_embed_t patch_embed_weights_load[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH],
        wt_bias_t patch_embed_bias_load[FEATURE_DIM],
        patch_blocks_t pos_embed
    )
    {
        #pragma HLS interface m_axi depth=1 port=images offset=slave max_widen_bitwidth=AXI_XFER_BIT_WIDTH
        #pragma HLS interface m_axi depth=1 port=x offset=slave max_widen_bitwidth=AXI_XFER_BIT_WIDTH
        #pragma HLS interface m_axi depth=1 port=patch_embed_weights_load offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
        #pragma HLS interface m_axi depth=1 port=patch_embed_bias_load offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
        #pragma HLS interface m_axi depth=1 port=pos_embed offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH

        load_one_time_weights(patch_embed_weights_load, patch_embed_bias_load);

        FOR_EACH(image, num_images)
        {
            compute_patch_embed(images[image], x[image], pos_embed);
        }
    }
}