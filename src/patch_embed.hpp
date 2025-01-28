#ifndef _PATCH_EMBED_HPP_
#define _PATCH_EMBED_HPP_


#include "conv.hpp"
#include <hls_stream.h>


extern "C" {    
    void patch_embed(
        unsigned int num_images,
        image_t images[],
        patch_blocks_t x[],
        wt_patch_embed_t patch_embed_weights_load[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH],
        wt_bias_t patch_embed_bias_load[FEATURE_DIM],
        patch_blocks_t pos_embed
    );
}

#endif