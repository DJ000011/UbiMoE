#ifndef __ATTENTION_HPP__
#define __ATTENTION_HPP__

#include "dcl.hpp"


extern fm_t attn_scale;

void compute_attn(patch_blocks_t q, patch_blocks_t k, patch_blocks_t v,patch_blocks_t attn_matmul_v);

#include "../src/attention.cpp"
#endif
