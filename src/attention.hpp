#ifndef __ATTENTION_HPP__
#define __ATTENTION_HPP__

#include "dcl.hpp"
#include "linear.hpp"

extern fm_t attn_scale;

void compute_attn(patch_blocks_t q, patch_blocks_t k, patch_blocks_t v,patch_blocks_t attn_matmul_v);
//void compute_max(patch_blocks_t q, patch_blocks_t k,patch_heads_t max,qxk_out_t attn, softmax_info_t attn_softmax_info);
#include "attention.cpp"
#endif
