#ifndef __LINEAR_HPP__
#define __LINEAR_HPP__

#include "dcl.hpp"
#include "hls_stream.h"

enum AttentionLinear {
    ATTN_Q = 0,
    ATTN_K = 1,
    ATTN_V = 2,
    ATTN_PROJ = 3,
    NUM_ATTN_LINEAR
};

constexpr unsigned int MAX_LINEAR_IN_DIM = VIT_HIDDEN_DIM;
constexpr unsigned int MAX_LINEAR_OUT_DIM = VIT_HIDDEN_DIM;
constexpr unsigned int MAX_LINEAR_DIM_PRODUCT = VIT_HIDDEN_DIM * FEATURE_DIM;
constexpr unsigned int QKV_LINEAR_DIM_PRODUCT = FEATURE_DIM * FEATURE_DIM;

typedef hls::vector<hls::vector<wt_linear_t, LINEAR_IN_SIZE>, LINEAR_OUT_SIZE> wt_linear_block_t;
typedef hls::vector<fm_t,LINEAR_OUT_SIZE>wt_bias_block_t;

extern wt_linear_block_t linear_weights_ping[ceildiv(MAX_LINEAR_DIM_PRODUCT, LINEAR_IN_SIZE * LINEAR_OUT_SIZE)];
extern wt_bias_block_t linear_bias_ping[ceildiv(MAX_LINEAR_OUT_DIM, LINEAR_OUT_SIZE)];
extern wt_linear_block_t linear_weights_pong[ceildiv(MAX_LINEAR_DIM_PRODUCT, LINEAR_IN_SIZE * LINEAR_OUT_SIZE)];
extern wt_bias_block_t linear_bias_pong[ceildiv(MAX_LINEAR_OUT_DIM, LINEAR_OUT_SIZE)];

extern wt_linear_block_t linear_weights_attn[NUM_ATTN_LINEAR][ceildiv(QKV_LINEAR_DIM_PRODUCT, LINEAR_IN_SIZE *LINEAR_OUT_SIZE)];
extern wt_bias_block_t linear_bias_attn[NUM_ATTN_LINEAR][ceildiv(FEATURE_DIM, LINEAR_OUT_SIZE)];



void load_linear_weights(
    wt_linear_block_t weights_dst[],
    wt_linear_t weights_src[],
    unsigned int out_dim,
    unsigned int in_dim
);
template<typename T>
void load_linear_bias(
    wt_bias_block_t bias_dst[],
    T bias_src[],
    unsigned int out_dim
);
void compute_linear(
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_linear_block_t weights[],
    const wt_bias_block_t bias[],
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int expert,
    bool use_gelu,
    bool use_expert,
    bool use_score
);
void compute_linear_single(
    fm_block_t dst[],
    const fm_block_t src[],
    const wt_linear_block_t weights[],
    const wt_bias_block_t bias[],
    unsigned int out_dim,
    unsigned int in_dim
);
#include "linear.cpp"
#endif
