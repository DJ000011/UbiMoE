#include "../include/linear.hpp"


void load_linear_weights(
    wt_linear_block_t weights_dst[],
    wt_linear_t weights_src[],
    unsigned int out_dim,
    unsigned int in_dim
)
{
    #pragma HLS inline off

    constexpr unsigned int WEIGHT_BLOCK_SIZE = AXI_XFER_BIT_WIDTH / wt_linear_t::width;
    constexpr unsigned int num_src_blocks_in_dst_block = LINEAR_IN_SIZE / WEIGHT_BLOCK_SIZE;
    constexpr unsigned int last_src_block_in_dst_block = num_src_blocks_in_dst_block - 1;

    static_assert(LINEAR_IN_SIZE % WEIGHT_BLOCK_SIZE == 0, "LINEAR_IN_SIZE must be a multiple of WEIGHT_BLOCK_SIZE");

    unsigned int num_src_col_blocks = ceildiv(in_dim, WEIGHT_BLOCK_SIZE);
    unsigned int last_src_col_block = num_src_col_blocks - 1;
    unsigned int num_src_blocks = out_dim * num_src_col_blocks;
    unsigned int last_src_row = out_dim - 1;
    unsigned int num_dst_col_blocks = ceildiv(in_dim, LINEAR_IN_SIZE);
    unsigned int last_dst_col_block = num_dst_col_blocks - 1;

    hls::vector<wt_linear_t, WEIGHT_BLOCK_SIZE>* weights_blocks = reinterpret_cast<hls::vector<wt_linear_t, WEIGHT_BLOCK_SIZE>*>(weights_src);
    hls::vector<wt_linear_t, WEIGHT_BLOCK_SIZE> weights_cache[LINEAR_OUT_SIZE][num_src_blocks_in_dst_block][ceildiv(MAX_LINEAR_IN_DIM, LINEAR_IN_SIZE)];
    #pragma HLS array_partition variable=weights_cache complete dim=1
    #pragma HLS array_partition variable=weights_cache complete dim=2

    unsigned int next_src_col_block = 0;
    unsigned int next_src_row = 0;
    unsigned int next_src_block_in_dst_block = 0;
    unsigned int next_dst_col_block = 0;
    unsigned int next_dst_row_offset = 0;
    unsigned int next_dst_block = 0;

    FOR_EACH(src_block, num_src_blocks)
    {
        #pragma HLS pipeline

        unsigned int src_col_block = next_src_col_block;
        next_src_col_block = (src_col_block == last_src_col_block) ? 0 : src_col_block + 1;
        unsigned int src_row = next_src_row;
        next_src_row = (src_col_block == last_src_col_block) ? src_row + 1 : src_row;
        unsigned int src_block_in_dst_block = next_src_block_in_dst_block;
        next_src_block_in_dst_block = (src_block_in_dst_block == last_src_block_in_dst_block) ? 0 : src_block_in_dst_block + 1;
        unsigned int dst_col_block = next_dst_col_block;
        next_dst_col_block = (src_block_in_dst_block == last_src_block_in_dst_block)
            ? ((dst_col_block == last_dst_col_block) ? 0 : dst_col_block + 1)
            : dst_col_block;
        unsigned int dst_row_offset = next_dst_row_offset;
        next_dst_row_offset = (src_col_block == last_src_col_block)
            ? ((dst_row_offset == LINEAR_OUT_SIZE - 1) ? 0 : dst_row_offset + 1)
            : dst_row_offset;

        weights_cache[dst_row_offset][src_block_in_dst_block][dst_col_block] = weights_blocks[src_block];

        if (
            (src_block_in_dst_block == last_src_block_in_dst_block || src_col_block == last_src_col_block)
            && (dst_row_offset == LINEAR_OUT_SIZE - 1 || src_row == last_src_row)
        )
        {
            unsigned int dst_block = next_dst_block;
            next_dst_block = dst_block + 1;

            wt_linear_block_t block_for_dst;
            FOR_EACH(row, LINEAR_OUT_SIZE)
            {
                hls::vector<wt_linear_t, LINEAR_IN_SIZE> row_for_dst;
                FOR_BLOCK(col, LINEAR_IN_SIZE, WEIGHT_BLOCK_SIZE)
                {
                    hls::vector<wt_linear_t, WEIGHT_BLOCK_SIZE> cached = weights_cache[row][col_block][dst_col_block];
                    FOR_OFFSET(col)
                    {
                        row_for_dst[col] = cached[col_offset];
                    }
                }
                block_for_dst[row] = row_for_dst;
            }
            weights_dst[dst_block] = block_for_dst;
        }
    }
}

template void load_linear_bias<wt_bias_t>(
    wt_bias_block_t bias_dst[],
    wt_bias_t bias_src[],
    unsigned int out_dim
);

template<typename T>
void load_linear_bias(
    wt_bias_block_t bias_dst[],
    T bias_src[],
    unsigned int out_dim
)
{
    #pragma HLS inline off

    constexpr unsigned int BIAS_BLOCK_SIZE = AXI_XFER_BIT_WIDTH / T::width;
    constexpr unsigned int num_src_blocks_in_dst_block = LINEAR_OUT_SIZE / BIAS_BLOCK_SIZE;
    constexpr unsigned int last_src_block_in_dst_block = num_src_blocks_in_dst_block - 1;

    static_assert(LINEAR_OUT_SIZE % BIAS_BLOCK_SIZE == 0, "LINEAR_OUT_SIZE must be a multiple of BIAS_BLOCK_SIZE");

    unsigned int num_src_blocks = ceildiv(out_dim, BIAS_BLOCK_SIZE);
    unsigned int last_src_block = num_src_blocks - 1;

    hls::vector<T, BIAS_BLOCK_SIZE>* bias_blocks = reinterpret_cast<hls::vector<T, BIAS_BLOCK_SIZE>*>(bias_src);
    hls::vector<T, BIAS_BLOCK_SIZE> bias_cache[num_src_blocks_in_dst_block];
    #pragma HLS array_partition variable=bias_cache complete dim=1
    unsigned int next_src_block_in_dst_block = 0;
    unsigned int next_dst_block = 0;

    FOR_EACH(src_block, num_src_blocks)
    {
        #pragma HLS pipeline II=8

        unsigned int src_block_in_dst_block = next_src_block_in_dst_block;
        next_src_block_in_dst_block = (src_block_in_dst_block == last_src_block_in_dst_block) ? 0 : src_block_in_dst_block + 1;

        bias_cache[src_block_in_dst_block] = bias_blocks[src_block];

        if (src_block_in_dst_block == last_src_block_in_dst_block || src_block == last_src_block)
        {
            unsigned int dst_block = next_dst_block;
            next_dst_block = dst_block + 1;

            wt_bias_block_t block;
            FOR_BLOCK(dim, LINEAR_OUT_SIZE, BIAS_BLOCK_SIZE)
            {

                FOR_OFFSET(dim)
                {
                    block[dim] = bias_cache[dim_block][dim_offset];
                }
            }
            bias_dst[dst_block] = block;
        }
    }
}

void read_in_stream_direct(
   hls::stream<linear_in_t>& in_stream,
   const fm_block_t src[],
   unsigned int in_dim
)
{
   #pragma HLS inline off

   static_assert(LINEAR_IN_SIZE % FEATURE_BLOCK_SIZE == 0, "LINEAR_IN_SIZE must be a multiple of FEATURE_BLOCK_SIZE");

   constexpr unsigned int num_blocks = LINEAR_IN_SIZE / FEATURE_BLOCK_SIZE;
   fm_block_t blocks[num_blocks];
   #pragma HLS array_partition variable=blocks complete

   constexpr unsigned int last_block = num_blocks - 1;
   unsigned int next_block = 0;

   unsigned int iters = NUM_PATCHES * ceildiv(in_dim, FEATURE_BLOCK_SIZE);

   FOR_EACH(i, iters)
   {
       #pragma HLS pipeline

       unsigned int block = next_block;
       next_block = (block == last_block) ? 0 : block + 1;

       blocks[block] = src[i];

       if (block == last_block)
       {
           #pragma HLS occurrence cycle=num_blocks

           linear_in_t stream_block;
           FOR_BLOCK(j, LINEAR_IN_SIZE, FEATURE_BLOCK_SIZE)
           {
               FOR_OFFSET_NOCHK(j)
               {
                   stream_block[j] = blocks[j_block][j_offset];
               }
           }
           in_stream << stream_block;
       }
   }
}

void write_out_stream_direct(
   fm_block_t dst[],
   hls::stream<linear_out_t>& out_stream,
   unsigned int out_dim
)
{
   #pragma HLS inline off

   static_assert(LINEAR_OUT_SIZE % FEATURE_BLOCK_SIZE == 0, "LINEAR_OUT_SIZE must be a multiple of FEATURE_BLOCK_SIZE");

   constexpr unsigned int num_blocks = LINEAR_OUT_SIZE / FEATURE_BLOCK_SIZE;
   fm_block_t blocks[num_blocks];
   #pragma HLS array_partition variable =blocks complete

   constexpr unsigned int last_block = num_blocks - 1;
   unsigned int next_block = 0;

   unsigned int iters = NUM_PATCHES * ceildiv(out_dim, FEATURE_BLOCK_SIZE);

   FOR_EACH(i, iters)
   {
       #pragma HLS pipeline

       unsigned int block = next_block;
       next_block = (block == last_block) ? 0 : block + 1;

       if (block == 0)
       {
           #pragma HLS occurrence cycle=num_blocks

           linear_out_t stream_block;
           out_stream >> stream_block;
           FOR_BLOCK(j, LINEAR_OUT_SIZE, FEATURE_BLOCK_SIZE)
           {
               fm_block_t slice;
               FOR_OFFSET_NOCHK(j)
               {
                   slice[j_offset] = stream_block[j];
               }
               blocks[j_block] = slice;
           }
       }
       dst[i] = blocks[block];
//       FOR_EACH(offset,FEATURE_BLOCK_SIZE)
//       {
//    	   printf("linear[%d][%d] is %lf\n",i,offset,dst[i][offset].to_double());
//       }
   }
}

void compute_linear_on_stream_single(
   hls::stream<linear_out_t>& out_stream,
   hls::stream<linear_in_t>& in_stream,
   const wt_linear_block_t weights[],
   const wt_bias_block_t bias[],
   unsigned int out_dim,
   unsigned int in_dim,
   unsigned int length,
   bool use_gelu
)
{
   #pragma HLS inline off

   #pragma HLS aggregate variable=weights
   #pragma HLS aggregate variable=bias

   linear_in_t in_blocks[ceildiv(MAX_LINEAR_IN_DIM, LINEAR_IN_SIZE)];
   fp32_out_t out_block;

   unsigned int out_dim_iters = ceildiv(out_dim, LINEAR_OUT_SIZE);
   unsigned int last_out_dim_iter = out_dim_iters - 1;
   unsigned int in_dim_iters = ceildiv(in_dim, LINEAR_IN_SIZE);
   unsigned int last_in_dim_iter = in_dim_iters - 1;
   unsigned int total_dim_iters = out_dim_iters * in_dim_iters;
   unsigned int last_total_dim_iter = total_dim_iters - 1;
   unsigned int iters = length * total_dim_iters;

   unsigned int next_total_dim_block = 0;
   unsigned int next_in_dim_block = 0;
   unsigned int next_out_dim_block = 0;

   FOR_EACH(i, iters)
   {
       #pragma HLS pipeline

       unsigned int total_dim_block = next_total_dim_block;
       next_total_dim_block = (total_dim_block == last_total_dim_iter) ? 0 : total_dim_block + 1;
       unsigned int in_dim_block = next_in_dim_block;
       next_in_dim_block = (in_dim_block == last_in_dim_iter) ? 0 : in_dim_block + 1;
       unsigned int out_dim_block = next_out_dim_block;
       next_out_dim_block = (total_dim_block == last_total_dim_iter) ? 0 : (in_dim_block == last_in_dim_iter) ? out_dim_block + 1 : out_dim_block;

       if (out_dim_block == 0)
       {
           in_stream >> in_blocks[in_dim_block];
       }
       linear_in_t in_block = in_blocks[in_dim_block];

       if (in_dim_block == 0)
       {
           wt_bias_block_t bias_block = bias[out_dim_block];
           FOR_EACH(out_dim_offset, LINEAR_OUT_SIZE)
           {
               out_block[out_dim_offset] = bias_block[out_dim_offset];
           }
       }

       FOR_EACH(in_dim_offset, LINEAR_IN_SIZE)
       {
           fp32_out_t addend;
           FOR_EACH(out_dim_offset, LINEAR_OUT_SIZE)
           {
               addend[out_dim_offset] = in_block[in_dim_offset] * weights[total_dim_block][out_dim_offset][in_dim_offset];
           }
           out_block += addend;
       }

       if (in_dim_block == last_in_dim_iter)
       {
    	   linear_out_t gelu_block;
    	   if(use_gelu){
    		   gelu_block = gelu(out_block);
    	   }

    	   else{
    		   FOR_EACH(out_dim_offset, LINEAR_OUT_SIZE)
    		{
    			gelu_block[out_dim_offset] = fm_t (out_block[out_dim_offset]);
    		}
    	   }

           out_stream << gelu_block;
       }
   }
}

void compute_linear_single(
    fm_block_t dst[],
    fm_block_t src[],
    const wt_linear_block_t weights[],
    const wt_bias_block_t bias[],
    unsigned int out_dim,
    unsigned int in_dim,
	bool use_gelu
)
{
    #pragma HLS inline off
    #pragma HLS dataflow

    hls::stream<linear_in_t> in_stream;
    hls::stream<linear_out_t> out_stream;
    read_in_stream_direct(in_stream, src, in_dim);
    compute_linear_on_stream_single(out_stream, in_stream, weights, bias, out_dim, in_dim, NUM_PATCHES,use_gelu);
    write_out_stream_direct(dst, out_stream, out_dim);
}




