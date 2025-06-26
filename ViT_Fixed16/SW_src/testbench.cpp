#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "dcl.hpp"


#include <iomanip>
#include <sstream>

#include "xcl2.hpp"

using std::ifstream;
using std::ofstream;
using std::ostream;
using std::ostringstream;
using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::flush;
using std::fixed;
using std::left;
using std::setprecision;
using std::setw;

constexpr double MSE_PASS_THRESHOLD = 0.1;
constexpr unsigned int DISPLAY_PATCH_LIMIT = 5;
constexpr unsigned int DISPLAY_DIM_LIMIT = 5;

constexpr unsigned int num_images = 1000;

patch_blocks_t input_ping[num_images/2];
patch_blocks_t input_pong[num_images/2];
patch_blocks_t output_ping[num_images/2];
patch_blocks_t output_pong[num_images/2];

patch_blocks_t referenced_output[num_images];

patch_blocks_t norm1[1];
patch_blocks_t norm2[1];

patch_blocks_t Q_linear_ping[1];
patch_blocks_t Q_linear_pong[1];
patch_blocks_t K_linear_ping[1];
patch_blocks_t K_linear_pong[1];
patch_blocks_t V_linear_ping[1];
patch_blocks_t V_linear_pong[1];
patch_blocks_t atten[1];
patch_blocks_t proj_linear[1];

patch_blocks_t temp[1];
fm_block_t hidden_ping[1][NUM_PATCHES * ceildiv(VIT_HIDDEN_DIM, FEATURE_BLOCK_SIZE)];
fm_block_t hidden_pong[1][NUM_PATCHES * ceildiv(VIT_HIDDEN_DIM, FEATURE_BLOCK_SIZE)];

wt_linear_t attn_weights[NUM_LAYERS][3][FEATURE_DIM][FEATURE_DIM];
wt_attn_bias_t attn_bias[NUM_LAYERS][3][FEATURE_DIM];
wt_linear_t proj_weights[NUM_LAYERS][FEATURE_DIM][FEATURE_DIM];
wt_attn_bias_t proj_bias[NUM_LAYERS][FEATURE_DIM];

wt_linear_t l1_weights[NUM_LAYERS][VIT_HIDDEN_DIM][FEATURE_DIM];
wt_bias_t l1_bias[NUM_LAYERS][VIT_HIDDEN_DIM];
wt_linear_t l2_weights[NUM_LAYERS][FEATURE_DIM][VIT_HIDDEN_DIM];
wt_bias_t l2_bias[NUM_LAYERS][FEATURE_DIM];

wt_norm_t norm_weights_l1[NUM_LAYERS][FEATURE_DIM];
wt_bias_t norm_bias_l1[NUM_LAYERS][FEATURE_DIM];
wt_norm_t norm_weights_l2[NUM_LAYERS][FEATURE_DIM];
wt_bias_t norm_bias_l2[NUM_LAYERS][FEATURE_DIM];


int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <XCLBIN> \n", argv[0]);
        return -1;
    }

    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::CommandQueue q_parallel;
    cl::Kernel ViT_compute;
    cl::Kernel FC;
    std::string binaryFile = argv[1];
    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();

    // read_binary_file() command will find the OpenCL binary file created using the V++ compiler
    // V++ compiler load into OpenCL Binary and return pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);

    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr,&err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q_parallel = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        // Program device with xclbin file
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Program program(context, {device}, bins, nullptr, &err);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, ViT_compute = cl::Kernel(program, "ViT_compute", &err));
            OCL_CHECK(err, FC = cl::Kernel(program, "fullconnect", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
        
    cout << "Loading weights... " << flush;
    {
        string base_dir = "/home/djl/weights/";
        ifstream ifs;
        for (unsigned int layer = 0; layer < NUM_LAYERS; layer++) {
        string layer_prefix = "l" + std::to_string(layer);
        
        // 加载attention权重
        // Query weights
        ifs.open(base_dir + layer_prefix + "_attn_query_weight_fp.bin", std::ios::binary);
        read(ifs, attn_weights[layer][0]);
        ifs.close();
        
        ifs.open(base_dir + layer_prefix + "_attn_query_bias_fp.bin", std::ios::binary);
        read(ifs, attn_bias[layer][0]);
        ifs.close();
        
        // Key weights
        ifs.open(base_dir + layer_prefix + "_attn_key_weight_fp.bin", std::ios::binary);
        read(ifs, attn_weights[layer][1]);
        ifs.close();
        
        ifs.open(base_dir + layer_prefix + "_attn_key_bias_fp.bin", std::ios::binary);
        read(ifs, attn_bias[layer][1]);
        ifs.close();
        
        // Value weights
        ifs.open(base_dir + layer_prefix + "_attn_value_weight_fp.bin", std::ios::binary);
        read(ifs, attn_weights[layer][2]);
        ifs.close();
        
        ifs.open(base_dir + layer_prefix + "_attn_value_bias_fp.bin", std::ios::binary);
        read(ifs, attn_bias[layer][2]);
        ifs.close();
        
        // Attention output weights
        ifs.open(base_dir + layer_prefix + "_attn_output_weight_fp.bin", std::ios::binary);
        read(ifs, proj_weights[layer]);
        ifs.close();
        
        ifs.open(base_dir + layer_prefix + "_attn_output_bias_fp.bin", std::ios::binary);
        read(ifs, proj_bias[layer]);
        ifs.close();
        
        // Layer norm1 weights
        ifs.open(base_dir + layer_prefix + "_ln_before_weight_fp.bin", std::ios::binary);
        read(ifs, norm_weights_l1[layer]);
        ifs.close();
        
        ifs.open(base_dir + layer_prefix + "_ln_before_bias_fp.bin", std::ios::binary);
        read(ifs, norm_bias_l1[layer]);
        ifs.close();

        // MLP weights
        ifs.open(base_dir + layer_prefix + "_mlp_intermediate_weight_fp.bin", std::ios::binary);
        read(ifs, l1_weights[layer]);
        ifs.close();

        ifs.open(base_dir + layer_prefix + "_mlp_intermediate_bias_fp.bin", std::ios::binary);
        read(ifs, l1_bias[layer]);
        ifs.close();

        ifs.open(base_dir + layer_prefix + "_mlp_output_weight_fp.bin", std::ios::binary);
        read(ifs, l2_weights[layer]);
        ifs.close();

        ifs.open(base_dir + layer_prefix + "_mlp_output_bias_fp.bin", std::ios::binary);
        read(ifs, l2_bias[layer]);
        ifs.close();

        // Layer norm2 weights
        ifs.open(base_dir + layer_prefix + "_ln_after_weight_fp.bin", std::ios::binary);
        read(ifs, norm_weights_l2[layer]);
        ifs.close();

        ifs.open(base_dir + layer_prefix + "_ln_after_bias_fp.bin", std::ios::binary);
        read(ifs, norm_bias_l2[layer]);
        ifs.close();
        }
        
        for (unsigned int index = 0; index < num_images; index++)
        {
        		
        		string index_id = std::to_string(index) + ".bin";

        		ifs.open("/home/djl/activations/embedding_" + index_id,std::ios::binary);
        		
                if(index < num_images / 2)
        		    read(ifs,input_Ping[index]);
                else
                    read(ifs,input_Pong[index - num_images / 2]);
        		ifs.close();

//        		ifs.open("/home/djl/activations/first_layer_attention_" + index_id,std::ios::binary);
//        		read(ifs,referenced_output[index]);
//        		ifs.close();

//        		ifs.open("/home/djl/activations/first_layer_output_" + index_id,std::ios::binary);
//        		read(ifs,referenced_output[index]);
//        		ifs.close();
        }
        
}
    cout << "done!" << endl;


    cout << "Creating buffers... " << flush;

    OCL_CHECK(err, cl::Buffer inputPingBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t) * num_images / 2, &input_ping, &err));
    OCL_CHECK(err, cl::Buffer inputPongBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t) * num_images / 2, &input_pong, &err));
    OCL_CHECK(err, cl::Buffer outputPingBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t) * num_images / 2, &output_ping, &err));
    OCL_CHECK(err, cl::Buffer outputPongBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t) * num_images / 2, &output_pong, &err));
    OCL_CHECK(err, cl::Buffer norm1Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &norm1, &err));
    OCL_CHECK(err, cl::Buffer norm2Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &norm2, &err));
    OCL_CHECK(err, cl::Buffer Q_linear_pingBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &Q_linear_ping, &err));
    OCL_CHECK(err, cl::Buffer Q_linear_pongBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &Q_linear_pong, &err));
    OCL_CHECK(err, cl::Buffer K_linear_pingBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &K_linear_ping, &err));
    OCL_CHECK(err, cl::Buffer K_linear_pongBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &K_linear_pong, &err));
    OCL_CHECK(err, cl::Buffer V_linear_pingBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &V_linear_ping, &err));
    OCL_CHECK(err, cl::Buffer V_linear_pongBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &V_linear_pong, &err));
    OCL_CHECK(err, cl::Buffer attenBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &atten, &err));
    OCL_CHECK(err, cl::Buffer proj_linearBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &proj_linear, &err));
    
    OCL_CHECK(err, cl::Buffer tempBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t), &temp, &err));
    OCL_CHECK(err, cl::Buffer hiddenPingBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(fm_block_t) * NUM_PATCHES * ceildiv(VIT_HIDDEN_DIM, FEATURE_BLOCK_SIZE), &hidden_ping, &err));
    OCL_CHECK(err, cl::Buffer hiddenPongBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(fm_block_t) * NUM_PATCHES * ceildiv(VIT_HIDDEN_DIM, FEATURE_BLOCK_SIZE), &hidden_pong, &err));

    OCL_CHECK(err, cl::Buffer attn_weightsBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * NUM_LAYERS * 3 * FEATURE_DIM * FEATURE_DIM, &attn_weights, &err));
    OCL_CHECK(err, cl::Buffer attn_biasBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_attn_bias_t) * NUM_LAYERS * 3 * FEATURE_DIM, &attn_bias, &err));
    OCL_CHECK(err, cl::Buffer proj_weightsBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * NUM_LAYERS * FEATURE_DIM * FEATURE_DIM, &proj_weights, &err));
    OCL_CHECK(err, cl::Buffer proj_biasBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_attn_bias_t) * NUM_LAYERS * FEATURE_DIM, &proj_bias, &err));
    
    OCL_CHECK(err, cl::Buffer l1_weightsBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * NUM_LAYERS * VIT_HIDDEN_DIM * FEATURE_DIM, &l1_weights, &err));
    OCL_CHECK(err, cl::Buffer l1_biasBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * NUM_LAYERS * VIT_HIDDEN_DIM, &l1_bias, &err));
    OCL_CHECK(err, cl::Buffer l2_weightsBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * NUM_LAYERS * FEATURE_DIM * VIT_HIDDEN_DIM, &l2_weights, &err));
    OCL_CHECK(err, cl::Buffer l2_biasBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * NUM_LAYERS * FEATURE_DIM, &l2_bias, &err));

    OCL_CHECK(err, cl::Buffer norm_weights_l1Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_norm_t) * NUM_LAYERS * FEATURE_DIM, &norm_weights_l1, &err));
    OCL_CHECK(err, cl::Buffer norm_bias_l1Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * NUM_LAYERS * FEATURE_DIM, &norm_bias_l1, &err));
    OCL_CHECK(err, cl::Buffer norm_weights_l2Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_norm_t) * NUM_LAYERS * FEATURE_DIM, &norm_weights_l2, &err));
    OCL_CHECK(err, cl::Buffer norm_bias_l2Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * NUM_LAYERS * FEATURE_DIM, &norm_bias_l2, &err));


    cout << "done!" << endl;

    
    cout<< "start setarg"<< endl;

    OCL_CHECK(err, err = ViT_compute.setArg(0, num_images));
    OCL_CHECK(err, err = ViT_compute.setArg(1, 0));
    OCL_CHECK(err, err = ViT_compute.setArg(2, inputPingBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(3, outputPingBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(4, norm1Buffer));
    OCL_CHECK(err, err = ViT_compute.setArg(5, Q_linear_pingBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(6, Q_linear_pongBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(7, K_linear_pingBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(8, K_linear_pongBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(9, V_linear_pingBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(10, V_linear_pongBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(11, attenBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(12, proj_linearBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(13, attn_weightsBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(14, attn_biasBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(15, proj_weightsBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(16, proj_biasBuffer));
    OCL_CHECK(err, err = ViT_compute.setArg(17, norm_weights_l1Buffer));
    OCL_CHECK(err, err = ViT_compute.setArg(18, norm_bias_l1Buffer));

    OCL_CHECK(err, err = FC.setArg(0, num_images));
    OCL_CHECK(err, err = FC.setArg(1, 0));
    OCL_CHECK(err, err = FC.setArg(2, outputPingBuffer));
    OCL_CHECK(err, err = FC.setArg(3, inputPingBuffer));
    OCL_CHECK(err, err = FC.setArg(4, norm2Buffer));
    OCL_CHECK(err, err = FC.setArg(5, tempBuffer));
    OCL_CHECK(err, err = FC.setArg(6, hiddenPingBuffer));
    OCL_CHECK(err, err = FC.setArg(7, hiddenPongBuffer));
    OCL_CHECK(err, err = FC.setArg(8, l1_weightsBuffer));
    OCL_CHECK(err, err = FC.setArg(9, l1_biasBuffer));
    OCL_CHECK(err, err = FC.setArg(10, l2_weightsBuffer));
    OCL_CHECK(err, err = FC.setArg(11, l2_biasBuffer));
    OCL_CHECK(err, err = FC.setArg(12, norm_weights_l2Buffer));
    OCL_CHECK(err, err = FC.setArg(13, norm_bias_l2Buffer));
    cout << "setarg done!" << endl;
    cout << "Copying data to device... " << flush;
    

 //   double kernel_time_in_sec = 0, result = 0;

//    std::chrono::duration<double> kernel_time(0);

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({attn_weightsBuffer, attn_biasBuffer, proj_weightsBuffer, proj_biasBuffer, norm_weights_l1Buffer, norm_bias_l1Buffer}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({inputPingBuffer, inputPongBuffer}, 0 /* 0 means from host*/));

//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({outputPingBuffer}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({l1_weightsBuffer, l1_biasBuffer, l2_weightsBuffer, l2_biasBuffer, norm_weights_l2Buffer, norm_bias_l2Buffer}, 0 /* 0 means from host*/));
    q.finish();

    cout << "done!" << endl;

    cl::Event MSA_Ping_event[NUM_LAYERS], FC_Ping_event[NUM_LAYERS];
    cl::Event MSA_Pong_event[NUM_LAYERS], FC_Pong_event[NUM_LAYERS];
   
    cout<<"start running"<<endl;

    auto kernel_start = std::chrono::high_resolution_clock::now();
    
    // initialization: launch first Ping and Pong tasks
    // MSA and FC Ping tasks
    OCL_CHECK(err,err = q.enqueueTask(ViT_compute,nullptr,&MSA_Ping_event[0]));
    std::vector<cl::Event> wait_for_First_Ping { MSA_Ping_event[0] };
    OCL_CHECK(err,err = q_parallel.enqueueTask(FC, &wait_for_First_Ping, &FC_Ping_event[0]));

    // next, launch first Pong tasks
    OCL_CHECK(err,err = ViT_compute.setArg(2, inputPongBuffer));
    OCL_CHECK(err,err = ViT_compute.setArg(3, outputPongBuffer));
    OCL_CHECK(err,err = q.enqueueTask(ViT_compute,nullptr, &MSA_Pong_event[0]));

    OCL_CHECK(err,err = FC.setArg(2, outputPongBuffer));
    OCL_CHECK(err,err = FC.setArg(3, inputPongBuffer));
    std::vector<cl::Event> wait_for_First_Pong { MSA_Pong_event[0] };
    OCL_CHECK(err,err = q_parallel.enqueueTask(FC, &wait_for_First_Pong, &FC_Pong_event[0]));

    for (unsigned int layer = 1; layer < NUM_LAYERS; layer++)
    {
        OCL_CHECK(err, err = ViT_compute.setArg(1, layer));
        OCL_CHECK(err, err = FC.setArg(1, layer));

        OCL_CHECK(err, err = ViT_compute.setArg(2, inputPingBuffer));
        OCL_CHECK(err, err = ViT_compute.setArg(3, outputPingBuffer));
        std::vector<cl::Event> wait_for_FC_Ping{FC_Ping_event[layer - 1]};
        OCL_CHECK(err, err = q_parallel.enqueueTask(FC, &wait_for_FC_Ping, &MSA_Ping_event[layer]));

        OCL_CHECK(err, err = FC.setArg(2, outputPingBuffer));
        OCL_CHECK(err, err = FC.setArg(3, inputPingBuffer));
        std::vector<cl::Event> wait_for_MSA_Ping{MSA_Ping_event[layer]};
        OCL_CHECK(err, err = q_parallel.enqueueTask(FC, &wait_for_MSA_Ping, &FC_Ping_event[layer]));

        OCL_CHECK(err, err = ViT_compute.setArg(2, inputPongBuffer));
        OCL_CHECK(err, err = ViT_compute.setArg(3, outputPongBuffer));
        std::vector<cl::Event> wait_for_FC_Pong{FC_Pong_event[layer - 1]};
        OCL_CHECK(err, err = q.enqueueTask(ViT_compute, &wait_for_FC_Pong, &MSA_Pong_event[layer]));

        OCL_CHECK(err, err = FC.setArg(2, outputPongBuffer));
        OCL_CHECK(err, err = FC.setArg(3, inputPongBuffer));
        std::vector<cl::Event> wait_for_MSA_Pong{MSA_Pong_event[layer]};
        OCL_CHECK(err, err = q_parallel.enqueueTask(FC, &wait_for_MSA_Pong, &FC_Pong_event[layer]));
    }
}

    cout << "kernel finished" << endl;

    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = kernel_end - kernel_start;
    cout << "kernel time: " << kernel_time.count() * 1000 << " ms" << endl;

    OCL_CHECK(err,err = q.enqueueMigrateMemObjects({inputPingBuffer,inputPongBuffer}, CL_MIGRATE_MEM_OBJECT_HOST) );
    q.finish();

    // release the events
    for (unsigned int layer = 0; layer < NUM_LAYERS; layer++) {
        MSA_Ping_event[layer].release();
        FC_Ping_event[layer].release();
        MSA_Pong_event[layer].release();
        FC_Pong_event[layer].release();
    }

    cout<<"write output to file"<<endl;
        ofstream ofs;

        
        for (unsigned int index = 0; index < num_images; index++)
        {
            string index_id = std::to_string(index) + ".bin";
            ofs.open("/home/djl/activations/final_output_" + index_id, std::ios::binary);
            if (index < num_images / 2)
                write(ofs, input_ping[index]);
            else
                write(ofs, input_pong[index - num_images / 2]);

            ofs.close();
        }
        
        cout << "done!" << endl;
        return 0;

//    FOR_EACH(image,num_images){
//    printf("test %d:\n",image);
//    double mse = 0;
//    double mae = 0;
//    FOR_EACH(patch, NUM_PATCHES)
//            {
//                FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
//                {
//                    FOR_OFFSET(dim)
//                    {
//                        double computed = output_ping[image][patch][dim_block][dim_offset].to_double();
//                        double actual = referenced_output[image][patch][dim_block][dim_offset].to_double();
//                        if(patch == 0 && dim_block == 0)
//                        	printf("result[%d][%d][%d] is %lf,%f\n",patch,dim_block,dim_offset,actual,computed);
//                        double error = actual - computed;
//                        double abs_error = (error < 0.0) ? -error : error;
//                        mse += error * error;
//                        mae += abs_error;
//                    }
//                }
//            }
//            mse /= static_cast<double>(NUM_PATCHES * FEATURE_DIM);
//            mae /= static_cast<double>(NUM_PATCHES * FEATURE_DIM);
//            cout << "MSE: " << mse << endl;
//            cout << "MAE: " << mae << endl;
////            if (mse> MSE_PASS_THRESHOLD){
////            	cout<< "test failed" << endl;
////            	return 1;
////            }
//    }
    return 0;
}


