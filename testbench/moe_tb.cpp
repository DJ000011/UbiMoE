#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "../libs/common/includes/add.hpp"
#include "../libs/common/includes/conv.hpp"
#include "../libs/common/includes/kernel.hpp"
#include "../libs/common/includes/moe.hpp"
#include "../libs/common/includes/xcl2/xcl2.hpp"

#include <iomanip>
#include <sstream>


constexpr double MSE_PASS_THRESHOLD = 0.1;
constexpr unsigned int DISPLAY_PATCH_LIMIT = 5;
constexpr unsigned int DISPLAY_DIM_LIMIT = 5;

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

unsigned int dataSize = 1024 * 1024;

unsigned int num_images = 1;
bool reload_one_time_weights = true;
image_t images[1];
patch_blocks_t x[1];
patch_blocks_t x_norm2[1];
patch_blocks_t final[1];
patch_blocks_t input[1];
// qxk_out_t attn;
// softmax_info_t attn_softmax_info;
wt_patch_embed_t patch_embed_weights_in[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH];
wt_bias_t patch_embed_bias_in[FEATURE_DIM];
patch_blocks_t pos_embed;
wt_linear_t attn_weights[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM][FEATURE_DIM];
wt_attn_bias_t attn_bias[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM];
wt_linear_t moe_w_gate[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM];
wt_linear_t moe_weights_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM][FEATURE_DIM];
wt_bias_t moe_bias_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM];
wt_linear_t moe_weights_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM][EXPERT_HIDDEN_DIM];
wt_bias_t moe_bias_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM];
wt_linear_t vit_weights_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM][FEATURE_DIM];
wt_bias_t vit_bias_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM];
wt_linear_t vit_weights_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM][VIT_HIDDEN_DIM];
wt_bias_t vit_bias_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM];
wt_norm_t norm_weights[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM];
wt_bias_t norm_bias[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM];
patch_heads_t result;


patch_blocks_t reference_x[1];


int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <XCLBIN> \n", argv[0]);
        return -1;
    }

    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel embed;
    cl::Kernel Vit_compute;
    cl::Kernel fc;
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

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, embed = cl::Kernel(program, "patch_embed", &err));
            OCL_CHECK(err, Vit_compute = cl::Kernel(program, "ViT_compute", &err));
            OCL_CHECK(err, fc = cl::Kernel(program, "fullconnect", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    cout << "Loading inputs... " << flush;
  /*{
    	string filename = "/home/dongjl/weights/tmp1.bin";
    	            ifstream ifs(filename, std::ios::binary);
    	            read(ifs,x_norm2[0]);
    	            if (!ifs)
    	            {
    	                cerr << "Error reading " << filename << endl;
    	                return 1;
    	            }
    }*/
        {
            string filename = "/home/dongjl/weights/image.float32.bin";
            ifstream ifs(filename, std::ios::binary);
            read(ifs,images[0]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
            cout<<"success"<<endl;
        }
       {
            string filename = "/home/dongjl/weights/patch_embed_weight.float32.bin";
            ifstream ifs(filename, std::ios::binary);
            read(ifs, patch_embed_weights_in);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        {
            string filename = "/home/dongjl/weights/patch_embed_bias.float32.bin";
            ifstream ifs(filename, std::ios::binary);
            read(ifs, patch_embed_bias_in);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        {
            string filename = "/home/dongjl/weights/pos_embed.float32.bin";
            ifstream ifs(filename, std::ios::binary);
            read(ifs, pos_embed);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(layer, NUM_LAYERS)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << layer << "_qkv_weight.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, attn_weights[layer][ATTN_Q]);
            read(ifs, attn_weights[layer][ATTN_K]);
            read(ifs, attn_weights[layer][ATTN_V]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(layer, NUM_LAYERS)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << layer << "_attn_proj_weight.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, attn_weights[layer][ATTN_PROJ]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(layer, NUM_LAYERS)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << layer << "_qkv_bias.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, attn_bias[layer][ATTN_Q]);
            read(ifs, attn_bias[layer][ATTN_K]);
            read(ifs, attn_bias[layer][ATTN_V]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(layer, NUM_LAYERS)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << layer << "_attn_proj_bias.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, attn_bias[layer][ATTN_PROJ]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(moe_layer, NUM_LAYERS / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (moe_layer * 2 + 1) << "_w_gate_T_task0.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, moe_w_gate[moe_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(moe_layer, NUM_LAYERS / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (moe_layer * 2 + 1) << "_htoh4_weight.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, moe_weights_l1[moe_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(moe_layer, NUM_LAYERS / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (moe_layer * 2 + 1) << "_htoh4_bias.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, moe_bias_l1[moe_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(moe_layer, NUM_LAYERS / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (moe_layer * 2 + 1) << "_h4toh_weight.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, moe_weights_l2[moe_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(moe_layer, NUM_LAYERS / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (moe_layer * 2 + 1) << "_h4toh_bias.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, moe_bias_l2[moe_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(vit_layer, (NUM_LAYERS + 1) / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (vit_layer * 2) << "_fc1_weight.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, vit_weights_l1[vit_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(vit_layer, (NUM_LAYERS + 1) / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (vit_layer * 2) << "_fc1_bias.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, vit_bias_l1[vit_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(vit_layer, (NUM_LAYERS + 1) / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (vit_layer * 2) << "_fc2_weight.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, vit_weights_l2[vit_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(vit_layer, (NUM_LAYERS + 1) / 2)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << (vit_layer * 2) << "_fc2_bias.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, vit_bias_l2[vit_layer]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(layer, NUM_LAYERS)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << layer << "_norm1_weight.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, norm_weights[layer][NORM_1]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(layer, NUM_LAYERS)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << layer << "_norm2_weight.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, norm_weights[layer][NORM_2]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(layer, NUM_LAYERS)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << layer << "_norm1_bias.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, norm_bias[layer][NORM_1]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        FOR_EACH(layer, NUM_LAYERS)
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/l" << layer << "_norm2_bias.float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs, norm_bias[layer][NORM_2]);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }

        string reference_var_name;
        {
            ostringstream oss;
            oss << "l" << (NUM_LAYERS - 1) << "_x_post_" << (((NUM_LAYERS - 1) % 2 == 0) ? "mlp" : "moe");
            reference_var_name = oss.str();
        }
        {
            ostringstream oss;
            oss << "/home/dongjl/weights/" << reference_var_name << ".float32.bin";
            string filename = oss.str();
            ifstream ifs(filename, std::ios::binary);
            read(ifs,reference_x);
            if (!ifs)
            {
                cerr << "Error reading " << filename << endl;
                return 1;
            }
        }
        cout << "done!" << endl;
        cout << "Running kernel... " << flush;

   cout << endl;



   OCL_CHECK(err, cl::Buffer imagesBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(image_t)*num_images, &images, &err));
   
    OCL_CHECK(err, cl::Buffer x_norm2Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t)*num_images, &x_norm2, &err));
  
    OCL_CHECK(err, cl::Buffer finalBuffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(patch_blocks_t)*num_images, &final, &err));

    OCL_CHECK(err, cl::Buffer patch_embed_weights_inBuffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_patch_embed_t) * FEATURE_DIM * INPUT_CHANNELS * PATCH_HEIGHT * PATCH_WIDTH, &patch_embed_weights_in, &err));

    OCL_CHECK(err, cl::Buffer patch_embed_bias_inBuffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * FEATURE_DIM, &patch_embed_bias_in, &err));

    OCL_CHECK(err, cl::Buffer pos_embedBuffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(patch_blocks_t), &pos_embed, &err));

    OCL_CHECK(err, cl::Buffer attn_weightsBuffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * NUM_LAYERS * NUM_ATTN_LINEAR * FEATURE_DIM * FEATURE_DIM, &attn_weights, &err));

    OCL_CHECK(err, cl::Buffer attn_biasBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_attn_bias_t) * NUM_LAYERS * NUM_ATTN_LINEAR * FEATURE_DIM, &attn_bias, &err));

    OCL_CHECK(err, cl::Buffer vit_weights_l1Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * max((NUM_LAYERS + 1) / 2, 1U) * VIT_HIDDEN_DIM * FEATURE_DIM, &vit_weights_l1, &err));

    OCL_CHECK(err, cl::Buffer vit_bias_l1Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * max((NUM_LAYERS + 1) / 2, 1U)  * VIT_HIDDEN_DIM, &vit_bias_l1, &err));

    OCL_CHECK(err, cl::Buffer vit_weights_l2Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * max((NUM_LAYERS + 1) / 2, 1U)  * FEATURE_DIM * VIT_HIDDEN_DIM, &vit_weights_l2, &err));

    OCL_CHECK(err, cl::Buffer vit_bias_l2Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * max((NUM_LAYERS + 1) / 2, 1U)  * FEATURE_DIM, &vit_bias_l2, &err));

    OCL_CHECK(err, cl::Buffer norm_weightsBuffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_norm_t) * NUM_LAYERS * NUM_LAYER_NORMS * FEATURE_DIM, &norm_weights, &err));

    OCL_CHECK(err, cl::Buffer norm_biasBuffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * NUM_LAYERS * NUM_LAYER_NORMS * FEATURE_DIM, &norm_bias, &err));

    OCL_CHECK(err, cl::Buffer moe_w_gateBuffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * max(NUM_LAYERS / 2, 1U) * NUM_EXPERTS * FEATURE_DIM, &moe_w_gate, &err));

    OCL_CHECK(err, cl::Buffer moe_weights_l1Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * max(NUM_LAYERS / 2, 1U) * NUM_EXPERTS * EXPERT_HIDDEN_DIM * FEATURE_DIM, &moe_weights_l1, &err));

    OCL_CHECK(err, cl::Buffer moe_weights_l2Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_linear_t) * max(NUM_LAYERS / 2, 1U) * NUM_EXPERTS * EXPERT_HIDDEN_DIM * FEATURE_DIM, &moe_weights_l2, &err));

    OCL_CHECK(err, cl::Buffer moe_bias_l1Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * max(NUM_LAYERS / 2, 1U) * NUM_EXPERTS * EXPERT_HIDDEN_DIM, &moe_bias_l1, &err));

    OCL_CHECK(err, cl::Buffer moe_bias_l2Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(wt_bias_t) * max(NUM_LAYERS / 2, 1U) * NUM_EXPERTS * FEATURE_DIM, &moe_bias_l2, &err));





    cout<< "start setarg"<< endl;
    OCL_CHECK(err, err = embed.setArg(0, 1));
    OCL_CHECK(err, err = embed.setArg(1, imagesBuffer));
    OCL_CHECK(err, err = embed.setArg(2, finalBuffer));
    OCL_CHECK(err, err = embed.setArg(3, patch_embed_weights_inBuffer));
    OCL_CHECK(err, err = embed.setArg(4, patch_embed_bias_inBuffer));
    OCL_CHECK(err, err = embed.setArg(5, pos_embedBuffer));

    OCL_CHECK(err, err = Vit_compute.setArg(0, 1));
    OCL_CHECK(err, err = Vit_compute.setArg(1, 0));
    OCL_CHECK(err, err = Vit_compute.setArg(2, finalBuffer));

    OCL_CHECK(err, err = Vit_compute.setArg(3, attn_weightsBuffer));
    OCL_CHECK(err, err = Vit_compute.setArg(4, attn_biasBuffer));
    OCL_CHECK(err, err = Vit_compute.setArg(5, norm_weightsBuffer));
    OCL_CHECK(err, err = Vit_compute.setArg(6, norm_biasBuffer));

    OCL_CHECK(err, err = Vit_compute.setArg(7, x_norm2Buffer));


    OCL_CHECK(err,err = fc.setArg(0,1));
    OCL_CHECK(err,err = fc.setArg(1,0));//layer
    OCL_CHECK(err,err = fc.setArg(2,x_norm2Buffer));//input
    OCL_CHECK(err,err = fc.setArg(3,finalBuffer));//output
    OCL_CHECK(err,err = fc.setArg(4,vit_weights_l1Buffer));
    OCL_CHECK(err,err = fc.setArg(5,vit_bias_l1Buffer));
    OCL_CHECK(err,err = fc.setArg(6,vit_weights_l2Buffer));
    OCL_CHECK(err,err = fc.setArg(7,vit_bias_l2Buffer));
    OCL_CHECK(err,err = fc.setArg(8,moe_w_gateBuffer));
    OCL_CHECK(err,err = fc.setArg(9,moe_weights_l1Buffer));
    OCL_CHECK(err,err = fc.setArg(10,moe_bias_l1Buffer));
    OCL_CHECK(err,err = fc.setArg(11,moe_weights_l2Buffer));
    OCL_CHECK(err,err = fc.setArg(12,moe_bias_l2Buffer));
    OCL_CHECK(err, err = fc.setArg(13, norm_weightsBuffer));
    OCL_CHECK(err, err = fc.setArg(14, norm_biasBuffer));


 //   double kernel_time_in_sec = 0, result = 0;

//    std::chrono::duration<double> kernel_time(0);

//    auto kernel_start = std::chrono::high_resolution_clock::now();
    cout<<"start run"<<endl;
    OCL_CHECK(err,err = q.enqueueMigrateMemObjects({imagesBuffer,patch_embed_weights_inBuffer, patch_embed_bias_inBuffer, pos_embedBuffer, attn_weightsBuffer, attn_biasBuffer }, 0 /* 0 means from host*/));

    OCL_CHECK(err,err = q.enqueueMigrateMemObjects({norm_weightsBuffer, norm_biasBuffer,vit_weights_l1Buffer,vit_bias_l1Buffer,vit_weights_l2Buffer,vit_bias_l2Buffer,moe_w_gateBuffer,moe_weights_l1Buffer,moe_weights_l2Buffer,moe_bias_l1Buffer,moe_bias_l2Buffer}, 0 /* 0 means from host*/));

    q.finish();

 //   std::vector<cl::Event> events(NUM_LAYERS);

    OCL_CHECK(err,err = q.enqueueTask(embed));

    for(unsigned int layer = 0;layer < NUM_LAYERS ; layer++){

    	// cout<<"layer"<< layer <<"start"<<endl;

        OCL_CHECK(err, err = Vit_compute.setArg(1, layer));
        OCL_CHECK(err,err = q.enqueueTask(Vit_compute));
        q.finish();

        // OCL_CHECK(err,err = q.enqueueMigrateMemObjects({x_norm2Buffer}, CL_MIGRATE_MEM_OBJECT_HOST) );
        // q.finish();
        // cout<<"layer "<<layer<<"part1 done"<<endl;
        // 	{
        //            ostream formatted(cout.rdbuf());
        //            formatted << setprecision(8) << fixed;
        //            FOR_EACH(patch, DISPLAY_PATCH_LIMIT)
        //            {
        //                    cout << ((patch == 0) ? "[[" : " [");
        //                    FOR_BLOCK(dim, DISPLAY_DIM_LIMIT, FEATURE_BLOCK_SIZE)
        //                    {
        //                        FOR_OFFSET(dim)
        //                        {
        //                            double actual = x_norm2[0][patch][dim_block][dim_offset].to_double();
        //                            if (actual >= 0.0) cout << " ";
        //                            formatted << setw(9 + (actual < 0.0)) << left << actual;
        //                            if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
        //                        }
        //                    }
        //                    cout << ((patch == DISPLAY_PATCH_LIMIT - 1) ? "]]" : "],") << endl;

        //            }
        //        }

        OCL_CHECK(err,err = fc.setArg(1,layer));
        OCL_CHECK(err,err = q.enqueueTask(fc));
        q.finish();
    //   OCL_CHECK(err,err = q.enqueueMigrateMemObjects({finalBuffer}, CL_MIGRATE_MEM_OBJECT_HOST) );
    //   q.finish();
    //   OCL_CHECK(err, err = q.enqueueCopyBuffer(finalBuffer,xBuffer,0,0,sizeof(patch_blocks_t)*num_images));
    //   OCL_CHECK(err,err = q.enqueueMigrateMemObjects({xBuffer}, 0) );
        // cout << "layer " << layer << "part2 done" << endl;
        // {
        //     ostream formatted(cout.rdbuf());
        //     formatted << setprecision(8) << fixed;
        //     FOR_EACH(patch, DISPLAY_PATCH_LIMIT)
        //     {
        //         cout << ((patch == 0) ? "[[" : " [");
        //         FOR_BLOCK(dim, DISPLAY_DIM_LIMIT, FEATURE_BLOCK_SIZE)
        //         {
        //             FOR_OFFSET(dim)
        //             {
        //                 double actual = final[0][patch][dim_block][dim_offset].to_double();
        //                 if (actual >= 0.0)
        //                     cout << " ";
        //                 formatted << setw(9 + (actual < 0.0)) << left << actual;
        //                 if (dim != DISPLAY_DIM_LIMIT - 1)
        //                     cout << ", ";
        //             }
        //         }
        //         cout << ((patch == DISPLAY_PATCH_LIMIT - 1) ? "]]" : "],") << endl;
        //     }
        // }
    }
//   auto kernel_end = std::chrono::high_resolution_clock::now();

    cout<<"all part done"<<endl;
    OCL_CHECK(err,err = q.enqueueMigrateMemObjects({finalBuffer}, CL_MIGRATE_MEM_OBJECT_HOST) );
    q.finish();


    // cout << "storing x_norm2 to tmp1.bin for testing"<< endl;
    // string filename = "/home/dongjl/weights/tmp1.bin";
    // ofstream ofs(filename, std::ios::binary);
    // write(ofs,x_norm2);
    // if(!ofs){
    // 	 cerr << "Error writing " << filename << endl;
    // 	 return 1;
    // }

// cout <<"max in subtract is"<< endl;
// FOR_EACH(patch,NUM_PATCHES){
// 	FOR_EACH(head,NUM_HEADS){
// 		double tmp = result[patch][head].to_double();
//      if (tmp >= 0.01) cout << tmp << endl;
// 	}
// }

    cout << "Sample of values from x vs. " << reference_var_name << ":" << endl;
        {
            ostream formatted(cout.rdbuf());
            formatted << setprecision(8) << fixed;
            FOR_EACH(patch, DISPLAY_PATCH_LIMIT)
            {
                {
                    cout << ((patch == 0) ? "[[" : " [");
                    FOR_BLOCK(dim, DISPLAY_DIM_LIMIT, FEATURE_BLOCK_SIZE)
                    {
                        FOR_OFFSET(dim)
                        {
                            double computed = final[0][patch][dim_block][dim_offset].to_double();
                            if (computed >= 0.0) cout << " ";
                            formatted << setw(9 + (computed < 0.0)) << left << computed;
                            if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
                        }
                    }
                    cout << ((patch == DISPLAY_PATCH_LIMIT - 1) ? "]]" : "],") << "    ";
                }
                {
                    cout << ((patch == 0) ? "[[" : " [");
                    FOR_BLOCK(dim, DISPLAY_DIM_LIMIT, FEATURE_BLOCK_SIZE)
                    {
                        FOR_OFFSET(dim)
                        {
                            double actual = reference_x[0][patch][dim_block][dim_offset].to_double();
                            if (actual >= 0.0) cout << " ";
                            formatted << setw(9 + (actual < 0.0)) << left << actual;
                            if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
                        }
                    }
                    cout << ((patch == DISPLAY_PATCH_LIMIT - 1) ? "]]" : "],") << endl;
                }
            }
        }
        cout << endl;

        double mse = 0.0;
        double mae = 0.0;

        FOR_EACH(patch, NUM_PATCHES)
        {
            FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
            {
                FOR_OFFSET(dim)
                {
                    double computed = final[0][patch][dim_block][dim_offset].to_double();
                    double actual = reference_x[0][patch][dim_block][dim_offset].to_double();
                    double error = actual - computed;
                    //double error = computed;


                    double abs_error = (error < 0.0) ? -error : error;
                    mse += error * error;
                    mae += abs_error;
                }
            }
        }
        mse /= static_cast<double>(NUM_PATCHES * FEATURE_DIM);
        mae /= static_cast<double>(NUM_PATCHES * FEATURE_DIM);
        cout << "MSE: " << mse << endl;
        cout << "MAE: " << mae << endl;
        if (mse> MSE_PASS_THRESHOLD){
        	cout<< "test failed" << endl;
        	return 1;
        }
        return 0;

}
