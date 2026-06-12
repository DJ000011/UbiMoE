# OpenCL Host 基本用法说明

如果只是想快速整理或生成 OpenCL/XRT host 代码、buffer 绑定、xclbin 加载和部分 Vitis 编译流程，推荐直接配合 [`xilinx-suite`](https://github.com/QingquanYao/xilinx-skill) 这类 skill 来做。个人使用下来，它对 host 侧信息和外围工程流程处理得比较顺；如果你想进一步理解这些代码为什么这么写，或者需要自己排查 buffer、HBM、event 并行等问题，可以继续看下面的细节。需要注意的是，实际 HLS kernel 的书写和性能优化目前仍然不能完全交给工具，通常还要结合综合报告、时序、资源和数据通路继续人工微调。

本文以 `ViT_Fixed16/SW_src/testbench.cpp` 为主线，说明 Vitis HLS/Xilinx FPGA 工程中 OpenCL host 程序的基本写法。最后附一个很短的 XRT C++ API 对照表。

参考文件：

- OpenCL host：`ViT_Fixed16/SW_src/testbench.cpp`
- MSA kernel：`ViT_Fixed16/include/ViT_compute.hpp`、`ViT_Fixed16/HW_src/ViT_compute.cpp`
- FFN kernel：`ViT_Fixed16/include/Feed_Forward.hpp`、`ViT_Fixed16/HW_src/Feed_Forward.cpp`
- XRT C++ 对照示例：`ViT_Fixed16/test/msa_layer_hls/host/msa_host.cpp`

## 1. 本工程 Host 在做什么

`testbench.cpp` 的 host 主要负责：

1. 读取 `.xclbin` 并配置 FPGA。
2. 创建 `ViT_compute` 和 `fullconnect` 两个 kernel。
3. 为输入、输出、中间结果、权重创建 `cl::Buffer`。
4. 用 `setArg` 把 buffer 绑定到 HLS 顶层函数参数。
5. 把输入和权重迁移到 device memory。
6. 用两个 command queue 和 `cl::Event` 组织 MSA/FC 的 Ping-Pong 执行。
7. 把最后结果迁移回 host。

两个 kernel 的逻辑关系大致是：

```text
ViT_compute: LayerNorm + Q/K/V + Attention + Projection
fullconnect: LayerNorm + FFN first linear + FFN second linear
```

Ping/Pong buffer 将 `num_images` 分成两半，避免同一块 buffer 同时被多个阶段读写。

## 2. OpenCL 初始化流程

OpenCL host 初始化可以概括为：

```text
get devices -> read xclbin -> create context -> create queue
            -> create program -> create kernel
```

`testbench.cpp` 中对应代码如下：

```cpp
auto devices = xcl::get_xil_devices();
auto fileBuf = xcl::read_binary_file(binaryFile);
cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];

    context = cl::Context(device, nullptr, nullptr, nullptr, &err);
    q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    q_parallel = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    cl::Program program(context, {device}, bins, nullptr, &err);
    ViT_compute = cl::Kernel(program, "ViT_compute", &err);
    FC = cl::Kernel(program, "fullconnect", &err);
}
```

几个核心对象：

- `cl::Context`：host 和某个 FPGA device 之间的上下文。
- `cl::CommandQueue`：提交数据迁移和 kernel 任务的队列。
- `cl::Program`：由 `.xclbin` 创建出的 OpenCL program。
- `cl::Kernel`：program 中某个 HLS 顶层函数的 host 侧句柄。

这里创建了两个队列 `q` 和 `q_parallel`，后面用 event 控制依赖，使不同 kernel 阶段有机会并行。

## 3. Kernel 参数与 setArg

OpenCL host 调 kernel 时，最重要的规则是：

```text
setArg 的编号必须和 HLS 顶层函数参数顺序完全一致。
```

例如 `ViT_compute` 的前几个参数是：

```cpp
void ViT_compute(
    unsigned int num_images,
    unsigned int layer,
    patch_blocks_t x[],
    patch_blocks_t output[],
    patch_blocks_t x_norm,
    ...
);
```

host 中就要按相同顺序绑定：

```cpp
ViT_compute.setArg(0, num_images);
ViT_compute.setArg(1, 0);
ViT_compute.setArg(2, inputPingBuffer);
ViT_compute.setArg(3, outputPingBuffer);
ViT_compute.setArg(4, norm1Buffer);
```

权重也是一样：

```cpp
ViT_compute.setArg(13, attn_weightsBuffer);
ViT_compute.setArg(14, attn_biasBuffer);
ViT_compute.setArg(15, proj_weightsBuffer);
ViT_compute.setArg(16, proj_biasBuffer);
ViT_compute.setArg(17, norm_weights_l1Buffer);
ViT_compute.setArg(18, norm_bias_l1Buffer);
```

`fullconnect` 对应 `FC.setArg(...)`，同样必须跟 `Feed_Forward.hpp` 中的参数顺序一致。

后续做 Ping/Pong 时，可以在多次 `enqueueTask` 之间修改部分参数：

```cpp
ViT_compute.setArg(2, inputPongBuffer);
ViT_compute.setArg(3, outputPongBuffer);
```

这表示下一次启动 `ViT_compute` 时改用 Pong buffer。

## 4. Buffer、权重和数据迁移

HLS 中的 `m_axi` 参数，在 host 侧通常对应一个 `cl::Buffer`：

```cpp
#pragma HLS interface m_axi port=x offset=slave bundle=in
#pragma HLS interface m_axi port=attn_weights offset=slave bundle=weights
```

对应 host：

```cpp
cl::Buffer inputPingBuffer(
    context,
    CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
    sizeof(patch_blocks_t) * num_images / 2,
    &input_ping,
    &err);

cl::Buffer attn_weightsBuffer(
    context,
    CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
    sizeof(wt_linear_t) * NUM_LAYERS * 3 * FEATURE_DIM * FEATURE_DIM,
    &attn_weights,
    &err);
```

常用 flag：

- `CL_MEM_USE_HOST_PTR`：使用已有 host 数组作为 buffer 的 host 侧存储。
- `CL_MEM_READ_WRITE`：kernel 可读写，适合输入/输出/中间 buffer。
- `CL_MEM_READ_ONLY`：从 kernel 视角只读，适合权重和 bias。

注意：buffer size 是字节数，不是元素个数。

创建 buffer 后，host 数据不会自动进入 FPGA device memory，需要迁移：

```cpp
q.enqueueMigrateMemObjects({inputPingBuffer, inputPongBuffer}, 0);
q.enqueueMigrateMemObjects({attn_weightsBuffer, attn_biasBuffer,
                            proj_weightsBuffer, proj_biasBuffer}, 0);
q.finish();
```

第二个参数 `0` 表示：

```text
host -> device
```

读回时使用：

```cpp
q.enqueueMigrateMemObjects({inputPingBuffer, inputPongBuffer},
                           CL_MIGRATE_MEM_OBJECT_HOST);
q.finish();
```

`CL_MIGRATE_MEM_OBJECT_HOST` 表示：

```text
device -> host
```

本工程最后读回的是 `inputPingBuffer/inputPongBuffer`，因为 `fullconnect` 的输出会写回输入 buffer 位置。

## 5. 权重与 HBM 通道分配

`cl::Buffer` 只表示 host 创建了一块 device buffer，并不保证它一定落到某个 HBM 通道。HBM 分配通常由两部分共同决定。

第一部分是 HLS 的 `m_axi bundle`：

```cpp
#pragma HLS interface m_axi port=attn_weights bundle=weights
#pragma HLS interface m_axi port=proj_weights bundle=weights
#pragma HLS interface m_axi port=vit_weights_l1 bundle=weights
#pragma HLS interface m_axi port=vit_weights_l2 bundle=weights
```

`bundle` 决定 HLS 生成哪些 AXI master 端口。如果多个权重共用同一个 bundle，它们可能共享同一个 AXI master，带宽也会共享。

第二部分是 Vitis link 阶段的 connectivity 配置：

```ini
[connectivity]
sp=ViT_compute_1.attn_weights:HBM[0]
sp=ViT_compute_1.proj_weights:HBM[1]
sp=fullconnect_1.vit_weights_l1:HBM[2]
sp=fullconnect_1.vit_weights_l2:HBM[3]
```

`sp` 的含义是：

```text
sp=<compute_unit>.<kernel_argument>:<memory_resource>
```

如果希望多个大权重真正并行读取，一般要同时做到：

1. HLS 侧拆出足够的 `m_axi bundle`。
2. link config 中用 `sp=` 把不同参数映射到不同 HBM bank。
3. host 侧创建对应 buffer 并按参数顺序 `setArg`。

Host 侧也可以用 Xilinx 扩展显式指定 bank，例如：

```cpp
cl_mem_ext_ptr_t ext;
ext.obj = attn_weights;
ext.param = 0;
ext.flags = XCL_MEM_TOPOLOGY | 0;

cl::Buffer attn_weightsBuffer(
    context,
    CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
    sizeof(attn_weights),
    &ext,
    &err);
```

不过实际工程中更推荐优先检查 `.xclbin` 的 connectivity 配置，避免 host 期望和 link 结果不一致。当前 `testbench.cpp` 没有显式指定 HBM bank，因此权重最终落在哪些通道，要看 link 配置和平台默认策略。

## 6. 多队列、Event 和 Ping-Pong 并行

OpenCL 的 `enqueueTask` 是异步提交。kernel 之间的先后关系可以用 `cl::Event` 表达。

本工程定义了四组 event：

```cpp
cl::Event MSA_Ping_event[NUM_LAYERS], FC_Ping_event[NUM_LAYERS];
cl::Event MSA_Pong_event[NUM_LAYERS], FC_Pong_event[NUM_LAYERS];
```

第一层 Ping 的依赖：

```cpp
q.enqueueTask(ViT_compute, nullptr, &MSA_Ping_event[0]);

std::vector<cl::Event> wait_for_First_Ping{MSA_Ping_event[0]};
q_parallel.enqueueTask(FC, &wait_for_First_Ping, &FC_Ping_event[0]);
```

含义：

```text
MSA_Ping[0] -> FC_Ping[0]
```

Pong 也是同样模式：

```text
MSA_Pong[0] -> FC_Pong[0]
```

后续层的依赖可以理解为：

```text
Ping: FC(layer - 1) -> MSA(layer) -> FC(layer)
Pong: FC(layer - 1) -> MSA(layer) -> FC(layer)
```

两个 command queue 的意义是让 runtime 有机会重叠不同任务：

```text
q:          MSA_Ping[0]        MSA_Pong[0]        ...
q_parallel:          FC_Ping[0]          FC_Pong[0] ...
```

但多队列不等于自动并行。真正并行还取决于：

- kernel 是否是不同 compute unit。
- kernel 之间是否没有未表达的数据依赖。
- buffer 是否避免读写冲突。
- memory bank/HBM 访问是否冲突。

`q.finish()` 和 event wait 的区别：

- `q.finish()` 等待整个队列之前提交的所有命令完成，比较粗。
- event wait list 只等待指定 event，更适合表达流水。

如果每个 kernel 后都 `finish()`，会把本来可以重叠的任务强行串行化。

## 7. 常见错误

- `setArg` 顺序和 HLS 顶层参数顺序不一致。
- `cl::Buffer` size 写成元素个数，而不是字节数。
- 输入和权重 buffer 创建了，但忘记 `enqueueMigrateMemObjects(..., 0)`。
- 读回了错误 buffer，没有根据最后一层输出位置判断。
- event wait list 配错，导致 kernel 提前读到未完成的数据。
- 以为多个 `cl::Buffer` 会自动分到不同 HBM，实际还要看 HLS bundle 和 `sp=` 配置。
- 多个大权重共用同一个 `bundle=weights`，导致 AXI/HBM 带宽没有真正拆开。

## 8. XRT C++ 简短对照

`test/msa_layer_hls/host/msa_host.cpp` 使用的是 XRT C++ API。它和 OpenCL C++ wrapper 的对应关系如下：

| OpenCL C++ wrapper | XRT C++ API | 作用 |
| --- | --- | --- |
| `xcl::get_xil_devices()` + `cl::Context` | `xrt::device device(0)` | 打开 FPGA 设备 |
| `xcl::read_binary_file` + `cl::Program` | `device.load_xclbin(path)` | 加载 xclbin |
| `cl::Kernel(program, "name")` | `xrt::kernel(device, uuid, "name")` | 打开 kernel |
| `cl::Buffer(context, flags, size, ptr)` | `xrt::bo(device, size, group_id)` | 创建 device buffer |
| `kernel.setArg(i, buffer)` | `kernel(buffer0, buffer1, ...)` | 设置参数并启动 |
| `q.enqueueMigrateMemObjects(..., 0)` | `bo.sync(XCL_BO_SYNC_BO_TO_DEVICE)` | host 到 device |
| `q.enqueueTask(kernel, wait, event)` | `auto run = kernel(...);` | 启动 kernel |
| `event.wait()` / `q.finish()` | `run.wait()` | 等待完成 |
| `q.enqueueMigrateMemObjects(..., CL_MIGRATE_MEM_OBJECT_HOST)` | `bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE)` | device 到 host |

XRT C++ 的优势是代码短，尤其是：

```cpp
auto run = kernel(in_bo, out_bo);
run.wait();
```

它把 OpenCL 中的 `setArg`、`enqueueTask`、event/run handle 封装到函数调用式语法里。理解旧工程、做复杂 event 流水时，OpenCL C++ wrapper 仍然很值得掌握；新工程如果没有特殊需求，XRT C++ 会更简洁。
