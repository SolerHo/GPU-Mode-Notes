<h1 align="center">📒 GPU-Mode Lecture 学习笔记</h1>


## 测试环境
- Hardware Infor : NVIDIA GeForce RTX 3060 Ti
- OS Version : Ubuntu 22.04
- Driver Version: 591.86
- CUDA Version: 13.1
- NVCC Version : cuda-toolkit-13-3
- NCU Version : 2026.2.1.0
- Python Version : 3.10.12
- Torch Version : 2.13.0
- Triton Version : 3.7.1
- Tilelang Version : 0.1.13


## 笔记内容

| Lecture| Topics Title| Speaker| Notes| Code|Status|
|--|--|--|--|--|--|
| 001| Profiling and Integrating CUDA kernels in PyTorch| [Mark Saroufim](https://x.com/marksaroufim)| [Lecture001_Notes](Lecture001/)|||
| 002| Recap Ch. 1-3 from the PMPP book| | Lecture002_Notes|||
| 003| Getting Started With CUDA| | Lecture003_Notes|||
| 004| Intro to Compute and Memory Architecture| | Lecture004_Notes|||
| 005| Going Further with CUDA for Python Programmers| | Lecture005_Notes|||
| 006| Optimizing PyTorch Optimizers| | Lecture006_Notes|||
| 007| Advanced Quantization| | Lecture007_Notes|||
| 008| CUDA Performance Checklist| | Lecture008_Notes|||
| 009| Reductions| | Lecture009_Notes|||
| 010| Build a Prod Ready CUDA Library| | Lecture010_Notes|||
| 011| Sparsity| | Lecture011_Notes|||
| 012| Flash Attention| | Lecture012_Notes|||
| 013| Ring Attention| | Lecture013_Notes|||
| 014| Practitioner's Guide to Triton| | Lecture014_Notes|||
| 015| CUTLASS| | Lecture015_Notes|||
| 016| On Hands profiling| | Lecture016_Notes|||
| Bonus Lecture| CUDA C++ llm.cpp| | Bonus_Lecture01_Notes|||
| 017| GPU Collective Communication (NCCL)| | Lecture017_Notes|||
| 018| Fused Kernels| | Lecture018_Notes|||
| 019| Data Processing on GPUs| | Lecture019_Notes|||
| 020| Scan Algorithm| | Lecture020_Notes|||
| 021| Scan Algorithm Part 2| | Lecture021_Notes|||
| 022| Hacker's Guide to Speculative Decoding in VLLM| | Lecture022_Notes|||
| 023| Tensor Cores| | Lecture023_Notes|||
| 024| Scan at the Speed of Light| | Lecture024_Notes|||
| 025| Speaking Composable Kernel (CK)| | Lecture025_Notes|||
| 026| SYCL MODE (Intel GPU)| | Lecture026_Notes|||
| 027| gpu.cpp - Portable GPU compute using WebGPU| | Lecture027_Notes|||
| 028| Liger Kernel - Efficient Triton Kernels for LLM Training| | Lecture028_Notes|||
|Keynotes| GPU Mode IRL 2024 Keynotes|||||
| 029| Triton Internals| | Lecture029_Notes|||
| 030| Quantized training| | Lecture030_Notes|||
| 031| Beginners Guide to Metal Kernels| | Lecture031_Notes|||
| 032| Unsloth - LLM Systems Engineering| | Lecture032_Notes|||
| 033| BitBLAS - Enabling Low Precision DL Computing| | Lecture033_Notes|||
| 034| Low Bit Triton Kernels with Gemlite| | Lecture034_Notes|||
| 035| SGLang Performance Optimization| | Lecture035_Notes|||
| 036| CUTLASS and Flash ATtention 3| | Lecture036_Notes|||
| 037| Introduction to SASS & GPU Microarchitecture| | Lecture037_Notes|||
| 038| Low bit kernels for ARM CPU| | Lecture038_Notes|||
| 039| TorchTitan| | Lecture039_Notes|||
| 040| CUDA Docs for Humans| | Lecture040_Notes|||
| 041| Flash Infer| | Lecture041_Notes|||
| 042| Mosaic GPU| | Lecture042_Notes|||
| 043| int8 Tensor Core Matmul for Turing| | Lecture043_Notes|||
| 044| NVIDIA Profilling| | Lecture044_Notes|||
| 045| Outperforming cuBLAS on H100| | Lecture045_Notes|||
| 046| Distributed GEMM| | Lecture046_Notes|||
| 047| KernelBot Benchmark GPU Kernels on Discord| | Lecture047_Notes|||
| 048| The Ultra Scale Playbook| | Lecture048_Notes|||
| 049| Low Bit Metal Kernels| | Lecture049_Notes|||
| 050| A Learning Journey : CUDA, Triton, Flash Attention| | Lecture050_Notes|||
| 051| Consumer GPU Performance| | Lecture051_Notes|||
| 052| Scaling Laws for Low Precision| | Lecture052_Notes|||
| 053| torch.compile Q & A| | Lecture053_Notes|||
| Bonus Lecture| AMD Developer Challenge 2025| | Bonus_Lecture02_Notes|||
| 054| Samll RL Models at the Speed of Light with LeanRL| | Lecture054_Notes|||
| 055| Mojo: Modular's unified device accelerator language| | Lecture055_Notes|||
| 056| Kernel Benchmarking Tales| | Lecture056_Notes|||
| 057| CuTe| | Lecture057_Notes|||
| 058| Disaggregated LLM Inference : Past, Present and Future| | Lecture058_Notes|||
| 059| FastVideo : Accelerating Large video diffusion models| | Lecture059_Notes|||
| 060| Optimizing Linear Attention| | Lecture060_Notes|||
| 061| d-Matrix Corsair : Low Latency betched Inference for Inference-time-compute| | Lecture061_Notes|||
| 062| Exo 2 : Growing a Scheduling Language| | Lecture062_Notes|||
| 063| Luminal : Search-Based Deep Learning Compilers| | Lecture063_Notes|||
| 064| Multi-GPU Programming| | Lecture064_Notes|||
| 065| Neighborhood Attention| | Lecture065_Notes|||
| 066| GramArena : Evaluating LLM Reasoning through Live Computer Games| | Lecture002_Notes|||
| 067| NCCL and NVSHMEM| | Lecture067_Notes|||
| 068| Landscape of GPU-centric communication| | Lecture068_Notes|||
| 069| Quartet v1: 4 bit training| | Lecture069_Notes|||
| 070| PCCL Fault tolerant collectives| | Lecture070_Notes|||
| 071| [ScaleML Series] Day01 FlexOlmo : Open Language Models for flexible Data Use| | Lecture071_Notes|||
| 072| [ScaleML Series] Day02 : Attention Sink0 Efficient $ Effective Long-Context MOdeling for Large Language Models| | Lecture072_Notes|||
| 073| [ScaleML Series] Day03 : Quantization in large Models| | Lecture073_Notes|||
| 074| [ScaleML Series] Day04 : Positional Encoding & PaTH Attention| | Lecture074_Notes|||
| 075| [ScaleML Series] Day05 : GPU Programming Fundamentals + ThunderKittens| | Lecture075_Notes|||
| 076| BackendBench : Evaluating LLMs at Generating PyTorch backends| | Lecture076_Notes|||
| 077| Domain specific languages for GPU kernels| | Lecture077_Notes|||
| 078| Iris : Multi-GPU Programming in Triton| | Lecture078_Notes|||
| 079| Mirage (MPK) : Compiling LLMs into Mega Kernels| | Lecture079_Notes|||
| 080| How FlashAttention 4 works| | Lecture080_Notes|||
| 081| futhark : High-performance purely functional data-parallerl array programming| | Lecture002_Notes|||
| 082| Helion : A High-level DSL for ML Kernels| | Lecture082_Notes|||
| 083| Formalized Deep Learning Architectures for Automated Low-Level Kernerl Optimizing| | Lecture083_Notes|||
| 084| Numerics and AI| | Lecture084_Notes|||
| 085| Factorio Learning Environment| | Lecture085_Notes|||
| 086| Getting Started with CuTe DSL| | Lecture086_Notes|||
| 087| Low Latency Communication Kernels with NVSHMEM| | Lecture087_Notes|||
| 088| cuTile (NVIDIA Team)| | Lecture088_Notes|||
| 089| Tiny TPU| | Lecture089_Notes|||
| 090| Building resilient ML Engineering Skills| | Lecture090_Notes|||
| 091| Reinforcement Learning, Agents & OpenEnv| | Lecture091_Notes|||
| 092| Smol Training Playbook| | Lecture092_Notes|||
| 093| Cornserve : Easy, Fast and Scalable Multimodal AI| | Lecture093_Notes|||
| 094| tvm-ffi : Open ABI and FFI for Machine Learning Systems| | Lecture094_Notes|||
| 095| Single controller programming with Monarch| | Lecture095_Notes|||
| 096| TLX : Triton-Like Simplicity, a Clear Path to Peak Performance| | Lecture096_Notes|||
| 097| HipKittens| | Lecture097_Notes|||
| 098| GPU Observability| | Lecture098_Notes|||
| 099| Distributed ML on consumer devices| | Lecture099_Notes|||
| 100| InferenceX : Continuous OSS Inference Benchmarking| | Lecture100_Notes|||
| 101| Learning CUTLASS the hard way| | Lecture101_Notes|||
| 102| Quartet v2 : Accurate LLM Pre-training in NVFP4 by Improved Unbiased Gradient Estimation| | Lecture102_Notes|||
| 103| Fundamentals of CuTe Layout Algebra and Category-theoretic Interpretation| | Lecture103_Notes|||
| 104| Gluon and Linear Layouts| | Lecture104_Notes|||
| 105| NVIDIA cuDNN mxfp8 Attention| | Lecture105_Notes|||
| 106| HuggingFace Kernels| | Lecture106_Notes|||



## 参考资源
- YouTube 视频：https://www.youtube.com/@GPUMODE/videos
- Github 地址：https://github.com/gpu-mode/lectures
- https://github.com/gpu-mode/awesomeMLSys

- Resource Stream : https://github.com/gpu-mode/resource-stream

- https://github.com/gpu-mode


关于支付和订单的也一并做了吧。这个前期没啥影响。后续我单独再开页面调试就方便了。CSS风格看起来就像老古董似的。所以最好看起来好看而且别那么多丑。