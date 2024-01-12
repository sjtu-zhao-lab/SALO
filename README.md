# SALO
This repository implements the proposed spatial accelerator design in the paper SALO: An Efficient Spatial Accelerator Enabling Hybrid Sparse Attention Mechanisms for Long Sequences (DAC-22)

## Overview
SALO is an spatial accelerator design aiming at efficiently handling hybrid sparse attention patterns statically, including sliding window attention, dilated sliding window attention, and global attention. It can primitively support popular NLP and CV models like Longformer, Vision Longformer.

## Performance Evaluation
SALO v1 is not a software-hardware co-design. It doesn't introduce intrusive modifications to the algorithm deisgn, which won't cause accuracy degradation. Thus in the software part, we mainly provide the performance comparison between SALO v1 and CPU/GPU, based on the inference speed. The benchmark code that evaluates the CPU and GPU performance on 3 workloads mentioned in the paper is located at `benchmark/bench_cpu_gpu.py`. To run the script successfully, we recommend our experiment settings:
+ CUDA >= 10.1
+ Python >= 3.8
+ PyTorch >= 1.7.0
+ timm >= 0.3.2

To evaluate the performance of SALO, we developed a cycle-accurate performance model in `performance_model/performance_model.py`. The performance model estimates the computation FLOPS in each stage and calculates the number of cycles to run.

## Hardware Design
The hardware design of SALO is located in `hardware\src\main\scala\sa`. It is implemented by Chisel3. Here we list the explanations of the main parts of SALO.

+ `hardware\src\main\scala\sa\Arithmetic.scala`: This file includes the supportive fixed-point arithmetic functions (Bitwidth up/down-cast) that are useful in the hardware implementation.
+ `hardware\src\main\scala\sa\InverseModule.scala`: This file includes the required module to obtain the inverse of the sum of exp($\frac{1}{\sum_{k}\exp(S_{ik})}$) in the sparse attention mechanism. Use a unified divisor and broadcast the result to the PE row can help save resources.
+ `hardware\src\main\scala\sa\PE.scala`: This file includes the core of the internal PE design in SALO. It internally uses Multiplexer to route the correct dataflow in different stages according to the signal. In the computation, it uses fixed-point arithmetic to minimize the hardware overhead. Specifically, SALO follows the method proposed in Softermax that uses a piece-wise linear function to fit the exponential function in attention mechanism. The slope and y-interception are stored in the lookup-table. We obtain the slope and y-interception by the code in `piecewise_linear/piecewise_linear.py`.
+ `hardware\src\main\scala\sa\WeightedSumModule.scala`: This file includes the weighted sum module design to support window splitting techniques mentioned in the paper.
+ `hardware\src\main\scala\sa\Mem.scala`: This file includes the memory part of SALO.
+ `hardware\src\main\scala\sa\PEArray.scala`: This file includes the PE array design in SALO. The PE design in `PE.scala` is duplicated and connected to build a PE array with the external inverse modules and weighted sum modules. 

## Hardware Synthesis
The Chisel implementation can be compiled to verilog to be further synthesized. To emit the verilog code, you can enter the directory `hardware` and run
```shell
sbt run
```
We use SynopsysDC2016 to synthesize the hardware design to obtain area and power report. We use [FreePDK45nm](https://vlsiarch.ecen.okstate.edu/flows/FreePDK_SRC/osu_freepdk_1.0/lib/files/) technology. To make a fair comparison with [Sanger](https://dl.acm.org/doi/abs/10.1145/3466752.3480125), we follow the 500MHz frequency setting in the hardware synthesis part. To reproduce the report, you need to copy the generated verilog file `PEArray.v`, the technology file `gscl45nm.db`, and the compile script `compile_dc.tcl` to the same directory. Then run the following command in the shell
```shell 
dc_shell-t -f compile_dc.tcl
```
The synthesis report is shown below:
|Parameter|Value|
|-|-|
|Frequency|500MHz|
|Power|532.66mW|
|Area|4.56 $mm^2$|

## Citation
Guan Shen, Jieru Zhao, Quan Chen, Jingwen Leng, Chao Li, and Minyi Guo. 2022. SALO: an efficient spatial accelerator enabling hybrid sparse attention mechanisms for long sequences. In Proceedings of the 59th ACM/IEEE Design Automation Conference (DAC '22).
