# SALOV2
This repository implements the proposed spatial accelerator design in the paper Hardware-Software Co-Design Enabling Static and Dynamic Sparse Attention Mechanisms

## Overview
SALOv2, a hardware-software co-design framework that facilitates efficient processing of static and dynamic sparse attention mechanisms. Effective techniques and designs are proposed at software and hardware levels, making SALOv2 applicable to various scenarios.

## Requirements

-  For software experiments
   -  CUDA SDK >= 10.1
   -  Python >= 3.7
   -  PyTorch >= 1.7.0
   -  :hugs: Transformers 4.7.0
-  For hardware experiments
   -  JDK 8 or 11
   -  Scala compiler `sbt`. 

## Software experiments

1.  Evaluate SALO2 performance

    1.  Train a model with SALOv2 sparse attention. 

        We provide scripts for training in the `scripts/` sub-directory. For example, to train a SALOv2_sparse BERT-Base model on SQuAD, you can execute `scripts/train_sparse_on_squad.sh`. Note that you have to pass in an appropriate configuration file, which you can find in `configs/`. You can skip this step if you choose to load a fine-tuned checkpoint directly.

    2.  Evaluate the fine-tuned model. 

        We also provide scripts for evaluation in `scripts/`. For example, to evaluate the sparse model from the last step, you can execute `scripts/eval_sparse_on_squad.sh`. If you need to load a checkpoint from a non-standard location, be sure to change the path in the script. When the evaluation is complete, the script should print out the accuracy.

    3.  Estimate the hardware performance of SALOv2. 

        We implement a simple simulator in `bench_salo.py` that estimates the latency of executing an attention layer on SALOv2.

2.  Comparison with dense attention and static sparse attention.

    1.  Train a model with dense or static sparse attention. 

        We provide dedicated scripts for train models with dense attention (e.g. `scripts/train_dense_on_squad.sh`). To train a model with static sparse attention, you can use the same script as Sanger and pass in an appropriate configuration file (e.g. `bert_base_longformer.json`).

    2.  Evaluate the fine-tuned model. 

        The process is similar to evaluating Sanger models. Note that you also need to use different scripts when evaluating dense models.

3.  Comparison with CPU and GPU.

    You can measure the latency of dense attention on CPU and GPU by executing `bench_cpu_gpu.py`.

## Internals

-  `configs/`: This sub-directory contains configuration files for dense models, sparse models, and static sparsity (BigBird, Longformer, etc.).
-  `data/`: This sub-directory is intended for storing manually downloaded datasets. Only the CLOTH dataset needs to be stored here, because GLUE and SQuAD are downloaded and managed automatically by the :hugs: ​transformers library.
-  `hardware/`: This sub-directory holds code related to the hardware implementation of Sanger. For the sake of clarity, we will describe this part separately in the next section.
   -  `src/main/scala/pe_row`: This sub-directory contains the main source code of the three hardware modules:
      -  `pe_row.scala`: The reconfigurable sparse PE array for computing SDDMM and SpMM.
      -  `mask.scala`: The dense low-bit PE array which produce the attention mask.
      -  `pack.scala`: The pack module which convert the attention mask to the configuration of the sparse PE array.
   -  `src/test/scala/pe_row`: This sub-directory contains the unit tests for the hardware modules.
      -  `pe_row_test.scala`: Unit test for the sparse PE array.
      -  `mask_test.scala`: Unit test for the dense PE array.
      -  `pack_text.scala`: Unit test for the pack module.
-  `outputs/`: This sub-directory is intended for storing training and evaluation results.
-  `scripts/`: This sub-directory holds the shell scripts for running experiments.
-  `bench_cpu_gpu.py`: This script benchmarks dense attention on CPU and GPU.
-  `bench_sanger.py`: This script is used to simulate the hardware performance of Sanger.
-  `modeling_<​​​​model>​​​.py`: These files contain implementations of the BERT, GPT2 and BART models, supporting both dense and sparse attention.
-  `modeling_sanger_attn.py`: This file contains an implementation of the sparse attention algorithm of Sanger, and some helper functions for measuring sparsity and load balance.
-  `modeling_static_spattn.py`: This file implements some attention mechanisms with static sparsity.
-  `run_<task>​​​​​​​.py`: These files are intended for training or evaluating models on GLUE, SQuAD or CLOTH.
-  `quant_utils.py`: This file contains some helper functions related to quantization.



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
