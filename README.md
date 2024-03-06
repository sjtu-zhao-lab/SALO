# SALOV2
This repository implements the proposed spatial accelerator design in the paper Hardware-Software Co-Design Enabling Static and Dynamic Sparse Attention Mechanisms (TCAD 2024)

Please cite our paper if you find SALO useful for your research:
```
@article{Shen2022SALOAE,
  title={SALO: an efficient spatial accelerator enabling hybrid sparse attention mechanisms for long sequences},
  author={Guan Shen and Jieru Zhao and Quan Chen and Jingwen Leng and C. Li and Minyi Guo},
  journal={Proceedings of the 59th ACM/IEEE Design Automation Conference},
  year={2022},
  url={https://api.semanticscholar.org/CorpusID:250113485}
}
```
and
```
@ARTICLE{10460307,
  author={Zhao, Jieru and Zeng, Pai and Shen, Guan and Chen, Quan and Guo, Minyi},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={Hardware-Software Co-Design Enabling Static and Dynamic Sparse Attention Mechanisms}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Sparse matrices;Task analysis;Transformers;Graphics processing units;Complexity theory;Vectors;Runtime;Attention acceleration;static/dynamic sparsity},
  doi={10.1109/TCAD.2024.3373592}}
```

## Overview
SALOv2, a hardware-software co-design framework that facilitates efficient processing of static and dynamic sparse attention mechanisms. Effective techniques and designs are proposed at software and hardware levels, making SALOv2 applicable to various scenarios.

## Requirements

-  For software experiments
   -  CUDA SDK >= 10.1
   -  Python >= 3.7
   -  PyTorch >= 1.7.0
   -  Transformers 4.7.0
-  For hardware experiments
   -  JDK 8 or 11
   -  Scala compiler `sbt`. 

## Performance Evaluation
We provide the performance comparison between SALOv2 and CPU/GPU, based on the inference speed. The benchmark code that evaluates the CPU and GPU performance on different workloads mentioned in the paper is located at `benchmark/bench_cpu_gpu.py`. 

To evaluate the performance of SALOv2, we developed a cycle-accurate performance model in `performance_model/performance_model.py`. The performance model estimates the computation FLOPS in each stage and calculates the number of cycles to run.


## Accuracy Evaluation

1.  Train a model with SALOv2 sparse attention. 

We provide scripts for training in the `scripts/` sub-directory. For example, to train a SALOv2_sparse BERT-Base model on SQuAD, you can execute `scripts/train_sparse_on_squad.sh`. Note that you have to pass in an appropriate configuration file, which you can find in `configs/`. You can skip this step if you choose to load a fine-tuned checkpoint directly.

2.  Evaluate the fine-tuned model. 

We also provide scripts for evaluation in `scripts/`. For example, to evaluate the sparse model from the last step, you can execute `scripts/eval_sparse_on_squad.sh`. If you need to load a checkpoint from a non-standard location, be sure to change the path in the script. When the evaluation is complete, the script should print out the accuracy.

## Hardware Design
-  `hardware/`: This sub-directory holds code related to the hardware implementation of SALOv2.
   -  `src/accelerator`: This sub-directory contains the main source code of spatial accelerator:
      -  `Arithmetic.scala`: This file includes the supportive fixed-point arithmetic functions (Bitwidth up/down-cast) that are useful in the hardware implementation.
      -  `PE.scala`: This file includes the core of the internal PE design in SALOv2. It internally uses Multiplexer to route the correct dataflow in different stages according to the signal. In the computation, it uses fixed-point arithmetic to minimize the hardware overhead. Specifically, SALOv2 follows the method proposed in Softermax that uses a piece-wise linear function to fit the exponential function in attention mechanism. The slope and y-interception are stored in the lookup-table. We obtain the slope and y-interception by the code in `piecewise_linear/piecewise_linear.py`.
      -  `PEArray.scala`: This file includes the PE array design in SALOv2. The PE design in `PE.scala` is duplicated and connected to build a PE array with the external weighted sum modules.
      -  `WeightedSumModule.scala`: This file includes the weighted sum module design to support window splitting techniques mentioned in the paper.
   -  `src/matching`: This sub-directory contains the main source code of pattern matching module:
      -  `PE.scala`: This file includes the core of the internal PE design of pattern matching module in SALOv2.
      -  `BitonicSort.scala`: The bitonic-sorter in pattern matching module which selects the topk elements of a sequence.
      -  `LocalModule.scala`: The sliding window sorter in pattern matching module which performs a sliding window summation and outputs the maximum value.

## Software Design
-  `benchmark/bench_cpu_gpu.py`: This script benchmarks on CPU and GPU.
-  `configs/`: This sub-directory contains configuration files for dense models, sparse models, and static sparsity (BigBird, Longformer, etc.).
-  `data/`: This sub-directory is intended for storing manually downloaded datasets. Only the CLOTH dataset needs to be stored here, because GLUE and SQuAD are downloaded and managed automatically by the transformers library.
-  `outputs/`: This sub-directory is intended for storing training and evaluation results.
-  `performance_model/performance_model.py`: The performance model estimates the computation FLOPS in each stage and calculates the number of cycles to run.
-  `scripts/`: This sub-directory holds the shell scripts for running experiments.
-  `modeling_bert​​​.py`: These files contain implementations of the BERT models, supporting both dense and sparse attention.
-  `salo_sparse.py`: This file contains an implementation of the sparse attention algorithm of SALOv2.
-  `modeling_static_spattn.py`: This file implements some attention mechanisms with static sparsity.
-  `run_<task>​​​​​​​.py`: These files are intended for training or evaluating models on GLUE, SQuAD or CLOTH.
-  `quant_utils.py`: This file contains some helper functions related to quantization.


## Hardware Synthesis Report
The Chisel implementation can be compiled to verilog to be further synthesized. To emit the verilog code, you can enter the directory `hardware` and run
```shell
sbt run
```
We use SynopsysDC2016 to synthesize the hardware design to obtain area and power report. We use [FreePDK45nm](https://vlsiarch.ecen.okstate.edu/flows/FreePDK_SRC/osu_freepdk_1.0/lib/files/) technology.
The synthesis report is shown below:
|Parameter|Value|
|-|-|
|Frequency|1 $GHz$|
|Power|3.75 $W$|
|Area|5.97 $mm^2$|
