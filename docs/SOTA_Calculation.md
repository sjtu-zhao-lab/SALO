## Latency comsumption
The evaluation is conducted on the same SQuAD v1.1 dataset with the same input sequence length of 384.
Assuming all the designs use 128 multipliers clocked at 1GHz.

The GOP of attention in SQuAD v1.1 is (1-layer BERT):

(384 $\times$ 768 $\times$ 768 $\times$ 3)+(384 $\times$ 768 $\times$ 384)+(384 $\times$ 384 $\times$ 768)+(384 $\times$ 768 $\times$ 768)=1.05GOP. 

### Energon throughput
From the Table 4 of Sanger, we have the throughput of A3, SpAtten, FTRANS and Sanger.
![image](https://github.com/sjtu-zhao-lab/SALO/assets/103621266/2e5d7766-68a8-4160-a294-b70a9bc6a0b4)
Energon shows 1.7 speedup than SpAtten, so the throughput of Energon is 360*1.7 = 612 GOP/s.

### DTQAtten throughput
From the Table 1 of DTQAtten, we have the throughput of DTQAtten.
![image](https://github.com/sjtu-zhao-lab/SALO/assets/103621266/e6ec08b3-5c6a-41c3-aede-d1985a8d5b18)

the throughput of DTQAtten is 952 GOP/s.

### FACT throughput
From the Table 3 of FACT, we have the throughput of FACT.
![image](https://github.com/sjtu-zhao-lab/SALO/assets/103621266/3285ca31-1324-477d-a8bb-f29e7bdf01f2)
the throughput of FACT is 928 GOP/s.

### Results
All 3 throughputs are data obtained by scaling down to a 128 multiplier. So we don't need to do additional scaling.

So, the attention time of Energon is 1.05Gop / 612 GOP/s = 1.72ms. the attention time of DTQAtten is 1.05Gop / 952 GOP/s = 1.10ms. the attention time of Energon is 1.05Gop / 928 GOP/s = 1.13ms.
For Sanger and SALOv2: We measure the performance with the same workload using their performance
models. The attention times of Sanger and SALO2 are 1.44ms and 0.96ms, respectively.
