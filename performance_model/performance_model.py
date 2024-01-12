import math

class TranSABench():
    def __init__(self, seq_len=4096, window_size=64, dim=768, num_heads=12, pe_rows=32, pe_cols=32, pe_glbs=1, freq=1):
        self.seq_len = seq_len
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_size = self.dim // self.num_heads
        self.pe_rows = pe_rows
        self.pe_cols = pe_cols
        self.pe_glbs = pe_glbs
        self.freq = freq
        print(f"seq_len={seq_len}, window_size={window_size}")

    def eval(self):
        hardware_gops = (self.pe_rows * self.pe_cols + self.pe_glbs * self.pe_rows + self.pe_cols)* 2 * self.freq
        stage1_gop = self.seq_len * (self.window_size + self.pe_glbs * 2) * self.hidden_size * self.num_heads * 2 / 1e9
        stage2_gop = self.seq_len * (self.window_size + self.pe_glbs * 2) * 2 / 1e9
        stage3_gop = self.seq_len * (self.window_size + self.pe_glbs * 2) / 1e9
        stage4_gop = self.seq_len * (self.window_size + 2 * self.pe_glbs) / 1e9
        stage5_gop = self.seq_len * (self.window_size + self.pe_glbs * 2) * self.hidden_size * self.num_heads * 2 / 1e9
        total_gop = stage1_gop + stage2_gop + stage3_gop + stage4_gop + stage5_gop
        #total_gop = stage1_gop + stage2_gop + stage3_gop + stage5_gop
        
        # The cycles of stage 1 -- q x k
        cycle_stage1 = self.hidden_size

        # The cycles of stage 2 -- exponential
        cycle_stage2 = 1

        # The cycles of stage 3 -- sum of exponential
        cycle_stage3 = 1

        # # The cycles of waiting -- waiting for the sum and inverse
        cycle_nops = self.pe_cols

        # # The cycles of stage 4 -- multiplication
        cycle_stage4 = 1

        # The cycles of stage 5
        cycle_stage5 = self.hidden_size

        # The cycles of flush the reg for the next iteration
        cycle_flush = 1

        # The cycles of the PE Array output the remaining results
        cycle_remaining = self.pe_cols + self.pe_glbs - 1

        num_iterations = math.ceil(self.seq_len / self.pe_rows) * math.ceil(self.window_size / self.pe_cols) * self.num_heads

        total_cycle = (cycle_stage1 + cycle_stage2 + cycle_stage3 + cycle_nops + cycle_stage4 + cycle_stage5) * num_iterations + cycle_flush * (num_iterations - 1) + cycle_remaining
        #total_cycle = (cycle_stage1 + cycle_stage2 + cycle_stage3 +  cycle_stage5) * num_iterations + cycle_flush * (num_iterations - 1) + cycle_remaining

        latency = total_cycle / self.freq / 1e9
        utilization = float(total_gop) / (latency * hardware_gops)
        print(f"before optimization")
        print(f"The utilization is : {utilization}")
        print(f"The latency is: {latency*1000}ms")



if __name__ == "__main__":
    # ViL-stage-1
    benchmark = TranSABench(seq_len=56*56, window_size=15*15, dim=384, num_heads=6)
    benchmark.eval()
    # ViL-stage-2
    benchmark = TranSABench(seq_len=28*28, window_size=15*15, dim=384, num_heads=6)
    benchmark.eval()
    # Longformer
    benchmark = TranSABench(seq_len=4096, window_size=512, dim=768, num_heads=12, pe_rows=32, pe_cols=32, pe_glbs=1, freq=1)
    benchmark.eval()