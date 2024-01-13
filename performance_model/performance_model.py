import math

class SALOBench():
    def __init__(self, seq_len=4096,block_size=64, window_size=8, nums_global=1, nums_random=3, dim=768, num_heads=12, pe_rows=8, pe_cols=8, pe_glbs=1, pe_randoms=3, freq=1):
        self.seq_len = seq_len
        self.block_size = block_size
        self.window_size = window_size
        self.nums_global = nums_global
        self.nums_random = nums_random
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_size = self.dim // self.num_heads
        self.pe_rows = pe_rows
        self.pe_cols = pe_cols
        self.pe_glbs = pe_glbs
        self.pe_randoms = pe_randoms
        self.freq = freq

    def eval(self):
        hardware_gops = (self.pe_rows * self.pe_cols + (self.pe_randoms + (2 * self.pe_glbs)) * self.pe_rows) * 2 * self.freq
        stage1_gop = (self.seq_len / self.block_size)**2 * self.block_size * (self.window_size + self.nums_random + 2*self.nums_global) * self.dim * 2 / 1e9
        stage2_gop = (self.seq_len / self.block_size)**2 * self.block_size * (self.window_size + self.nums_random + 2*self.nums_global) * 2 / 1e9
        stage3_gop = (self.seq_len / self.block_size)**2 * self.block_size * (self.window_size + self.nums_random + 2*self.nums_global) / 1e9
        stage4_gop = (self.seq_len / self.block_size)**2 * self.block_size * (self.window_size + self.nums_random + 2*self.nums_global) * self.dim * 2 / 1e9
        total_gop = stage1_gop + stage2_gop + stage3_gop + stage4_gop

        # The cycles of stage 1 -- q x k
        cycle_stage1 = self.hidden_size

        # The cycles of stage 2 -- exponential
        cycle_stage2 = 1

        # The cycles of stage 3 -- sum of exponential
        cycle_stage3 = 1

        cycle_stage4 = self.hidden_size

        # The cycles of flush the reg for the next iteration
        cycle_flush = 1

        # The cycles of the PE Array output the remaining results
        cycle_remaining = self.pe_cols + self.pe_randoms + self.pe_glbs - 1

        num_iterations = (self.seq_len / self.block_size)**2 * math.ceil(self.block_size / self.pe_rows) * math.ceil(self.window_size / self.pe_cols) * self.num_heads

        #total_cycle = (cycle_stage1 + cycle_stage2 + cycle_stage3 + cycle_nops + cycle_stage4 + cycle_stage5) * num_iterations + cycle_flush * (num_iterations - 1) + cycle_remaining
        total_cycle = (cycle_stage1 + cycle_stage2 + cycle_stage3 +  cycle_stage4) * num_iterations + cycle_flush * (num_iterations - 1) + cycle_remaining

        qkv_sparse_lat = total_cycle / self.freq / 1e9 
        utilization = float(total_gop) / (qkv_sparse_lat * hardware_gops)

        qkv_proj = self.seq_len * self.dim * self.dim * 3 * 2
        qkv_proj_lat = qkv_proj * 0.14 / 1e9 /hardware_gops / utilization
                
        out_proj_flops = self.seq_len * self.dim * self.dim * 2
        out_proj_lat = out_proj_flops * 0.14 / hardware_gops / 1e9 / utilization
        
        print(f"The utilization is : {utilization}")
        return qkv_proj_lat, qkv_sparse_lat, out_proj_lat


class MatchingBench():
    def __init__(self, seq_len=4096, block_size=64, dim=768, num_heads=12, freq=1):
        self.seq_len = seq_len
        self.block_size = block_size
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_size = self.dim // self.num_heads
        self.freq = freq

    def eval(self):
        hardware_gops = (self.block_size * self.block_size) * 2 * self.freq
        stage1_gop = self.seq_len * self.seq_len * self.hidden_size * self.num_heads * 2 / 1e9
        stage2_gop = self.seq_len * self.seq_len / 1e9

        total_gop = stage1_gop + stage2_gop
        
        # The cycles of stage 1 -- q x k
        cycle_stage1 = self.hidden_size

        # The cycles of stage 2 -- transmit qk to next pe
        cycle_stage2 = 1

        cycle_flush = 1

        # The cycles of the PE Array output the remaining results
        cycle_remaining = 2*self.block_size + 2

        num_iterations = math.ceil(self.seq_len / self.block_size) * math.ceil(self.seq_len / self.block_size) * self.num_heads

        total_cycle = (cycle_stage1 + cycle_stage2) * num_iterations + cycle_flush * (num_iterations - 1) + cycle_remaining

        perprocess_lat = total_cycle / self.freq / 1e9
        utilization = float(total_gop) / (perprocess_lat * hardware_gops)

        return perprocess_lat


if __name__ == "__main__":
    seq_lens = [128, 384, 512]
    seq_lens = [384]
    for seq_len in seq_lens:
        benchmark = MatchingBench(seq_len=seq_len, block_size=64, dim=768, num_heads=16, freq=1)
        perprocess_lat = benchmark.eval()
        benchmark = SALOBench(seq_len=seq_len,block_size=64, window_size=8, nums_global=1,nums_random=3, dim=768, num_heads=16, pe_rows=64, pe_cols=8, pe_glbs=1, pe_randoms=3, freq=1)
        qkv_proj_lat, qkv_sparse_lat, out_proj_lat = benchmark.eval()
        acc_total_lat = qkv_proj_lat + qkv_sparse_lat + out_proj_lat
        print(f"128 max latency: {(32 * perprocess_lat + ((64 * 12 + 8 ) / 128) * (qkv_proj_lat + qkv_sparse_lat + out_proj_lat)/((seq_len / 64)**2)) * 1000:.5f}")