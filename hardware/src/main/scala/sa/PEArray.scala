package sa

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint
import chisel3.stage.ChiselStage

class PEArray(num_rows: Int, num_cols: Int, num_hglbs: Int, num_vglbs: Int, dim: Int) (implicit ev: Arithmetic[FixedPoint]) extends Module {
    import ev._
    val inputType = FixedPoint(9.W, 4.BP)
    val probType = FixedPoint(9.W, 7.BP)
    val accType = FixedPoint(18.W, 8.BP) // Large enough to avoid overflow
    val expType = Input(SInt(5.W))
    val io = IO(new Bundle {
        val kv_ports = Input(Vec(num_rows + num_cols + num_vglbs - 1, inputType))
        val q_ports = Input(Vec(num_rows + num_hglbs, inputType))
        val stage_ports = Input(Vec(num_rows + num_hglbs, UInt(3.W)))
        val weight_control = Input(UInt(2.W))
        val out_ports = Output(Vec(num_rows + num_hglbs, accType))
    })

    val local_pes = for (i <- 0 until num_rows) yield 
        for (j <- 0 until num_cols) yield Module(new PE(j == 0))

    val global_col_pes = for (i <- 0 until num_rows) yield
        for (j <- 0 until num_vglbs) yield Module(new PE(false))

    val global_row_pes = for (i <- 0 until num_hglbs) yield
        for (j <- 0 until num_cols) yield Module(new PE(j==0))

    val inv_modules = for (i <- 0 until num_rows+num_hglbs) yield Module(new InverseModule)

    val weighted_sum_modules = for (i <- 0 until num_rows+num_hglbs) yield Module(new WeightedSumModule(dim))

    for (i <- 0 until num_cols) {
        local_pes(0)(num_cols-i-1).io.in_kv := io.kv_ports(i)
    }

    for (i <- 1 until num_rows) {
        local_pes(i)(0).io.in_kv := io.kv_ports(num_cols+i-1)
    }

    for (i <- 1 until num_rows) {
        for (j <- 1 until num_cols) {
            local_pes(i)(j).io.in_kv := local_pes(i-1)(j-1).io.out_kv
        }
    }

    for (i <- 0 until num_rows) {
        for (j <- 0 until num_cols) {
            local_pes(i)(j).io.in_q := (if (j==0) io.q_ports(i) else local_pes(i)(j-1).io.out_q)
            local_pes(i)(j).io.in_stage := (if (j==0) io.stage_ports(i) else local_pes(i)(j-1).io.out_stage)
            local_pes(i)(j).io.in_sum := (if (j==0) DontCare else local_pes(i)(j-1).io.out_sum)
            local_pes(i)(j).io.in_sum_exp := (if (j==0) DontCare else local_pes(i)(j-1).io.out_sum_exp)
            local_pes(i)(j).io.in_inv_sum := inv_modules(i).io.out_inv_sum
            local_pes(i)(j).io.in_inv_sum_exp := inv_modules(i).io.out_inv_sum_exp
        }
    }

    for (i <- 0 until num_rows) {
        for (j <- 0 until num_vglbs) {
            global_col_pes(i)(j).io.in_q := (if (j==0) local_pes(i)(num_cols-1).io.out_q else global_col_pes(i)(j-1).io.out_q)
            global_col_pes(i)(j).io.in_stage := (if (j==0) local_pes(i)(num_cols-1).io.out_stage else global_col_pes(i)(j-1).io.out_stage)
            global_col_pes(i)(j).io.in_sum := (if (j==0) local_pes(i)(num_cols-1).io.out_sum else global_col_pes(i)(j-1).io.out_sum)
            global_col_pes(i)(j).io.in_sum_exp := (if (j==0) local_pes(i)(num_cols-1).io.out_sum_exp else global_col_pes(i)(j-1).io.out_sum_exp)
            global_col_pes(i)(j).io.in_kv := io.kv_ports(num_rows+num_cols-1+j)
            global_col_pes(i)(j).io.in_inv_sum := inv_modules(i).io.out_inv_sum
            global_col_pes(i)(j).io.in_inv_sum_exp := inv_modules(i).io.out_inv_sum_exp
        }
    }

    for (i <- 0 until num_hglbs) {
        for (j <- 0 until num_cols) {
            global_row_pes(i)(j).io.in_q := (if (j==0) io.q_ports(num_rows+i) else global_row_pes(i)(j-1).io.out_q)
            global_row_pes(i)(j).io.in_stage := (if (j==0) io.stage_ports(num_rows+i) else global_row_pes(i)(j-1).io.out_stage)
            global_row_pes(i)(j).io.in_sum := (if (j==0) DontCare else global_row_pes(i)(j-1).io.out_sum)
            global_row_pes(i)(j).io.in_sum_exp := (if (j==0) DontCare else global_row_pes(i)(j-1).io.out_sum_exp)
            global_row_pes(i)(j).io.in_kv := (if (i==0) local_pes(num_rows-1)(j).io.out_kv else global_row_pes(i-1)(j).io.out_kv)
            global_row_pes(i)(j).io.in_inv_sum := inv_modules(num_rows+i).io.out_inv_sum
            global_row_pes(i)(j).io.in_inv_sum_exp := inv_modules(num_rows+i).io.out_inv_sum_exp
        }
    } 

    for (i <- 0 until num_rows) {
        inv_modules(i).io.in_sum := (if (num_vglbs==0) local_pes(i)(num_cols-1).io.out_sum else global_col_pes(i)(num_vglbs-1).io.out_sum)
        inv_modules(i).io.in_exp := (if (num_vglbs==0) local_pes(i)(num_cols-1).io.out_sum_exp else global_col_pes(i)(num_vglbs-1).io.out_sum_exp)
    }

    for (i <- 0 until num_hglbs) {
        inv_modules(num_rows+i).io.in_sum := global_row_pes(i)(num_cols-1).io.out_sum
        inv_modules(num_rows+i).io.in_exp := global_row_pes(i)(num_cols-1).io.out_sum_exp
    }

    for (i <- 0 until num_rows) {
        weighted_sum_modules(i).io.in_sum := (if (num_vglbs==0) local_pes(i)(num_cols-1).io.out_sum else global_col_pes(i)(num_vglbs-1).io.out_sum)
        weighted_sum_modules(i).io.in_exp := (if (num_vglbs==0) local_pes(i)(num_cols-1).io.out_sum_exp else global_col_pes(i)(num_vglbs-1).io.out_sum_exp)
        weighted_sum_modules(i).io.control := io.weight_control
        io.out_ports(i) := weighted_sum_modules(i).io.out_port
    }

    for (i <- 0 until num_hglbs) {
        weighted_sum_modules(num_rows+i).io.in_sum := global_row_pes(i)(num_cols-1).io.out_sum
        weighted_sum_modules(num_rows+i).io.in_exp := global_row_pes(i)(num_cols-1).io.out_sum_exp
        weighted_sum_modules(num_rows+i).io.control := io.weight_control
        io.out_ports(num_rows+i) := weighted_sum_modules(num_rows+i).io.out_port
    }
}

object PEArray extends App {
  (new ChiselStage).emitVerilog(new PEArray(32, 32, 1, 1, 64), args)
}
