package accelerator

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint
import chisel3.stage.ChiselStage

class PEArray(num_rows: Int, num_cols: Int, num_hglbs: Int, num_vglbs: Int,num_random_column: Int, dim: Int,bits: Int,pointWide: Int) (implicit ev: Arithmetic[FixedPoint]) extends Module {
    import ev._
    val inputType = FixedPoint(bits.W, pointWide.BP)
    val accType = FixedPoint((2*bits).W, (2*pointWide).BP) // Large enough to avoid overflow
    val io = IO(new Bundle {
        val kv_ports = Input(Vec(num_rows + num_cols + num_vglbs - 1, inputType))

        //random column input
        val random_kv_ports = Input(Vec(num_rows * num_random_column, inputType))


        val q_ports = Input(Vec(num_rows + num_hglbs, inputType))
        val stage_ports = Input(Vec(num_rows + num_hglbs, UInt(2.W)))
        val weight_control = Input(UInt(2.W))
        val out_ports = Output(Vec(num_rows + num_hglbs, accType))
    })

    val local_pes = for (i <- 0 until num_rows) yield 
        for (j <- 0 until num_cols) yield Module(new PE(j == 0,bits,pointWide))

    val global_col_pes = for (i <- 0 until num_rows) yield
        for (j <- 0 until num_vglbs) yield Module(new PE(false,bits,pointWide))

    val global_row_pes = for (i <- 0 until num_hglbs) yield
        for (j <- 0 until num_cols) yield Module(new PE(j==0,bits,pointWide))

    // random pes
    val random_col_pes = for (i <- 0 until num_rows) yield
        for(j <- 0 until num_random_column) yield Module(new PE(false,bits,pointWide))


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
            local_pes(i)(j).io.in_sum_exp := (if (j==0) DontCare else local_pes(i)(j-1).io.out_sum_exp)
            local_pes(i)(j).io.in_stage := (if (j==0) io.stage_ports(i) else local_pes(i)(j-1).io.out_stage)
            local_pes(i)(j).io.in_sum := (if (j==0) DontCare else local_pes(i)(j-1).io.out_sum)
        }
    }

    for (i <- 0 until num_rows) {
        for (j <- 0 until num_vglbs) {
            global_col_pes(i)(j).io.in_q := (if (j==0) local_pes(i)(num_cols-1).io.out_q else global_col_pes(i)(j-1).io.out_q)
            global_col_pes(i)(j).io.in_stage := (if (j==0) local_pes(i)(num_cols-1).io.out_stage else global_col_pes(i)(j-1).io.out_stage)
            global_col_pes(i)(j).io.in_sum := (if (j==0) local_pes(i)(num_cols-1).io.out_sum else global_col_pes(i)(j-1).io.out_sum)
            global_col_pes(i)(j).io.in_sum_exp := (if (j==0) local_pes(i)(num_cols-1).io.out_sum_exp else global_col_pes(i)(j-1).io.out_sum_exp)
            global_col_pes(i)(j).io.in_kv := io.kv_ports(num_rows+num_cols-1+j)
        }
    }

    
    //random cols connect global
    for (i <- 0 until num_rows){
        for (j <- 0 until num_random_column){
            random_col_pes(i)(j).io.in_q := (if (j==0) global_col_pes(i)(num_vglbs-1).io.out_q else random_col_pes(i)(j-1).io.out_q)
            random_col_pes(i)(j).io.in_stage := (if (j==0) global_col_pes(i)(num_vglbs-1).io.out_stage else random_col_pes(i)(j-1).io.out_stage)
            random_col_pes(i)(j).io.in_sum := (if (j==0) global_col_pes(i)(num_vglbs-1).io.out_sum else random_col_pes(i)(j-1).io.out_sum)
            random_col_pes(i)(j).io.in_sum_exp := (if (j==0) global_col_pes(i)(num_vglbs-1).io.out_sum_exp else random_col_pes(i)(j-1).io.out_sum_exp)
            random_col_pes(i)(j).io.in_kv := io.random_kv_ports(i*num_random_column+j)
        }
    }


    for (i <- 0 until num_hglbs) {
        for (j <- 0 until num_cols) {
            global_row_pes(i)(j).io.in_q := (if (j==0) io.q_ports(num_rows+i) else global_row_pes(i)(j-1).io.out_q)
            global_row_pes(i)(j).io.in_stage := (if (j==0) io.stage_ports(num_rows+i) else global_row_pes(i)(j-1).io.out_stage)
            global_row_pes(i)(j).io.in_sum := (if (j==0) DontCare else global_row_pes(i)(j-1).io.out_sum)
            global_row_pes(i)(j).io.in_sum_exp := (if (j==0) DontCare else global_row_pes(i)(j-1).io.out_sum_exp)
            global_row_pes(i)(j).io.in_kv := (if (i==0) local_pes(num_rows-1)(j).io.out_kv else global_row_pes(i-1)(j).io.out_kv)
        }
    } 

    for (i <- 0 until num_rows) {
        weighted_sum_modules(i).io.in_sum := (if (num_vglbs==0 && num_random_column==0) local_pes(i)(num_cols-1).io.out_sum else if(num_random_column == 0) global_col_pes(i)(num_vglbs-1).io.out_sum else random_col_pes(i)(num_random_column-1).io.out_sum)
        weighted_sum_modules(i).io.in_exp := (if (num_vglbs==0 && num_random_column==0) local_pes(i)(num_cols-1).io.out_sum_exp else if(num_random_column == 0) global_col_pes(i)(num_vglbs-1).io.out_sum_exp else random_col_pes(i)(num_random_column-1).io.out_sum_exp)
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
  (new ChiselStage).emitVerilog(new PEArray(64, 8, 1, 1, 3, 64, 9, 4), args)
}
