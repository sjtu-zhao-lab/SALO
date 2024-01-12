package matching

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint
import chisel3.stage.ChiselStage

class PEArray(num_rows: Int, num_cols: Int, bits: Int, maxTopk: Int = 2) (implicit ev: Arithmetic[FixedPoint]) extends Module {
    import ev._
    val inputType = UInt(bits.W)
    val accType = UInt((2*bits).W) // Large enough to avoid overflow
    val io = IO(new Bundle {
        val k_ports = Input(Vec(num_cols, inputType))
        val q_ports = Input(Vec(num_rows, inputType))
        val stage_ports = Input(Vec(num_rows, UInt(2.W)))

        val localControl = Input(UInt(2.W))
        val window = Input(UInt(log2Ceil(num_cols).W))
        val muxSel = Input(UInt(log2Ceil(num_cols).W))
        val sortSel = Input(UInt(log2Ceil(num_cols+2).W))

        val sortOut = Output(Vec(maxTopk,UInt(log2Ceil(num_cols).W)))
        val localOut = Output(UInt(log2Ceil(2*num_cols).W))
    })

    val local_pes = for (i <- 0 until num_rows) yield 
        for (j <- 0 until num_cols) yield Module(new PE(j == 0,bits))
    
    val randomReg = for( i <- 0 until num_rows+2) yield Reg(Vec(num_cols,UInt((2*bits).W)))
    
    val global_row_reg = for (i <- 0 until num_cols) yield RegInit(0.U(4.W))
    val global_col_reg = for (i <- 0 until num_rows) yield RegInit(0.U(4.W))


    for (i <- 0 until num_cols) {
        local_pes(0)(i).io.in_k := io.k_ports(i)
    }

    for (i <- 1 until num_rows) {
        for (j <- 0 until num_cols) {
            local_pes(i)(j).io.in_k := local_pes(i-1)(j).io.out_k
        }
    }

    for (i <- 0 until num_rows) {
        for (j <- 0 until num_cols) {
            local_pes(i)(j).io.in_q := (if (j==0) io.q_ports(i) else local_pes(i)(j-1).io.out_q)
            local_pes(i)(j).io.in_stage := (if (j==0) io.stage_ports(i) else local_pes(i)(j-1).io.out_stage)
            local_pes(i)(j).io.row_sum := (if (j==0) DontCare else local_pes(i)(j-1).io.out_row)
            local_pes(i)(j).io.col_sum := (if (i==0) DontCare else local_pes(i-1)(j).io.out_col)
            local_pes(i)(j).io.dia_sum := (if (j==0 || i==0) 0.U else local_pes(i-1)(j-1).io.out_dia)
            local_pes(i)(j).io.in_diaSign := (if (j==0 || i==0) false.B else local_pes(i-1)(j-1).io.out_diaSign)
            randomReg(i)(j) := local_pes(i)(j).io.out_random
        }
    }

    for (i <- 0 until num_rows){
        randomReg(num_rows)(i) := local_pes(i)(num_cols-1).io.out_row
    }

    for (i <- 0 until num_cols){
        randomReg(num_rows+1)(i) := local_pes(num_rows-1)(i).io.out_col
    }

    val localModule = Module(new LocalModule(2*bits,num_cols))

    val condition1 = Seq.tabulate(num_cols)(idx => {
        idx.U -> local_pes(num_rows-1)(idx).io.out_dia
    })
    val condition2 = Seq.tabulate(num_rows)(idx => {
        idx.U -> local_pes(idx)(num_cols-1).io.out_dia
    })

    localModule.io.numberIn1 := MuxLookup(io.muxSel,DontCare,condition1)
    localModule.io.numberIn2 := MuxLookup(io.muxSel,DontCare,condition2)

    localModule.io.stage := io.localControl
    localModule.io.slideWindowControl := io.window
    io.localOut := localModule.io.outI

    val sorter = Module(new Bitonic(num_cols, 2*bits, UInt(log2Ceil(num_cols).W), 1, maxTopk))
    val constant = Seq.tabulate(num_cols){
        idx => RegInit(idx.U(log2Ceil(num_cols).W))
    }

    val sortCon = Seq.tabulate(num_rows+2)(idx => {
        idx.U -> randomReg(idx)
    })

    sorter.io.tagIn := DontCare
    sorter.io.payloadIn := constant
    sorter.io.numberIn := MuxLookup(io.sortSel,DontCare,sortCon)

    for (i <- 0 until maxTopk){
        io.sortOut(i) := sorter.io.payloadOut(i+maxTopk)
    }

}

object PEArray extends App {
  (new ChiselStage).emitVerilog(new PEArray(8, 8, 4, 4), args)
}
