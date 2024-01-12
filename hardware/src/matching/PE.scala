// Refer to Gemmini https://github.com/ucb-bar/gemmini/blob/master/src/main/scala/gemmini/PE.scala
package matching

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

class PE(left_most: Boolean,bits: Int) extends Module {// SInt(bits.W): 5+4, probType: 2+7, SInt((2*bits).W): 10+8
    val io = IO(new Bundle {
        val in_q = Input(UInt(bits.W)) // For transmit q and v_sum
        val row_sum = Input(UInt((2*bits).W))
        val col_sum = Input(UInt((2*bits).W))
        val dia_sum = Input(UInt((2*bits).W))
        val in_k = Input(UInt(bits.W))
        val in_stage = Input(UInt(2.W))
        val in_diaSign = Input(Bool())

        val out_q = Output(UInt(bits.W))
        val out_dia = Output(UInt((2*bits).W))
        val out_row = Output(UInt((2*bits).W))
        val out_col = Output(UInt((2*bits).W))
        val out_k = Output(UInt(bits.W))
        val out_stage = Output(UInt(2.W))
        val out_diaSign = Output(Bool())
        val out_random = Output(UInt((2*bits).W))
    })
    val q = Reg(UInt(bits.W))
    val k = Reg(UInt(bits.W))
    val row = Reg(UInt((2*bits).W))
    val col = Reg(UInt((2*bits).W))
    val dia = Reg(UInt((2*bits).W))
    val qkreg = Reg(UInt((2*bits).W))

    val qktemp = Reg(UInt((2*bits).W))
    val stage = Reg(UInt(2.W))

    io.out_q := q;
    io.out_k := k;

    io.out_dia := DontCare
    io.out_row := row
    io.out_col := col

    stage := io.in_stage
    io.out_stage := stage

    io.out_random := qktemp

    io.out_diaSign := false.B
    

    when(io.in_stage===0.U) {
        row := 0.U
        col := 0.U
        dia := 0.U
        qkreg := 0.U
    }.elsewhen(io.in_stage===1.U){
        q := io.in_q
        k := io.in_k
        qkreg := qkreg + q * k
        row := row
        col := col
        when(io.in_diaSign){
            dia := io.dia_sum
        }.otherwise{
            dia := dia 
        }
        qktemp := qktemp
    }.elsewhen(io.in_stage===2.U){
        row := io.row_sum + qkreg
        col := io.col_sum + qkreg
        io.out_dia := qkreg + dia
        io.out_diaSign := true.B
        io.out_random := qkreg
        qktemp := qkreg
    }
}