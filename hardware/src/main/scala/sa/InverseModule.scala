package sa

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

class InverseModule(implicit ev: Arithmetic[FixedPoint]) extends Module {
    import ev._
    val inputType = FixedPoint(9.W, 4.BP)
    val probType = FixedPoint(9.W, 7.BP)
    val accType = FixedPoint(18.W, 8.BP) // Large enough to avoid overflow
    val expType = Input(SInt(5.W))
    val io = IO(new Bundle{
        val in_sum = Input(accType)
        val in_exp = Input(expType)
        val out_inv_sum = Output(inputType)
        val out_inv_sum_exp = Output(expType)
    })

    io.out_inv_sum := (((1<<17).U / io.in_sum.asUInt()) >> 9).asFixedPoint(8.BP)
    io.out_inv_sum_exp := io.in_exp
}