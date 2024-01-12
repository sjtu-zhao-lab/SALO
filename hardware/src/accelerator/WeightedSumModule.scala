package accelerator

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

class WeightedSumModule(dim: Int) (implicit ev: Arithmetic[FixedPoint]) extends Module {// inputType: 5+4, probType: 2+7, accType: 10+8
    import ev._
    val inputType = FixedPoint(9.W, 4.BP)
    val probType = FixedPoint(9.W, 7.BP)
    val accType = FixedPoint(18.W, 8.BP) // Large enough to avoid overflow
    val expType = Input(SInt(5.W))

    val io = IO(new Bundle {
        val in_sum = Input(accType)
        val in_exp = Input(expType)
        val control = Input(UInt(2.W))
        val out_port = Output(accType)
    })

    val sum = Reg(accType)
    val exp = Reg(expType)
    val c  = Reg(accType)
    val c_exp = Reg(expType)

    val w1 = Reg(accType)
    val w2 = Reg(accType)
    val buffer = Reg(Vec(dim, accType))
    val (cnt, ismax) = Counter(io.control===3.U, dim)

    val shifted_sum = Wire(accType)
    val shifted_in_sum = Wire(accType)
    val max_exp = Wire(expType)

    shifted_sum := DontCare
    shifted_in_sum := DontCare
    max_exp := DontCare
    io.out_port := DontCare

    when(io.control===0.U){
        for (i <- 0 until dim) {
            buffer(i) := 0.F(8.BP)
        }
        sum := 0.F(8.BP)
        exp := 0.S(5.W)
    }.elsewhen(io.control===1.U){
        when(exp < io.in_exp) {
            shifted_sum := sum >> (io.in_exp - exp).asUInt()
            shifted_in_sum := io.in_sum
            max_exp := io.in_exp
        }.otherwise {
            shifted_sum := sum
            shifted_in_sum := io.in_sum >> (exp - io.in_exp).asUInt()
            max_exp := exp
        }
        c := shifted_sum + shifted_in_sum
        c_exp := max_exp
        w1 := shifted_sum
        w2 := shifted_in_sum
        for (i <- 0 until dim) {
            buffer(i) := buffer(i)
        }
    }.elsewhen(io.control===2.U){
        w1 := (((w1 << 4).asUInt() / c.asUInt()) >> 4).asFixedPoint(8.BP)
        w2 := ((1.U / c.asUInt()) >> 4).asFixedPoint(8.BP)
        sum := c
        exp := c_exp
        for (i <- 0 until dim) {
            buffer(i) := buffer(i)
        }
    }.elsewhen(io.control===3.U){
        buffer(cnt) := w1 * buffer(cnt) + w2 * io.in_sum
        io.out_port := buffer(cnt)
    }
}