// Refer to Gemmini https://github.com/ucb-bar/gemmini/blob/master/src/main/scala/gemmini/PE.scala
package sa

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

// Calculate a * b + c
class MAC[T <: Data](aType: T, bType: T, cType: T, outputType: T)
    (implicit ev: Arithmetic[T]) extends Module {
    
    import ev._

    val io = IO(new Bundle {
        val in_a = Input(aType)
        val in_b = Input(bType)
        val in_c = Input(cType)
        val output = Output(outputType)
    })

    val a = Reg(io.in_a)
    val b = Reg(io.in_b)
    val c = Reg(io.in_c)
    io.output := c.mac(a, b).clippedToWidthOf(outputType)
}

class PE(left_most: Boolean) (implicit ev: Arithmetic[FixedPoint]) extends Module {// inputType: 5+4, probType: 2+7, accType: 10+8
    import ev._
    val inputType = FixedPoint(9.W, 4.BP)
    val probType = FixedPoint(9.W, 7.BP)
    val accType = FixedPoint(18.W, 8.BP) // Large enough to avoid overflow
    val expType = Input(SInt(5.W))
    val io = IO(new Bundle {
        val in_q = Input(inputType) // For transmit q and v_sum
        val in_sum = Input(accType)
        val in_sum_exp = Input(expType)
        val in_kv = Input(inputType)
        val in_inv_sum_exp = Input(expType)
        val in_inv_sum = Input(probType)
        val in_stage = Input(UInt(3.W))
        val out_q = Output(inputType)
        val out_sum = Output(accType)
        val out_sum_exp = Output(expType)
        val out_kv = Output(inputType)
        val out_stage = Output(UInt(3.W))

        val out_acc = Output(accType) // For test
        val out_prob = Output(probType)
    })
    val q = Reg(inputType)
    val kv = Reg(inputType)
    val sum = Reg(accType)
    val sum_exp = Reg(expType)
    val stage = Reg(UInt(3.W))

    q := io.in_q
    kv := io.in_kv
    stage := io.in_stage

    io.out_q := q
    io.out_kv := kv
    io.out_sum := sum
    io.out_sum_exp := sum_exp
    io.out_stage := stage

    val reg_acc = Reg(accType)
    val reg_prob = Reg(inputType)

    io.out_acc := reg_acc
    io.out_prob := reg_prob.asFixedPoint(7.BP)
    
    val fraction_bits = Wire(inputType)
    val integer_bits = Wire(expType)

    val reg_max_exp = Reg(expType)
    val max_exp = Wire(expType)
    val shifted_acc = Wire(accType)
    val shifted_sum = Wire(accType)

    val slopes = Seq(0.75532268, 0.89775075, 1.06735514, 1.27081757)
    val intercepts = Seq(0.99742072, 0.96181372, 0.87701153, 0.72441471)
    val lut_k = RegInit(VecInit(slopes.map((x:Double) => FixedPoint.fromDouble(x, 9.W, 4.BP))))
    val lut_b = RegInit(VecInit(intercepts.map((x:Double) => FixedPoint.fromDouble(x, 18.W, 8.BP))))

    val k = Wire(inputType)
    val b = Wire(accType)

    integer_bits := reg_acc(15, 11).asSInt()
    fraction_bits := reg_acc(10, 3).zext().asFixedPoint(inputType.binaryPoint) // Shift 3 bits to divide sqrt(dk)
    k := lut_k(fraction_bits(7, 6))
    b := lut_b(fraction_bits(7, 6))
    printf(p"reg_acc = ${Binary(reg_acc.asUInt())}\n")
    // printf(p"fraction bits = ${Binary(fraction_bits.asUInt())}\n")
    // printf(p"integer part = $integer_bits\n")

    val oprand1 = Wire(inputType)
    val oprand2 = Wire(inputType)
    val oprand3 = Wire(accType)
    val product = Wire(accType)
    val oprand4 = Wire(accType)
    val result = Wire(accType)

    printf(p"ext_acc = ${Binary(reg_acc(8, 1).zext())}\n")
    printf(p"inv_sum = ${Binary(io.in_inv_sum.asUInt())}\n")
    printf(p"reg_prob = ${Binary(reg_prob.asUInt())}\n")
                                                                                // reg_acc will not be greater than 1
    oprand1 := MuxLookup(io.in_stage, DontCare, Array(0.U->io.in_q, 1.U->k, 3.U->reg_acc(8, 1).zext().asFixedPoint(4.BP), 4.U->reg_prob))
    oprand2 := MuxLookup(io.in_stage, DontCare, Array(0.U->io.in_kv, 1.U->fraction_bits, 3.U->io.in_inv_sum.asFixedPoint(4.BP), 4.U->io.in_kv))
    oprand3 := MuxLookup(io.in_stage, DontCare, Array(0.U->reg_acc, 1.U->b, 2.U->shifted_sum, 3.U->accType.zero, 4.U->(if (left_most) accType.zero else io.in_sum)))
    
    product := Mux(io.in_stage===2.U, shifted_acc, oprand1 * oprand2)
    oprand4 := Mux(io.in_stage===4.U, product >> 3, Mux(io.in_stage===1.U, product >> 4, product))
    result := oprand3 + oprand4

    shifted_acc := DontCare
    shifted_sum := DontCare
    max_exp := DontCare

    when(io.in_stage===0.U) {
        reg_acc := result
        reg_prob := DontCare
        reg_max_exp := DontCare
        sum := DontCare
        sum_exp := DontCare
    }.elsewhen(io.in_stage===1.U) {
        reg_acc := result
        reg_prob := DontCare
        reg_max_exp := integer_bits
        sum := DontCare
        sum_exp := DontCare
    }.elsewhen(io.in_stage===2.U) {
        if (!left_most) {
            when(reg_max_exp < io.in_sum_exp) {
                shifted_acc := reg_acc >> (io.in_sum_exp - reg_max_exp).asUInt()
                shifted_sum := io.in_sum
                max_exp := io.in_sum_exp
            }.otherwise {
                shifted_acc := reg_acc
                shifted_sum := io.in_sum >> (reg_max_exp - io.in_sum_exp).asUInt()
                max_exp := reg_max_exp
            }
        }
        reg_acc := shifted_acc
        reg_prob := DontCare
        reg_max_exp := max_exp
        sum := result
        sum_exp := max_exp        
    }.elsewhen(io.in_stage===3.U) {
        reg_acc := reg_acc
        reg_prob := (result >> (io.in_inv_sum_exp - reg_max_exp).asUInt())(15,7).asFixedPoint(4.BP)
        reg_max_exp := reg_max_exp
        sum := DontCare
        sum_exp := DontCare
    }.elsewhen(io.in_stage===4.U) {
        reg_acc := DontCare
        reg_prob := reg_prob
        reg_max_exp := DontCare
        sum := result
        sum_exp := DontCare
    }
}