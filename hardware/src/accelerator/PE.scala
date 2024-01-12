// Refer to Gemmini https://github.com/ucb-bar/gemmini/blob/master/src/main/scala/gemmini/PE.scala
package accelerator

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint
import chisel3.stage.ChiselStage
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

class PE(left_most: Boolean,bits: Int,pointWide: Int) (implicit ev: Arithmetic[FixedPoint]) extends Module {// inputType: 5+4, probType: 2+7, accType: 10+8
    import ev._
    val inputType = FixedPoint(bits.W, pointWide.BP)
    val accType = FixedPoint((2*bits).W, (2*pointWide).BP) // Large enough to avoid overflow
    val expType = Input(SInt(5.W))
    val io = IO(new Bundle {
        val in_q = Input(inputType) // For transmit q and v_sum
        val in_sum = Input(accType)
        val in_sum_exp = Input(expType)
        val in_kv = Input(inputType)

        val in_stage = Input(UInt(2.W))

        val out_q = Output(inputType)
        val out_sum = Output(accType)
        val out_sum_exp = Output(expType)
        val out_kv = Output(inputType)
        val out_stage = Output(UInt(2.W))

        val out_acc = Output(accType) // For test
    })
    val q = Reg(inputType)
    val kv = Reg(inputType)
    val sum = Reg(accType)
    val sum_exp = Reg(expType)
    val stage = Reg(UInt(2.W))

    q := io.in_q
    kv := io.in_kv
    stage := io.in_stage

    io.out_q := q
    io.out_kv := kv
    io.out_sum := sum
    io.out_sum_exp := sum_exp
    io.out_stage := stage

    val reg_acc = Reg(accType)

    val reg_prob = Reg(accType)

    io.out_acc := reg_acc
    
    val fraction_bits = Wire(inputType)
    val integer_bits = Wire(expType)

    val reg_max_exp = Reg(expType)
    val max_exp = Reg(expType)
    val shifted_acc = Wire(accType)
    val shifted_sum = Wire(accType)

    val slopes = Seq(0.75532268, 0.89775075, 1.06735514, 1.27081757)
    val intercepts = Seq(0.99742072, 0.96181372, 0.87701153, 0.72441471)
    val lut_k = RegInit(VecInit(slopes.map((x:Double) => FixedPoint.fromDouble(x, bits.W, pointWide.BP))))
    val lut_b = RegInit(VecInit(intercepts.map((x:Double) => FixedPoint.fromDouble(x, (2*bits).W, (2*pointWide).BP))))

    val k = Wire(inputType)
    val b = Wire(accType)

    integer_bits := reg_acc(2*bits-1, 2*pointWide+3).asSInt()
    fraction_bits := reg_acc(2*pointWide+2, 3).zext().asFixedPoint(inputType.binaryPoint) // Shift 3 bits to divide sqrt(dk)
    k := lut_k(fraction_bits(2*pointWide-1, 2*pointWide-2))
    b := lut_b(fraction_bits(2*pointWide-1, 2*pointWide-2))

    val oprand1 = Wire(inputType)
    val oprand2 = Wire(inputType)
    val oprand3 = Wire(accType)
    val product = Wire(accType)
    val oprand4 = Wire(accType)
    val result = Wire(accType)


    shifted_acc := reg_acc
    shifted_sum := io.in_sum

    oprand1 := MuxLookup(io.in_stage, DontCare, Array(0.U->io.in_q, 1.U->k, 3.U->reg_acc))
    oprand2 := MuxLookup(io.in_stage, DontCare, Array(0.U->io.in_kv, 1.U->fraction_bits, 3.U->io.in_kv))
    oprand3 := MuxLookup(io.in_stage, DontCare, Array(0.U->reg_acc, 1.U->b, 2.U->shifted_sum, 3.U->(if (left_most) accType.zero else io.in_sum)))
    
    product := Mux(io.in_stage===2.U, shifted_acc, oprand1 * oprand2)
    oprand4 := Mux(io.in_stage===3.U, product, Mux(io.in_stage===1.U, product >> pointWide, product))
    result := oprand3 + oprand4

    when(io.in_stage===0.U) {
        reg_acc := result
        reg_prob := DontCare
        sum := DontCare
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
        sum := shifted_acc + shifted_sum
        sum_exp := max_exp        
    }.elsewhen(io.in_stage===3.U) {
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
        sum := shifted_acc + shifted_sum
        sum_exp := max_exp  
    }
}

object PE extends App {
  (new ChiselStage).emitVerilog(new PE(false,8,4), args)
}