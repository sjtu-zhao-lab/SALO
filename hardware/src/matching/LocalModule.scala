package matching

import chisel3._
import chisel3.util._
import math.pow
import chisel3.stage.ChiselStage

object LocalModule extends App {
  (new ChiselStage).emitVerilog(new LocalModule(8, 64), args)
}

class LocalModule[T <: Data](wide:Int, size:Int) extends Module {
    val io = IO(new Bundle {
        val numberIn1 = Input(UInt(8.W))
        val numberIn2 = Input(UInt(8.W))
        val stage = Input(UInt(2.W))
        val slideWindowControl = Input(UInt(log2Ceil(size).W))
        val outI = Output(UInt(log2Ceil(2*size).W))
        val out = Output(UInt(8.W))
    })

    val stack1 = Module(new Stack(wide, size))
    val stack2 = Module(new Stack(wide, size))

    stack1.io.in := io.numberIn1
    stack2.io.in := io.numberIn2

    val fifo1in = Wire(UInt(wide.W))
    val fifo2in = Wire(UInt(wide.W))
    stack1.io.enable := (io.stage===3.U)
    stack2.io.enable := (io.stage===3.U)

    fifo1in := Mux(io.stage===3.U,stack2.io.out, io.numberIn1)
    fifo2in := Mux(io.stage===3.U,stack1.io.out, io.numberIn2)

    val fifo1 = Module(new myShiftRegister(wide, size))
    val fifo2 = Module(new myShiftRegister(wide, size))

    val reg1 = RegInit(0.U(8.W))
    val reg2 = RegInit(0.U(8.W))
    val regMax = RegInit(0.U(8.W))
    val regIdx = RegInit(0.U(log2Ceil(2*size).W))

    fifo1.io.in := fifo1in
    fifo2.io.in := fifo2in

    val acc1 = Wire(UInt(8.W))
    acc1 := reg1 + fifo1.io.out(0)
    val acc2 = Wire(UInt(8.W))
    acc2 := reg2 + fifo2.io.out(0)

    val outIdx = VecInit(Seq.fill(2 * size - 1)(0.U(log2Ceil(2*size).W)))
    for( i <- 0 until 2*size-1){
        outIdx(i) := i.U
    }

    val sub1 = Seq.tabulate(size)(idx => {
        fifo1.io.out(idx)
    })

    val sub2 = Seq.tabulate(size)(idx => {
        fifo2.io.out(idx)
    })

    val condition1 = Seq.tabulate(size)(idx => {
        idx.U -> sub1(idx)
    })
    val condition2 = Seq.tabulate(size)(idx => {
        idx.U -> sub2(idx)
    })
    val subValue1 = Wire(UInt(wide.W))
    val subValue2 = Wire(UInt(wide.W))
    subValue1 := MuxLookup(io.slideWindowControl, DontCare, condition1)
    subValue2 := MuxLookup(io.slideWindowControl, DontCare, condition2)
    
    val temp1 = Wire(UInt(8.W))
    temp1 := Mux(reg1 > reg2, reg1, reg2)
    val temp2 = Wire(UInt(8.W))
    temp2 := Mux(regMax > temp1, regMax, temp1)

    val ex = io.stage===2.U | io.stage===3.U

    val (cnt, ismax) = Counter(ex, size+1)
    when (io.stage === 0.U){
        reg1 := 0.U
        reg2 := 0.U
        regMax := 0.U
    }.elsewhen(io.stage === 1.U){
        reg1 := acc1
        reg2 := acc2 
    }.elsewhen(io.stage === 2.U | io.stage === 3.U){
        when(temp2 > regMax){
            when(reg1 > reg2){
                regIdx := outIdx(cnt-1.U)
            }.otherwise{
                regIdx := outIdx((2 * size - 1).U - cnt)
            }
        }
        reg1 := acc1 - subValue1
        reg2 := acc2 - subValue2
        regMax := temp2
    }

    io.out := regMax

    io.outI := Mux(regIdx > (size-1).U, regIdx-(io.slideWindowControl-1.U), regIdx)
}

