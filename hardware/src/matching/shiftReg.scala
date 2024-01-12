package matching

import chisel3._
import chisel3.util._

class myShiftRegister(wide:Int, size:Int) extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(wide.W))
    val out = Vec(size,Output(UInt(wide.W)))
  })

  val reg = Seq.fill(size)(RegNext(0.U(wide.W)))
  reg(0) := io.in
  for (i <- 0 until size - 1){
    reg(i+1) := reg(i)
  }

  for (i <- 0 until size){
    io.out(i) := reg(i)
  }

}
