package matching

import chisel3._


class myReg(wide:Int) extends Module{
    val io = IO(new Bundle{
        val inIn = Input(UInt(wide.W))
        val outIn = Input(UInt(wide.W))
        val inOut = Output(UInt(wide.W))
        val outOut = Output(UInt(wide.W))
        val enable = Input(Bool())
    })

    val reg = RegInit(0.U(wide.W))
    val value = Wire(UInt(wide.W))

    value := Mux(io.enable,io.outIn,io.inIn)

    reg := value

    io.inOut := reg
    io.outOut := reg

}

class Stack(wide:Int, size:Int) extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(wide.W))
    val out = Output(UInt(wide.W))
    val enable = Input(Bool())
  })

  val reg = Seq.fill(size)(Module(new myReg(wide)))
  reg(0).io.inIn := io.in
  reg(0).io.enable := io.enable
  for (i <- 0 until size - 1){
    reg(i+1).io.enable := reg(i).io.enable
    reg(i+1).io.inIn := reg(i).io.inOut
    reg(i).io.outIn := reg(i+1).io.outOut
  }

  reg(size-1).io.outIn := DontCare

  io.out := reg(1).io.outOut

}
