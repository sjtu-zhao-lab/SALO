;buildInfoPackage: chisel3, version: 3.4.3, scalaVersion: 2.12.12, sbtVersion: 1.3.10
circuit Mem : 
  module Mem : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enable : UInt<1>, flip write : UInt<1>, flip addr : UInt<16>, flip dataIn : UInt<8>, dataOut : UInt<8>}
    
    smem mem : UInt<8>[16384], undefined @[Mem.scala 18:24]
    node _T = bits(io.addr, 13, 0)
    write mport MPORT = mem[_T], clock
    MPORT <= io.dataIn
    wire _io_dataOut_WIRE : UInt @[Mem.scala 21:25]
    _io_dataOut_WIRE is invalid @[Mem.scala 21:25]
    when io.enable : @[Mem.scala 21:25]
      _io_dataOut_WIRE <= io.addr @[Mem.scala 21:25]
      node _io_dataOut_T = or(_io_dataOut_WIRE, UInt<14>("h00")) @[Mem.scala 21:25]
      node _io_dataOut_T_1 = bits(_io_dataOut_T, 13, 0) @[Mem.scala 21:25]
      read mport io_dataOut_MPORT = mem[_io_dataOut_T_1], clock @[Mem.scala 21:25]
      skip @[Mem.scala 21:25]
    io.dataOut <= io_dataOut_MPORT @[Mem.scala 21:14]
    
