package matching

import chisel3._
import chisel3.util._
import math.pow
import chisel3.stage.ChiselStage

object BitonicSort {
  def switch[T <: Data](
      asc: Boolean,
      in1: UInt,
      payload1: T,
      in2: UInt,
      payload2: T,
      out1: UInt,
      payloadOut1: T,
      out2: UInt,
      payloadOut2: T
  ) = {
    val exchg = if (asc) in2 < in1 else in1 < in2
    out1 := Mux(exchg, in2, in1)
    payloadOut1 := Mux(exchg, payload2, payload1)
    out2 := Mux(exchg, in1, in2)
    payloadOut2 := Mux(exchg, payload1, payload2)
  }
}

class BitonicStage[T <: Data](
    n: Int,
    w: Int,
    payload: T,
    stage: Int,
    pipelineFrac: Int = 1
) extends Module {

  val io = IO(new Bundle {
    val tagIn = Input(UInt(8.W))
    val numberIn = Input(Vec(n, UInt(w.W)))
    val payloadIn = Input(Vec(n, payload))
    val tagOut = Output(UInt(8.W))
    val numberOut = Output(Vec(n, UInt(w.W)))
    val payloadOut = Output(Vec(n, payload))
  })

  val numbers_w_in = Seq.fill(stage)(Wire(Vec(n, UInt(w.W))))
  val numbers_w_out = Seq.fill(stage)(Wire(Vec(n, UInt(w.W))))

  val payloads_w_in = Seq.fill(stage)(Wire(Vec(n, payload)))
  val payloads_w_out = Seq.fill(stage)(Wire(Vec(n, payload)))

  var stageBias = 0
  for(i <- 0 until stage){
    stageBias += i
  }
  var delay = 0
  val numbers_r = Seq.tabulate(stage)(idx => {
    if(pipelineFrac == 0){
      Wire(UInt(numbers_w_in(0).getWidth.W))
    } else if ((stageBias + idx) % pipelineFrac == 0) {
      delay += 1
      RegInit(0.U(numbers_w_in(0).getWidth.W))
    } else Wire(UInt(numbers_w_in(0).getWidth.W))
  })
  val payloads_r = Seq.tabulate(stage)(idx => {
    if(pipelineFrac == 0) Wire(UInt(payloads_w_in(0).getWidth.W))
    else if ((stageBias + idx) % pipelineFrac == 0) RegInit(0.U(payloads_w_in(0).getWidth.W))
    else Wire(UInt(payloads_w_in(0).getWidth.W))
  })

  io.tagOut := ShiftRegister(io.tagIn, delay)

  for (i <- 0 until stage) {
    numbers_r(i) := numbers_w_in(i).asTypeOf(numbers_r(i))
    payloads_r(i) := payloads_w_in(i).asTypeOf(payloads_r(i))
    numbers_w_out(i) := numbers_r(i).asTypeOf(numbers_w_out(i))
    payloads_w_out(i) := payloads_r(i).asTypeOf(payloads_w_out(i))
  }

  val ascRef = Array.fill(n)(false)
  var ref = true
  val gap = pow(2, stage).intValue()
  var ptr = 0
  while (ptr < n) {
    for (i <- ptr until ptr + gap) {
      ascRef(i) = ref
    }
    ptr += gap
    ref = !ref
  }

  for (i <- 0 until n) {
    numbers_w_in(0)(i) := io.numberIn(i)
    payloads_w_in(0)(i) := io.payloadIn(i)
  }

  for (
    i <- 1 until stage;
    gap = pow(2, stage - i).intValue;
    group = n / 2 / gap
  ) {
    for (groupNr <- 0 until group; groupStartIdx = groupNr * gap * 2) {
      val asc = ascRef(groupStartIdx)
      for (j <- groupStartIdx until groupStartIdx + gap) {
        BitonicSort.switch(
          asc,
          numbers_w_out(i - 1)(j),
          payloads_w_out(i - 1)(j),
          numbers_w_out(i - 1)(j + gap),
          payloads_w_out(i - 1)(j + gap),
          numbers_w_in(i)(j),
          payloads_w_in(i)(j),
          numbers_w_in(i)(j + gap),
          payloads_w_in(i)(j + gap)
        )
      }
    }
  }

  for (top <- 0 until n by 2; bottom = top + 1) {
    BitonicSort.switch(
      ascRef(top),
      numbers_w_out(stage - 1)(top),
      payloads_w_out(stage - 1)(top),
      numbers_w_out(stage - 1)(bottom),
      payloads_w_out(stage - 1)(bottom),
      io.numberOut(top),
      io.payloadOut(top),
      io.numberOut(bottom),
      io.payloadOut(bottom)
    )
  }

}

class Bitonic[T <: Data](n: Int, w: Int, payload: T, pipelineFrac:Int = 1, topk:Int = 1) extends Module {
  
  val stage = log2Ceil(n)
  val finalStages = log2Ceil(topk)
  var originN = n
  var groupOpt = n / 2 / pow(2, finalStages).intValue
  val remainStage = log2Ceil(groupOpt)
  
  var newN = pow(2,finalStages).intValue * groupOpt
  val newStage = log2Ceil(newN)

  if (finalStages >= stage-1){
    newN = n
  }

  val finalSize = pow(2,log2Ceil(topk)+1).intValue

  
  val io = IO(new Bundle {
    val tagIn = Input(UInt(8.W))
    val numberIn = Input(Vec(n, UInt(w.W)))
    val payloadIn = Input(Vec(n, payload))
    val tagOut = Output(UInt(8.W))
    val numberOut = Output(Vec(finalSize, UInt(w.W)))
    val payloadOut = Output(Vec(finalSize, payload))
  })

  for (i <- 0 until n){
    printf(p"numberin=${io.numberIn(i)}\n")
  }

  val bitonicStages = if (finalStages < stage-1){
    Seq.tabulate(stage)((idx: Int) => {
        if (idx > finalStages){
          val module = Module(new BitonicStage(newN, w, payload, newStage-remainStage+1, pipelineFrac))
          newN = (newN)/2
          module
        } 
        else Module(new BitonicStage(n, w, payload, idx + 1, pipelineFrac))
      })
  }else{
    Seq.tabulate(stage)((idx: Int) => Module(new BitonicStage(n, w, payload, idx + 1, pipelineFrac)))
  }


  bitonicStages(0).io.tagIn := io.tagIn
  bitonicStages(0).io.numberIn := io.numberIn
  bitonicStages(0).io.payloadIn := io.payloadIn

  if (finalStages < stage-1){
    for (i <- 1 until finalStages+1) {
        bitonicStages(i).io.tagIn := bitonicStages(i - 1).io.tagOut
        bitonicStages(i).io.numberIn := bitonicStages(i - 1).io.numberOut
        bitonicStages(i).io.payloadIn := bitonicStages(i - 1).io.payloadOut
      }

      for ( idx <- finalStages+1 until stage){
        var ref = true
        val gap = originN / groupOpt
        for (i <- 0 until groupOpt){
          if (ref){
            for ( j <- 0 until gap/2){
              bitonicStages(idx).io.numberIn(i * (gap/2) + j) := bitonicStages(idx - 1).io.numberOut(i * gap + j + (gap/2))
              bitonicStages(idx).io.payloadIn(i * (gap/2) + j) := bitonicStages(idx - 1).io.payloadOut(i * gap + j + (gap/2))
            }
          }else{
            for ( j <- 0 until gap/2){
              bitonicStages(idx).io.numberIn(i * (gap/2) + j) := bitonicStages(idx - 1).io.numberOut(i * gap + j)
              bitonicStages(idx).io.payloadIn(i * (gap/2) + j) := bitonicStages(idx - 1).io.payloadOut(i * gap + j)
            }
          }
          ref = !ref
        }

        bitonicStages(idx).io.tagIn := bitonicStages(idx - 1).io.tagOut

        originN = originN / 2
        groupOpt = groupOpt / 2
      }
  }else{
    for (i <- 1 until stage) {
      bitonicStages(i).io.tagIn := bitonicStages(i - 1).io.tagOut
      bitonicStages(i).io.numberIn := bitonicStages(i - 1).io.numberOut
      bitonicStages(i).io.payloadIn := bitonicStages(i - 1).io.payloadOut
    }
  }
  

  io.tagOut := bitonicStages(stage - 1).io.tagOut
  io.numberOut := bitonicStages(stage - 1).io.numberOut
  io.payloadOut := bitonicStages(stage - 1).io.payloadOut

}
object Bitonic extends App {
  (new ChiselStage).emitVerilog(new Bitonic(64, 8, UInt(6.W), 1, 4), args)
}