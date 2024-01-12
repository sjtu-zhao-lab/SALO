package sa

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

abstract class Arithmetic[T <: Data] {
    implicit def cast(t: T) : ArithmeticOps[T]
}

abstract class ArithmeticOps[T <: Data](self: T) {
    def *(t: T): T
    def +(t: T): T
    def mac(m1: T, m2: T): T // Returns (m1 * m2 + self)
    def >>(u: UInt): T // This is a rounding shift! Rounds away from 0
    def >(t: T): Bool
    def identity: T
    def withWidthOf(t: T): T
    def clippedToWidthOf(t: T): T // Like "withWidthOf", except that it saturates
    def zero: T
}

object Arithmetic {
    implicit object FixedPointArithmetic extends Arithmetic[FixedPoint] {
        override implicit def cast(self: FixedPoint) = new ArithmeticOps(self) {
            override def *(t: FixedPoint) = self * t
            override def +(t: FixedPoint) = self + t
            override def mac(m1: FixedPoint, m2: FixedPoint) = m1 * m2 + self
            
            override def >>(u: UInt) = {
                self >> u
            }

            override def >(u: FixedPoint) = self > u
            override def identity = 1.F(self.getWidth.W, self.binaryPoint)

            override def withWidthOf(t: FixedPoint): FixedPoint = {
                val shifted = self.setBinaryPoint(t.binaryPoint.get.toInt)
                // In https://github.com/chipsalliance/chisel3/blob/master/src/test/scala/chiselTests/FixedPointSpec.scala, Chisel can automatically extend the sign bits
                shifted(t.getWidth - 1, 0).asFixedPoint(t.binaryPoint)
            }

            override def clippedToWidthOf(t: FixedPoint): FixedPoint = {
                // Calculate the max value and the min value that can be expressed in type t
                val maxsat = ((1 << (t.getWidth - 1)) - 1).F(t.binaryPoint)
                val minsat = (-(1 << (t.getWidth - 1))).F(t.binaryPoint)
                MuxCase(self.withWidthOf(t), Seq((self > maxsat) -> maxsat, (self < minsat) -> minsat))
            }

            override def zero: FixedPoint = 0.F(self.getWidth.W, self.binaryPoint)
        }
    }
}