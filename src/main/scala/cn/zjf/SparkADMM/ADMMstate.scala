https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
package cn.zjf.SparkADMM

//import org.apache.spark.mllib.linalg.DenseVector
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

case class ADMMState(var x:DenseVector[Double],
                     var z:DenseVector[Double],u:DenseVector[Double],var z_pre:DenseVector[Double],var w:DenseVector[Double], var dim : Double)
object ADMMState {
  def apply(initialWeight:Array[Double]):ADMMState={
    new ADMMState(
      x = DenseVector(initialWeight),
      z = zeros(initialWeight.length),
      u = zeros(initialWeight.length),
      z_pre = zeros(initialWeight.length),
      w = zeros(initialWeight.length),
      dim = initialWeight.length
    )
  }
  def zeros(n:Int):DenseVector[Double]={
    DenseVector.zeros(n)
  }
}
