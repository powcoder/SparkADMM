https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
package cn.zjf.SparkADMM

import breeze.linalg.{DenseVector, norm}
import javafx.beans.property.DoublePropertyBase
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.Map
import scala.util.control.Breaks


class ADMMOptimizer(numIterations:Int,updater:ADMMUpdater)
  extends Optimizer with Serializable {
  override def optimize(data: RDD[(Double, linalg.Vector)], initialWeights: linalg.Vector): linalg.Vector = {

    //根据分区将数据划分
    //    val tmp = data.map{case (zeroOnelabel,features) =>{
    //      val scaledLabel = 2 * zeroOnelabel - 1
    //      new LabeledPoint(scaledLabel,features)
    //    }}
    //val dataSize = data.collect().length
    //println("dataSize:"+dataSize)
    var admmData=data.map{ case (zeroOnelabel,features) =>{
      val scaledLabel = 2 * zeroOnelabel - 1
      new LabeledPoint(scaledLabel,features)
    }}.mapPartitionsWithIndex{
      (partIdx,iter)=>{
        var part_map = Map[Int,List[LabeledPoint]]()
        while(iter.hasNext){
          var elem = iter.next()
          if(part_map.contains(partIdx)){
            var elems = part_map(partIdx)
            elems::=elem
            part_map(partIdx) = elems
          }else{
            part_map(partIdx) = List[LabeledPoint]{new LabeledPoint(elem.label,elem.features)}
          }
        }
        part_map.iterator
      }
    }
    admmData.cache()
    val numPartions = data.partitions.length
    println("numPartitions:"+numPartions)
    var admmStates = admmData.map(map=>{
      (map._1,ADMMState(initialWeights.toArray))
    })

    println("x lenght: "+initialWeights.size)
    println("#iternum"+"\t"+"prires"+"\t"+"eps_pres"+"\t"+"dualres"+"\t"+"eps_dual")
    //
    admmData.persist()
    val loop = new Breaks
    loop.breakable{
      for(i<- 0 to numIterations){
        print(i+"\t")
        admmStates = admmData.join(admmStates).map{
          map=>{
            (map._1,updater.xUpdate(map._2._1.toArray,map._2._2))
          }
        }

        admmStates.cache()
        admmStates = updater.zUpdate(admmStates,initialWeights.size)
        admmStates.cache()
        if(updater.isStop(admmStates))
          loop.break()
        admmStates = admmStates.map{
          map=>
            (map._1,updater.uUpdate(map._2))}
        admmStates.cache()
      }
    }

    // return average of final weight vectors across the partitions
    val ret = ADMMUpdater.average(admmStates.map(map=>map._2.x)).data
    linalg.Vectors.dense(ret)
  }
}
