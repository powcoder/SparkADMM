https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
package cn.zjf.SparkADMM

import java.util.Date

import cn.zjf.SparkADMM.MyLoadLibSVMFile.loadLibSVMFile
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object logistic {
  def main(args: Array[String]): Unit = {
    val minPartition = args(0).toInt;
    val conf = new SparkConf()
    conf.setAppName("logistic")
    //conf.setMaster("local[4]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val start_time = new Date().getTime
    val trainingData:RDD[LabeledPoint] = loadLibSVMFile(sc,"hdfs://node0:9000/data/data_0.dat",-1,minPartition)
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = loadLibSVMFile(sc,"hdfs://node0:9000/data/rcv1_train.binary",-1,minPartition)
    val predictionLabel =  predictData.map { lp =>
      val prediction = model.predict(lp.features)
      (lp.label,prediction)
    }
    val metrics = new BinaryClassificationMetrics(predictionLabel)
    println("precision: "+metrics.areaUnderPR())
  }

}
