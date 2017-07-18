import org.apache.spark.sql.hive.HiveContext
import java.math.BigDecimal
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.commons.math3.stat.descriptive._
import org.apache.commons.math3.distribution._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import java.util.Calendar
 
val accntlvl2016 = sqlContext.sql("select loannum,acc_cb_raw_scr_no,acc_pdue_pplint_qt from ale_homeloans_dev1.accountlevel where asofdate='2016-03-31' or asofdate='2016-04-30' or asofdate='2016-05-31' or asofdate='2016-06-30' or asofdate='2016-09-30' or asofdate='2016-10-31'")
accntlvl2016.count
res0: Long = 147991710
val selectfields= accntlvl2016.select("acc_cb_raw_scr_no","acc_pdue_pplint_qt")
val newlayout = selectfields.withColumn("customer_type",when($"acc_cb_raw_scr_no" > 600 && $"acc_pdue_pplint_qt" <= 0,"good").when($"acc_cb_raw_scr_no" < 300 && $"acc_pdue_pplint_qt" > 0,"bad").when($"acc_cb_raw_scr_no" > 600 && $"acc_pdue_pplint_qt" > 0,"ClassC").when($"acc_cb_raw_scr_no" < 300 && $"acc_pdue_pplint_qt" <= 0,"ClassD").when($"acc_cb_raw_scr_no" <= 600 && $"acc_cb_raw_scr_no" >= 300 && $"acc_pdue_pplint_qt" <= 0,"ClassE").when($"acc_cb_raw_scr_no" <= 600 && $"acc_cb_raw_scr_no" >= 300 && $"acc_pdue_pplint_qt" > 0,"ClassF").otherwise("NA"))
val fields = Array(StructField("FICOScore", DoubleType),
                   StructField("Delinquency", DoubleType),
                   StructField("CustomerType", StringType))
val schema = StructType(fields)
val format2016 = newlayout.rdd.map{row =>
  val temp1 = row.apply(0).asInstanceOf[Long]
  val temp2 = row.apply(1).asInstanceOf[Long]
  val temp3 = row.apply(2).asInstanceOf[String]
  (temp1.toDouble,temp2.toDouble,temp3)
}
val data2016 = format2016.map(values => Row(values._1.toDouble,values._2.toDouble,values._3.toString))
val df2016 = sqlContext.createDataFrame(data2016,schema)
val Custindexer = new StringIndexer().setInputCol("CustomerType").setOutputCol("CustomerTypeIndex").fit(df2016)
val Custindexed = Custindexer.transform(df2016)
val Custcoder = new OneHotEncoder().setInputCol("CustomerTypeIndex").setOutputCol("CustomerTypeVec")
val Custcoded = Custcoder.transform(Custindexed)
val InputCols = Custcoded.select($"CustomerTypeIndex",$"FICOScore",$"Delinquency")
InputCols.select($"CustomerTypeIndex").distinct.show
+-----------------+
|CustomerTypeIndex|
+-----------------+
|              1.0|
|              3.0|
|              5.0|
|              0.0|
|              4.0|
|              2.0|
+-----------------+
val Array(trainingKM,testKM) = InputCols.randomSplit(Array(0.7, 0.3), seed = 11L)
val vectors = trainingKM.rdd.map(r => Vectors.dense( r.getDouble(1), r.getDouble(2)))
val startKM=System.nanoTime()
val kMeansModel = KMeans.train(vectors, 6, 20)
val endKM = System.nanoTime()
println("Time elapsed: " + (endKM-startKM)/1000 + " microsecs")
Time elapsed: 141851391 microsecs
kMeansModel.clusterCenters.foreach(println)
[644.1809922731394,34.65057376765554]
[0.2641854604095591,17894.347077819464]
[603.2568761104418,1147.6236280770433]
[726.4487093396643,-9999.0]
[592.4747051418602,2448.060484979251]
[510.2054323900801,4640.332476401345]
//Model on training data
val predictionsKMTrain = trainingKM.rdd.map{r => (r.getDouble(0), kMeansModel.predict(Vectors.dense(r.getDouble(1), r.getDouble(2)) ))}
 
val predictionAndLabelsKMTrain = predictionsKMTrain.map { x => (x._1.toDouble,x._2.toDouble) }
val metricsKMTrain = new MulticlassMetrics(predictionAndLabelsKMTrain)
println("Confusion matrix:")
println(metricsKMTrain.confusionMatrix)
5.7423811E7  1.689372E7  7811756.0  1380528.0  1339491.0  2658282.0
0.0          169.0       0.0        493409.0   26.0       0.0
0.0          5232872.0   0.0        789471.0   714073.0   0.0
305614.0     0.0         3285.0     0.0        0.0        5320.0
0.0          2926373.0   0.0        456929.0   550461.0   0.0
0.0          3031040.0   0.0        1222475.0  364212.0   0.0
val labelsKMTrain = metricsKMTrain.labels
labelsKMTrain.foreach { l =>
  println(s"Precision($l) = " + metricsKMTrain.precision(l))
}
Precision(0.0) = 0.9947060965876587
Precision(1.0) = 6.017624018424042E-6
Precision(2.0) = 0.0
Precision(3.0) = 0.0
Precision(4.0) = 0.18544886352725484
Precision(5.0) = 0.0
// Recall by label
labelsKMTrain.foreach { l =>
  println(s"Recall($l) = " + metricsKMTrain.recall(l))
}
Recall(0.0) = 0.6562152187305174
Recall(1.0) = 3.423797213960989E-4
Recall(2.0) = 0.0
Recall(3.0) = 0.0
Recall(4.0) = 0.1399324260256655
Recall(5.0) = 0.0
// False positive rate by label
labelsKMTrain.foreach { l =>
  println(s"FPR($l) = " + metricsKMTrain.falsePositiveRate(l))
}
FPR(0.0) = 0.01898727295918066
FPR(1.0) = 0.27237012094098256
FPR(2.0) = 0.08067813586810214
FPR(3.0) = 0.04204521178024035
FPR(4.0) = 0.024258180186097753
FPR(5.0) = 0.02690898745968984
// F-measure by label
labelsKMTrain.foreach { l =>
  println(s"F1-Score($l) = " + metricsKMTrain.fMeasure(l))
}
F1-Score(0.0) = 0.7907600110173016
F1-Score(1.0) = 1.1827371603208618E-5
F1-Score(2.0) = 0.0
F1-Score(3.0) = 0.0
F1-Score(4.0) = 0.15950707806664305
F1-Score(5.0) = 0.0
// Weighted stats
println(s"Weighted precision: ${metricsKMTrain.weightedPrecision}")
Weighted precision: 0.8472107715367829
println(s"Weighted recall: ${metricsKMTrain.weightedRecall}")
Weighted recall: 0.5595809350389814
println(s"Weighted F1 score: ${metricsKMTrain.weightedFMeasure}")
Weighted F1 score: 0.673964619597576
println(s"Weighted false positive rate: ${metricsKMTrain.weightedFalsePositiveRate}")
Weighted false positive rate: 0.024828844233274725
//Test data prediction
val predictionsKMTest = testKM.rdd.map{r => (r.getDouble(0), kMeansModel.predict(Vectors.dense(r.getDouble(1), r.getDouble(2)) ))}
//Print the center of each cluster
 
val predictionAndLabelsKMTest = predictionsKMTest.map { x => (x._1.toDouble,x._2.toDouble) }
val metricsKMTest = new MulticlassMetrics(predictionAndLabelsKMTest)
println("Confusion matrix:")
println(metricsKMTest.confusionMatrix)
2.4601146E7  7236610.0  3351421.0  590876.0  572837.0  1138945.0
0.0          59.0       0.0        211237.0  16.0      0.0
0.0          2241552.0  0.0        337760.0  305245.0  0.0
131317.0     0.0        1393.0     0.0       0.0       2235.0
0.0          1253967.0  0.0        196728.0  236072.0  0.0
0.0          1299815.0  0.0        523268.0  155894.0  0.0
val labelsKMTest = metricsKMTest.labels
labelsKMTest.foreach { l =>
  println(s"Precision($l) = " + metricsKMTest.precision(l))
}
Precision(0.0) = 0.9946905004972615
Precision(1.0) = 4.90358920289498E-6
Precision(2.0) = 0.0
Precision(3.0) = 0.0
Precision(4.0) = 0.18587409768326635
Precision(5.0) = 0.0
 
// Recall by label
labelsKMTest.foreach { l =>
  println(s"Recall($l) = " + metricsKMTest.recall(l))
}
Recall(0.0) = 0.656173430828339
Recall(1.0) = 2.792079957598243E-4
Recall(2.0) = 0.0
Recall(3.0) = 0.0
Recall(4.0) = 0.13995531095877498
Recall(5.0) = 0.0
// False positive rate by label
labelsKMTest.foreach { l =>
  println(s"FPR($l) = " + metricsKMTest.falsePositiveRate(l))
}
FPR(0.0) = 0.019040947672737618
FPR(1.0) = 0.27235715279603917
FPR(2.0) = 0.08078323169935425
FPR(3.0) = 0.04202766301961375
FPR(4.0) = 0.024214347247573195
FPR(5.0) = 0.026908646891058343
// F-measure by label
labelsKMTest.foreach { l =>
  println(s"F1-Score($l) = " + metricsKMTest.fMeasure(l))
}
F1-Score(0.0) = 0.7907247422863655
F1-Score(1.0) = 9.637912607819045E-6
F1-Score(2.0) = 0.0
F1-Score(3.0) = 0.0
F1-Score(4.0) = 0.1596790618063731
F1-Score(5.0) = 0.0
// Weighted stats
println(s"Weighted precision: ${metricsKMTest.weightedPrecision}")
Weighted precision: 0.8472102031498392
println(s"Weighted recall: ${metricsKMTest.weightedRecall}")
Weighted recall: 0.5595444061243668
println(s"Weighted F1 score: ${metricsKMTest.weightedFMeasure}")
Weighted F1 score: 0.6739389050842745
println(s"Weighted false positive rate: ${metricsKMTest.weightedFalsePositiveRate}")
Weighted false positive rate: 0.024876395450743143
// Save and load model
val now = Calendar.getInstance()
val modelName = "mlinput/KMeansModel_"+now.get(Calendar.YEAR).toString+(now.get(Calendar.MONTH)+1).toString+now.get(Calendar.DAY_OF_MONTH).toString+now.get(Calendar.HOUR).toString+now.get(Calendar.MINUTE).toString+now.get(Calendar.SECOND).toString
modelName: String = mlinput/KMeansModel_201711292322
kMeansModel.save(sc, modelName)
val sameModel = KMeansModel.load(sc, modelName)
sameModel: org.apache.spark.mllib.clustering.KMeansModel = org.apache.spark.mllib.clustering.KMeansModel@a8728c4
