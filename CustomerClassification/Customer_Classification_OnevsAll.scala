import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
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
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import java.util.Calendar
import org.apache.spark.ml.util.MLWritable
 
val accntlvl2016 = sqlContext.sql("select loannum,acc_cb_raw_scr_no,acc_pdue_pplint_qt from ale_homeloans_dev1.accountlevel where asofdate='2016-03-31' or asofdate='2016-04-30' or asofdate='2016-05-31' or asofdate='2016-06-30' or asofdate='2016-09-30' or asofdate='2016-10-31'")
accntlvl2016.count
res0: Long = 24307508
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
val OVAindexer = new StringIndexer().setInputCol("CustomerType").setOutputCol("CustomerTypeIndex").fit(df2016)
val OVAindexed = OVAindexer.transform(df2016)
val OVAcoder = new OneHotEncoder().setInputCol("CustomerTypeIndex").setOutputCol("CustomerTypeVec")
val OVAcoded = OVAcoder.transform(OVAindexed)
OVAcoded.select("CustomerType", "CustomerTypeIndex").distinct.show
+------------+-----------------+
|CustomerType|CustomerTypeIndex|
+------------+-----------------+
|      ClassE|              5.0|
|        good|              0.0|
|      ClassC|              1.0|
|      ClassD|              2.0|
|      ClassF|              4.0|
|         bad|              3.0|
+------------+-----------------+
val assembler = new VectorAssembler().setInputCols(Array("FICOScore","Delinquency")).setOutputCol("features")
val rdd = assembler.transform(OVAcoded)
val splits = rdd.randomSplit(Array(0.7,0.2))
val (trainingOVA, testOVA) = (splits(0), splits(1))
// instantiate the base classifier
val classifier = new LogisticRegression().
  setMaxIter(10).
  setTol(1E-6).
  setFitIntercept(true).
  setLabelCol("CustomerTypeIndex")
 
// instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setLabelCol("CustomerTypeIndex").setClassifier(classifier)
 
val startMLP=System.nanoTime()
val ovrModel = ovr.fit(trainingOVA)
val endMLP = System.nanoTime()
println("Time elapsed: " + (endMLP-startMLP)/1000 + " microsecs")
Time elapsed: 12988079 microsecs
Time elapsed: 526502972 microsecs
// score the model on training data.
val predictionsOVATrain = ovrModel.transform(trainingOVA)
val predOVATrain = predictionsOVATrain.select("prediction", "CustomerTypeIndex")
predOVATrain.select($"prediction").distinct.show
+----------+
|prediction|
+----------+
|       1.0|
|       3.0|
|       5.0|
|       0.0|
|       2.0|
+----------+
val predTrain = predOVATrain.select($"CustomerTypeIndex",$"prediction").toDF("Act", "Cluster")
predTrain.where($"Act" === $"Cluster").count
res6: Long = 15472764
val accuracyOVATrain = (predTrain.where($"Act" === $"Cluster").count.toDouble)/(predOVATrain.count.toDouble)
accuracyOVATrain: Double = 0.8184539125872811
val accuracyOVA1Train = (predOVATrain.filter($"prediction" === $"CustomerTypeIndex").count.toDouble)/(predOVATrain.count.toDouble) * 100
accuracyOVA1Train: Double = 81.8453912587281
 
val predictionAndLabelsOVATrain = predOVATrain.rdd.map { x => (x.apply(0).asInstanceOf[Double],x.apply(1).asInstanceOf[Double]) }
val metricsOVATrain = new MulticlassMetrics(predictionAndLabelsOVATrain)
println("Confusion matrix:")
println(metricsOVATrain.confusionMatrix)
1.0654018E7  0.0        0.0        0.0       0.0  0.0
2187002.0    2793195.0  0.0        86.0      0.0  0.0
599.0        0.0        1449101.0  0.0       0.0  805.0
0.0          0.0        222302.0   564640.0  0.0  0.0
50433.0      486306.0   0.0        1990.0    0.0  182.0
482345.0     55.0       0.0        0.0       0.0  11810.0
 
val labelsOVATrain = metricsOVATrain.labels
labelsOVATrain.foreach { l =>
  println(s"Precision($l) = " + metricsOVATrain.precision(l))
}
Precision(0.0) = 0.7965980073718464
Precision(1.0) = 0.8516991324435381
Precision(2.0) = 0.8669967685830408
Precision(3.0) = 0.9963367895030315
Precision(4.0) = 0.0
Precision(5.0) = 0.922872548253497
 
// Recall by label
labelsOVATrain.foreach { l =>
  println(s"Recall($l) = " + metricsOVATrain.recall(l))
}
Recall(0.0) = 1.0
Recall(1.0) = 0.5608506584866764
Recall(2.0) = 0.9990320612476344
Recall(3.0) = 0.7175115828104232
Recall(4.0) = 0.0
Recall(5.0) = 0.023896724064668865
 
// False positive rate by label
labelsOVATrain.foreach { l =>
  println(s"FPR($l) = " + metricsOVATrain.falsePositiveRate(l))
}
FPR(0.0) = 0.32970889911840606
FPR(1.0) = 0.03492821976897554
FPR(2.0) = 0.012736184486584558
FPR(3.0) = 1.1458264513373964E-4
FPR(4.0) = 0.0
FPR(5.0) = 5.361024828062917E-5
 
// F-measure by label
labelsOVATrain.foreach { l =>
  println(s"F1-Score($l) = " + metricsOVATrain.fMeasure(l))
}
F1-Score(0.0) = 0.8867849169410467
F1-Score(1.0) = 0.6763315846713235
F1-Score(2.0) = 0.9283431798758964
F1-Score(3.0) = 0.8342432135738864
F1-Score(4.0) = 0.0
F1-Score(5.0) = 0.04658712798837096
 
// Weighted stats
println(s"Weighted precision: ${metricsOVATrain.weightedPrecision}")
Weighted precision: 0.8054225329685216
println(s"Weighted recall: ${metricsOVATrain.weightedRecall}")
Weighted recall: 0.818453912587281
println(s"Weighted F1 score: ${metricsOVATrain.weightedFMeasure}")
Weighted F1 score: 0.7851012579005626
println(s"Weighted false positive rate: ${metricsOVATrain.weightedFalsePositiveRate}")
Weighted false positive rate: 0.1959954088546616
 
// score the model on test data.
val predictionsOVATest = ovrModel.transform(testOVA)
val predOVATest = predictionsOVATest.select("prediction", "CustomerTypeIndex")
predOVATest.select($"prediction").distinct.show
+----------+
|prediction|
+----------+
|       1.0|
|       3.0|
|       5.0|
|       0.0|
|       2.0|
+----------+
val predTest = predOVATest.select($"CustomerTypeIndex",$"prediction").toDF("Act", "Cluster")
predTest.where($"Act" === $"Cluster").count
res18: Long = 4422241
val accuracyOVATest = (predTest.where($"Act" === $"Cluster").count.toDouble)/(predOVATest.count.toDouble)
accuracyOVATest: Double = 0.8185334981663591
val accuracyOVA1Test = (predOVATest.filter($"prediction" === $"CustomerTypeIndex").count.toDouble)/(predOVATest.count.toDouble) * 100
accuracyOVA1Test: Double = 81.85334981663591
val predictionAndLabelsOVATest = predOVATest.rdd.map { x => (x.apply(0).asInstanceOf[Double],x.apply(1).asInstanceOf[Double]) }
val metricsOVATest = new MulticlassMetrics(predictionAndLabelsOVATest)
println("Confusion matrix:")
println(metricsOVATest.confusionMatrix)
 
3046167.0  0.0       0.0       0.0       0.0  0.0
624571.0   797088.0  0.0       33.0      0.0  0.0
184.0      0.0       413683.0  0.0       0.0  272.0
0.0        0.0       63362.0   161886.0  0.0  0.0
14270.0    139497.0  0.0       554.0     0.0  59.0
137580.0   16.0      0.0       0.0       0.0  3417.0
 
val labelsOVATest = metricsOVATest.labels
labelsOVATest.foreach { l =>
  println(s"Precision($l) = " + metricsOVATest.precision(l))
}
Precision(0.0) = 0.7968476801650739
Precision(1.0) = 0.8510432937825179
Precision(2.0) = 0.867178148811957
Precision(3.0) = 0.9963870920091338
Precision(4.0) = 0.0
Precision(5.0) = 0.9116862326574173
 
// Recall by label
labelsOVATest.foreach { l =>
  println(s"Recall($l) = " + metricsOVATest.recall(l))
}
Recall(0.0) = 1.0
Recall(1.0) = 0.5606615216235302
Recall(2.0) = 0.99889892041078
Recall(3.0) = 0.7187011649382015
Recall(4.0) = 0.0
Recall(5.0) = 0.024231808414827003
 
// False positive rate by label
labelsOVATest.foreach { l =>
  println(s"FPR($l) = " + metricsOVATest.falsePositiveRate(l))
}
FPR(0.0) = 0.32956258338736893
FPR(1.0) = 0.03504517894862705
FPR(2.0) = 0.012701613711536533
FPR(3.0) = 1.1337756796811367E-4
FPR(4.0) = 0.0
FPR(5.0) = 6.290831009273559E-5
 
// F-measure by label
labelsOVATest.foreach { l =>
  println(s"F1-Score($l) = " + metricsOVATest.fMeasure(l))
}
F1-Score(0.0) = 0.8869395986774667
F1-Score(1.0) = 0.6759872500999663
F1-Score(2.0) = 0.9283896479290471
F1-Score(3.0) = 0.8350643890839031
F1-Score(4.0) = 0.0
F1-Score(5.0) = 0.047208847686877
 
// Weighted stats
println(s"Weighted precision: ${metricsOVATest.weightedPrecision}")
Weighted precision: 0.8050470615503066
println(s"Weighted recall: ${metricsOVATest.weightedRecall}")
Weighted recall: 0.8185334981663591
println(s"Weighted F1 score: ${metricsOVATest.weightedFMeasure}")
Weighted F1 score: 0.7851806880320141
println(s"Weighted false positive rate: ${metricsOVATest.weightedFalsePositiveRate}")
Weighted false positive rate: 0.19601916009195244
 
// Save and load model
ovrModel.models.zipWithIndex.foreach {
  case (model: MLWritable, i: Int) =>
    model.save(s"model-${model.uid}-$i")
}
model-logreg_012f4813e14d-0
model-logreg_012f4813e14d-1
model-logreg_012f4813e14d-2
model-logreg_012f4813e14d-3
model-logreg_012f4813e14d-4
model-logreg_012f4813e14d-5
