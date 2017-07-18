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
import org.apache.spark.util.SizeEstimator
import java.util.Calendar
val sqlContext = new HiveContext(sc)
 
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
df2016.select($"CustomerType").distinct.show
+------------+
|CustomerType|
+------------+
|      ClassC|
|      ClassD|
|      ClassE|
|      ClassF|
|         bad|
|        good|
+------------+
val MLPindexer = new StringIndexer().setInputCol("CustomerType").setOutputCol("CustomerTypeIndex").fit(df2016)
val MLPindexed = MLPindexer.transform(df2016)
val MLPcoder = new OneHotEncoder().setInputCol("CustomerTypeIndex").setOutputCol("CustomerTypeVec")
val MLPcoded = MLPcoder.transform(MLPindexed)
MLPcoded.select("CustomerType", "CustomerTypeIndex").distinct.show
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
MLPcoded.select("CustomerType", "CustomerTypeIndex","CustomerTypeVec").distinct.show
+------------+-----------------+---------------+
|CustomerType|CustomerTypeIndex|CustomerTypeVec|
+------------+-----------------+---------------+
|        good|              0.0|  (5,[0],[1.0])|
|      ClassF|              4.0|  (5,[4],[1.0])|
|         bad|              3.0|  (5,[3],[1.0])|
|      ClassD|              2.0|  (5,[2],[1.0])|
|      ClassC|              1.0|  (5,[1],[1.0])|
|      ClassE|              5.0|      (5,[],[])|
+------------+-----------------+---------------+
MLPcoded.printSchema
root
|-- FICOScore: double (nullable = true)
|-- Delinquency: double (nullable = true)
|-- CustomerType: string (nullable = true)
|-- CustomerTypeIndex: double (nullable = true)
|-- CustomerTypeVec: vector (nullable = true)
 
val assembler = new VectorAssembler().setInputCols(Array("FICOScore","Delinquency")).setOutputCol("features")
val rdd = assembler.transform(MLPcoded)
val splits = rdd.randomSplit(Array(0.7,0.2))
val (trainingMLP, testMLP) = (splits(0), splits(1))
 
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](2, 6, 6)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().
  setLayers(layers).
  setBlockSize(128).
  setSeed(1234L).
  setMaxIter(100).
  setLabelCol("CustomerTypeIndex")
val startMLP=System.nanoTime()
// train the model
val model = trainer.fit(trainingMLP)
// compute accuracy on the test set
val endMLP = System.nanoTime()
println("Time elapsed: " + (endMLP-startMLP)/1000 + " microsecs")
Time elapsed: 240398466 microsecs
Time elapsed: 223675103 microsecs
 
//training data prediction
val resultMLPTrain = model.transform(trainingMLP)
resultMLPTrain.select($"prediction").distinct.show
+----------+
|prediction|
+----------+
|       1.0|
|       3.0|
|       5.0|
|       0.0|
|       2.0|
+----------+
 
val predMLPTrain = resultMLPTrain.select($"CustomerTypeIndex",$"prediction").toDF("Act", "Cluster")
predMLPTrain.where($"Act" === $"Cluster").count
res8: Long = 17026539
 
val accuracyMLPTrain = (predMLPTrain.where($"Act" === $"Cluster").count.toDouble)/(resultMLPTrain.count.toDouble)
accuracyMLPTrain: Double = 0.9006595882191573
predMLPTrain.where($"Act" !== $"Cluster").count
res9: Long = 1877983
 
val predictionAndLabelsMLPTrain = resultMLPTrain.select("prediction", "CustomerTypeIndex")
val predictionAndLabelsMLPRDDTrain = predictionAndLabelsMLPTrain.rdd.map { x => (x.apply(0).asInstanceOf[Double],x.apply(1).asInstanceOf[Double]) }
val metricsMLPTrain = new MulticlassMetrics(predictionAndLabelsMLPRDDTrain)
println("Confusion matrix:")
println(metricsMLPTrain.confusionMatrix)
1.059826E7  0.0        56751.0    0.0    0.0  0.0
0.0         4978270.0  0.0        0.0    0.0  0.0
42.0        418.0      1449152.0  0.0    0.0  431.0
0.0         781692.0   4974.0     857.0  0.0  0.0
0.0         539404.0   0.0        0.0    0.0  0.0
493295.0    0.0        976.0      0.0    0.0  0.0
 
val labelsMLPTrain = metricsMLPTrain.labels
labelsMLPTrain.foreach { l =>
  println(s"Precision($l) = " + metricsMLPTrain.precision(l))
}
Precision(0.0) = 0.9555215538393614
Precision(1.0) = 0.7902286808563596
Precision(2.0) = 0.9585270525639729
Precision(3.0) = 1.0
Precision(4.0) = 0.0
Precision(5.0) = 0.0
 
// Recall by label
labelsMLPTrain.foreach { l =>
  println(s"Recall($l) = " + metricsMLPTrain.recall(l))
}
Recall(0.0) = 0.9946737736826363
Recall(1.0) = 1.0
Recall(2.0) = 0.9993855354634311
Recall(3.0) = 0.0010882221852568116
Recall(4.0) = 0.0
Recall(5.0) = 0.0
 
// False positive rate by label
labelsMLPTrain.foreach { l =>
  println(s"FPR($l) = " + metricsMLPTrain.falsePositiveRate(l))
}
FPR(0.0) = 0.05980196886821534
FPR(1.0) = 0.094893730201062
FPR(2.0) = 0.0035922584684423982
FPR(3.0) = 0.0
FPR(4.0) = 0.0
FPR(5.0) = 2.3410870389545476E-5
 
// F-measure by label
labelsMLPTrain.foreach { l =>
  println(s"F1-Score($l) = " + metricsMLPTrain.fMeasure(l))
}
F1-Score(0.0) = 0.9747046527899891
F1-Score(1.0) = 0.8828242886583093
F1-Score(2.0) = 0.9785299686417079
F1-Score(3.0) = 0.0021740784900682415
F1-Score(4.0) = 0.0
F1-Score(5.0) = 0.0
 
// Weighted stats
println(s"Weighted precision: ${metricsMLPTrain.weightedPrecision}")
Weighted precision: 0.8618304575393813
println(s"Weighted recall: ${metricsMLPTrain.weightedRecall}")
Weighted recall: 0.9006595882191574
println(s"Weighted F1 score: ${metricsMLPTrain.weightedFMeasure}")
Weighted F1 score: 0.8569933234419458
println(s"Weighted false positive rate: ${metricsMLPTrain.weightedFalsePositiveRate}")
Weighted false positive rate: 0.05897095662726658
 
val accuracyMLPTrain = (resultMLPTrain.filter($"prediction" === $"CustomerTypeIndex").count.toDouble)/(resultMLPTrain.count.toDouble) * 100
accuracyMLPTrain: Double = 90.06595882191573
 
//test data prediction
val resultMLP = model.transform(testMLP)
resultMLP.select($"prediction").distinct.show
+----------+
|prediction|
+----------+
|       1.0|
|       3.0|
|       5.0|
|       0.0|
|       2.0|
+----------+
val predMLP = resultMLP.select($"CustomerTypeIndex",$"prediction").toDF("Act", "Cluster")
predMLP.where($"Act" === $"Cluster").count
res21: Long = 4867439
 
val accuracyMLP = (predMLP.where($"Act" === $"Cluster").count.toDouble)/(resultMLP.count.toDouble)
accuracyMLP: Double = 0.9008794396283832
accuracyMLP: Double = 0.9014125606940132
predMLP.where($"Act" !== $"Cluster").count
res22: Long = 535547
 
val predictionAndLabelsMLP = resultMLP.select("prediction", "CustomerTypeIndex")
val predictionAndLabelsMLPRDD = predictionAndLabelsMLP.rdd.map { x => (x.apply(0).asInstanceOf[Double],x.apply(1).asInstanceOf[Double]) }
val metricsMLP = new MulticlassMetrics(predictionAndLabelsMLPRDD)
println("Confusion matrix:")
println(metricsMLP.confusionMatrix)
3029115.0  0.0        16059.0   0.0    0.0  0.0
0.0        1423705.0  0.0       0.0    0.0  0.0
10.0       109.0      414359.0  0.0    0.0  123.0
0.0        223049.0   1358.0    260.0  0.0  0.0
0.0        153887.0   0.0       0.0    0.0  0.0
140671.0   0.0        281.0     0.0    0.0  0.0
 
val labelsMLP = metricsMLP.labels
labelsMLP.foreach { l =>
  println(s"Precision($l) = " + metricsMLP.precision(l))
}
Precision(0.0) = 0.9556182795359701
Precision(1.0) = 0.790617798139664
Precision(2.0) = 0.9590378121405277
Precision(3.0) = 1.0
Precision(4.0) = 0.0
Precision(5.0) = 0.0
 
// Recall by label
labelsMLP.foreach { l =>
  println(s"Recall($l) = " + metricsMLP.recall(l))
}
Recall(0.0) = 0.9947264097224001
Recall(1.0) = 1.0
Recall(2.0) = 0.9994163062800138
Recall(3.0) = 0.0011572683126582898
Recall(4.0) = 0.0
Recall(5.0) = 0.0
 
// False positive rate by label
labelsMLP.foreach { l =>
 println(s"FPR($l) = " + metricsMLP.falsePositiveRate(l))
}
FPR(0.0) = 0.05966591059847011
FPR(1.0) = 0.09475204188897442
FPR(2.0) = 0.0035478416361207086
FPR(3.0) = 0.0
FPR(4.0) = 0.0
FPR(5.0) = 2.3374991495683987E-5
 
// F-measure by label
labelsMLP.foreach { l =>
  println(s"F1-Score($l) = " + metricsMLP.fMeasure(l))
}
F1-Score(0.0) = 0.9747802483358728
F1-Score(1.0) = 0.883067060945183
F1-Score(2.0) = 0.9788108067247933
F1-Score(3.0) = 0.0023118611816278166
F1-Score(4.0) = 0.0
F1-Score(5.0) = 0.0
 
// Weighted stats
println(s"Weighted precision: ${metricsMLP.weightedPrecision}")
Weighted precision: 0.8621002325416675
println(s"Weighted recall: ${metricsMLP.weightedRecall}")
Weighted recall: 0.9008794396283832
println(s"Weighted F1 score: ${metricsMLP.weightedFMeasure}")
Weighted F1 score: 0.8572922077058692
println(s"Weighted false positive rate: ${metricsMLP.weightedFalsePositiveRate}")
Weighted false positive rate: 0.058868608743444184
 
val accuracyMLP = (resultMLP.filter($"prediction" === $"CustomerTypeIndex").count.toDouble)/(resultMLP.count.toDouble) * 100
accuracyMLP: Double = 90.08794396283832
