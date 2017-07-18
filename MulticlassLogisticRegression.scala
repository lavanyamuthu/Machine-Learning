val selectfields= accntlvl2016.select("acc_cb_raw_scr_no","acc_pdue_pplint_qt")
val newlayout = selectfields.withColumn("customertype",when($"acc_cb_raw_scr_no" > 600 && $"acc_pdue_pplint_qt" <= 0,0.0).when($"acc_cb_raw_scr_no" < 300 && $"acc_pdue_pplint_qt" > 0,1.0).when($"acc_cb_raw_scr_no" > 600 && $"acc_pdue_pplint_qt" > 0,2.0).when($"acc_cb_raw_scr_no" < 300 && $"acc_pdue_pplint_qt" <= 0,3.0).when($"acc_cb_raw_scr_no" <= 600 && $"acc_cb_raw_scr_no" >= 300 && $"acc_pdue_pplint_qt" <= 0,4.0).when($"acc_cb_raw_scr_no" <= 600 && $"acc_cb_raw_scr_no" >= 300 && $"acc_pdue_pplint_qt" > 0,5.0).otherwise(6.0))
val format2016 = newlayout.rdd.map{row =>
  val temp1 = row.apply(0).asInstanceOf[Long]
  val temp2 = row.apply(1).asInstanceOf[Long]
  val temp3 = row.apply(2).asInstanceOf[Double]
  (temp1.toDouble,temp2.toDouble,temp3)
}
val fields = Array(StructField("FICOScore", DoubleType),
                   StructField("Delinquency", DoubleType),
                   StructField("label", DoubleType))
val schema = StructType(fields)
 
val data2016 = format2016.map(values => Row(values._1.toDouble,values._2.toDouble,values._3.toDouble))
val df2016 = sqlContext.createDataFrame(data2016,schema)
val assembler = new VectorAssembler().setInputCols(Array("FICOScore","Delinquency")).setOutputCol("features")
val dfrdd = assembler.transform(df2016)
dfrdd.select($"label").distinct.show
+-----+
|label|
+-----+
|  1.0|
|  3.0|
|  5.0|
|  0.0|
|  4.0|
|  2.0|
+-----+
dfrdd.filter($"label" === 0.0).count
res31: Long = 82461888
 
val labeledPointsLR = dfrdd.rdd.map(parts => LabeledPoint(parts.getAs[Double](2), Vectors.dense(parts.getAs[Double](0),parts.getAs[Double](1))))
val Array(trainingLR,testLR) = labeledPointsLR.randomSplit(Array(0.6, 0.4), seed = 11L)
val a = trainingLR.toDF(“label”, “features”)
a.select($"label").distinct.show
 
//LBGFS
val startLR=System.nanoTime()
val LRmodel = new LogisticRegressionWithLBFGS().setNumClasses(6).run(trainingLR)
val endLR = System.nanoTime()
println("Time elapsed: " + (endLR-startLR)/1000 + " microsecs")
Time elapsed: 129784739 microsecs
Time elapsed: 59571750 microsecs
2.5 G Time elapsed: 115470087 microsecs
//training data prediction
val labelAndPredsLRTrain = trainingLR.map { point =>
  val prediction = LRmodel.predict(point.features)
  (point.label, prediction)
}
 
val predictionAndLabelsLRTrain = labelAndPredsLRTrain.map { x => (x._1.toDouble,x._2.toDouble) }
val metricsLRTrain = new MulticlassMetrics(predictionAndLabelsLRTrain)
println("Confusion matrix:")
println(metricsLRTrain.confusionMatrix)
4.9482365E7  207.0      1771391.0    6698437.0
0.0          3717434.0  140.0        0.0
0.0          3899.0     2.2300625E7  0.0
0.0          0.0        0.0          2832.0
val accuracyLRTrain = labelAndPredsLRTrain.filter(r => r._1 == r._2).count.toDouble / trainingLR.count.toDouble
accuracyLRTrain: Double = 0.8502475104531453
 
val labelsLRTrain = metricsLRTrain.labels
labelsLRTrain.foreach { l =>
  println(s"Precision($l) = " + metricsLRTrain.precision(l))
}
Precision(0.0) = 1.0
Precision(1.0) = 0.9988966933043848
Precision(2.0) = 0.9264074642919397
Precision(3.0) = 4.226065242269785E-4
 
// Recall by label
labelsLRTrain.foreach { l =>
  println(s"Recall($l) = " + metricsLRTrain.recall(l))
}
Recall(0.0) = 0.8206599017537267
Recall(1.0) = 0.9999467402905936
Recall(2.0) = 0.8997551927532278
Recall(3.0) = 1.0
 
// False positive rate by label
labelsLRTrain.foreach { l =>
  println(s"FPR($l) = " + metricsLRTrain.falsePositiveRate(l))
}
FPR(0.0) = 0.0
FPR(1.0) = 4.825826563836365E-5
FPR(2.0) = 0.0276731290636533
FPR(3.0) = 0.07543397890981926
 
// F-measure by label
labelsLRTrain.foreach { l =>
  println(s"F1-Score($l) = " + metricsLRTrain.fMeasure(l))
}
F1-Score(0.0) = 0.9014971999583632
F1-Score(1.0) = 0.999421440988325
F1-Score(2.0) = 0.9128868377483275
F1-Score(3.0) = 8.448560067934538E-4
 
// Weighted stats
println(s"Weighted precision: ${metricsLRTrain.weightedPrecision}")
Weighted precision: 0.9793816636369204
println(s"Weighted recall: ${metricsLRTrain.weightedRecall}")
Weighted recall: 0.8502475104531454
println(s"Weighted F1 score: ${metricsLRTrain.weightedFMeasure}")
Weighted F1 score: 0.9087469675872641
println(s"Weighted false positive rate: ${metricsLRTrain.weightedFalsePositiveRate}")
Weighted false positive rate: 0.007728219072163228
 
//test data prediction
val labelAndPredsLRTest = testLR.map { point =>
  val prediction = LRmodel.predict(point.features)
  (point.label, prediction)
}
val trainErr = labelAndPredsLRTest.filter(r => r._1 != r._2).count.toDouble / testLR.count
trainErr: Double = 0.14981991615910523
val trainAcc = labelAndPredsLRTest.filter(r => r._1 == r._2).count.toDouble / testLR.count
trainAcc: Double = 0.8501800838408947
 
val evaluationMetricsLRTest = new MulticlassMetrics(labelAndPredsLRTest.map(x =>
  (x._1, x._2)))
evaluationMetricsLRTest.precision
res19: Double = 0.8501800838408947
evaluationMetricsLRTest.recall
res20: Double = 0.8501800838408947
evaluationMetricsLRTest.fMeasure
res21: Double = 0.8501800838408947
evaluationMetricsLRTest.weightedFMeasure
res22: Double = 0.9087038438294285
evaluationMetricsLRTest.weightedPrecision
res23: Double = 0.9793687724257282
evaluationMetricsLRTest.weightedFalsePositiveRate
res24: Double = 0.007732885300484439
evaluationMetricsLRTest.weightedTruePositiveRate
res25: Double = 0.850180083840894
 
val predLRTest = labelAndPredsLRTest.toDF("Act", "Cluster")
predLRTest.select($"Cluster").distinct.show
+-------+
|Cluster|
+-------+
|    1.0|
|    3.0|
|    0.0|
|    2.0|
+-------+
predLRTest.where($"Act" === $"Cluster").count
res29: Long = 50322336
val accuracyLRTest = (predLRTest.where($"Act" === $"Cluster").count.toDouble)/(predLRTest.count.toDouble)
accuracyLRTest: Double = 0.8501800838408947
 
val predictionAndLabelsLRTest = labelAndPredsLRTest.map { x => (x._1.toDouble,x._2.toDouble) }
val metricsLRTest = new MulticlassMetrics(predictionAndLabelsLRTest)
println("Confusion matrix:")
println(metricsLRTest.confusionMatrix)
3.2979523E7  139.0      1181380.0    4464740.0
0.0          2478414.0  88.0         0.0
0.0          2588.0     1.4862553E7  0.0
0.0          0.0        0.0          1846.0
val accuracyLRTest = labelAndPredsLRTest.filter(r => r._1 == r._2).count.toDouble / testLR.count.toDouble
accuracyLRTest: Double = 0.8501800838408947
 
val labelsLRTest = metricsLRTest.labels
labelsLRTest.foreach { l =>
  println(s"Precision($l) = " + metricsLRTest.precision(l))
}
Precision(0.0) = 1.0
Precision(1.0) = 0.9989009088963505
Precision(2.0) = 0.9263608543020481
Precision(3.0) = 4.132910460024726E-4
 
// Recall by label
labelsLRTest.foreach { l =>
  println(s"Recall($l) = " + metricsLRTest.recall(l))
}
Recall(0.0) = 0.820615937154342
Recall(1.0) = 0.9999443223301346
Recall(2.0) = 0.8996121739594665
Recall(3.0) = 1.0
 
// False positive rate by label
labelsLRTest.foreach { l =>
  println(s"FPR($l) = " + metricsLRTest.falsePositiveRate(l))
}
FPR(0.0) = 0.0
FPR(1.0) = 4.8085353035714565E-5
FPR(2.0) = 0.027689050501391626
FPR(3.0) = 0.07543273456359546
 
// F-measure by label
labelsLRTest.foreach { l =>
  println(s"F1-Score($l) = " + metricsLRTest.fMeasure(l))
}
F1-Score(0.0) = 0.9014706730920752
F1-Score(1.0) = 0.9994223432781021
F1-Score(2.0) = 0.9127905934109559
F1-Score(3.0) = 8.262406141572705E-4
 
// Weighted stats
println(s"Weighted precision: ${metricsLRTest.weightedPrecision}")
Weighted precision: 0.9793687724257282
println(s"Weighted recall: ${metricsLRTest.weightedRecall}")
Weighted recall: 0.8501800838408948
println(s"Weighted F1 score: ${metricsLRTest.weightedFMeasure}")
Weighted F1 score: 0.9087038438294285
println(s"Weighted false positive rate: ${metricsLRTest.weightedFalsePositiveRate}")
Weighted false positive rate: 0.007732885300484439
 
//save Model
val now = Calendar.getInstance()
val modelName = "mlinput/LogitRegressionModel_"+now.get(Calendar.YEAR).toString+(now.get(Calendar.MONTH)+1).toString+now.get(Calendar.DAY_OF_MONTH).toString+now.get(Calendar.HOUR).toString+now.get(Calendar.MINUTE).toString+now.get(Calendar.SECOND).toString
modelName: String = mlinput/LogitRegressionModel_201711303021
LRmodel.save(sc, modelName)
val sameModel = LogisticRegressionModel.load(sc, modelName)
sameModel: org.apache.spark.mllib.classification.LogisticRegressionModel = org.apache.spark.mllib.classification.LogisticRegressionModel: intercept = 0.0, numFeatures = 10, numClasses = 6, threshold = 0.5