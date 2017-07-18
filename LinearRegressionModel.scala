//Linear regression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
 
val sqlContext = new SQLContext(sc)
 
// Make the schema
val fields = Array(StructField("DurationSince", DoubleType),
                                   StructField("Count", DoubleType))
val schema = StructType(fields)
 
val rdd = sc.textFile("input.csv")
 
val header = rdd.first()
val detail = rdd.filter(row => row != header)
val data = detail.map(_.split(",")).map(values => Row(values(2).toDouble,values(3).toDouble))
val df = sqlContext.createDataFrame(data,schema)
val assembler = new VectorAssembler().setInputCols(Array("DurationSince")).setOutputCol("Features")
val rdd = assembler.transform(df)
val splits = rdd.randomSplit(Array(0.8,0.2))
val training = splits(0).cache()
val test = splits(1).cache()
val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFeaturesCol("features").setLabelCol("Count").setPredictionCol("Predicted_Count")
val lrmodel = lr.fit(training)
println(s"Coefficients: $")
println(s"Coefficients: $ { lrmodel.coefficients} Intercept: ${lrmodel.intercept}")
val trainingSummary = lrmodel.summary
println(s" numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
lrmodel.transform(test).show()
val model_result = lrmodel.transform(test) import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import java.math.BigDecimal
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.ml.regression.LinearRegression
import java.math.BigDecimal
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import java.math.BigDecimal
 
val sqlContext = new HiveContext(sc)
 
val summary = sqlContext.sql("select count(*) as COUNT,sum(acc_ldgr_bl) as ACC_LDGR_SUM,sum(int_acru_am) as INT_ACRU_SUM,sum(intreservebal) as INTERESERVEBAL_SUM,YEAR(asofdate) as YEAR,MONTH(asofdate) as MONTH,DAYOFMONTH(asofdate) as DAY from ale_homeloans_dev1.accountlevel group by asofdate order by YEAR,MONTH,DAY")
val header = summary.first
val data = summary.rdd.filter(row => row.toString != header.toString)
 
case class summarydata(count: Long, ldgrsum: java.math.BigDecimal, acrusum: java.math.BigDecimal, intreservesum: java.math.BigDecimal, year: Integer, month: Integer, day:Integer)
val summaryDF = data.map{case Row(n1:Long , n2:java.math.BigDecimal ,n3:java.math.BigDecimal ,n4:java.math.BigDecimal ,n5:Integer ,n6:Integer ,n7:Integer ) => summarydata(n1,n2,n3,n4,n5,n6,n7)}.toDF()
 
val temp = summaryDF.rdd.map{ row =>
  val temp1 = row.getAs[Long](0)
  val temp2 = row.getAs[java.math.BigDecimal](1)
  val temp3 = row.getAs[java.math.BigDecimal](2)
  val temp4 = row.getAs[java.math.BigDecimal](3)
  val temp5 = row.getAs[Integer](4)
  val temp6 = row.getAs[Integer](5)
  val temp7 = row.getAs[Integer](6)
  (temp1.toDouble,
   temp2.setScale(10, BigDecimal.ROUND_HALF_DOWN).toPlainString().toDouble,
   temp3.setScale(10, BigDecimal.ROUND_HALF_DOWN).toPlainString().toDouble,
   temp4.setScale(10, BigDecimal.ROUND_HALF_DOWN).toPlainString().toDouble,
   temp5.toDouble,temp6.toDouble,temp7.toDouble)
}
 
val temp = summaryDF.rdd.map{ row =>
  val temp1 = row.getAs[Long](0)
  val temp2 = row.getAs[java.math.BigDecimal](1)
  val temp3 = row.getAs[java.math.BigDecimal](2)
  val temp4 = row.getAs[java.math.BigDecimal](3)
  val temp5 = row.getAs[Integer](4)
  val temp6 = row.getAs[Integer](5)
  val temp7 = row.getAs[Integer](6)
 (temp1.toDouble,
   temp2.setScale(20, BigDecimal.ROUND_HALF_DOWN).toPlainString(),
   temp3.setScale(20, BigDecimal.ROUND_HALF_DOWN).toPlainString(),
   temp4.setScale(20, BigDecimal.ROUND_HALF_DOWN).toPlainString(),
   temp5.toDouble,temp6.toDouble,temp7.toDouble)
}
 
 
case class filtersummary(count: Double, ldgrsum: Double, acrusum: Double, intreservesum: Double, year: Double, month: Double, day:Double)
 
val input = temp.map{ case(count,ldgrsum,acrusum,intreservesum,year,month,day) => (count,ldgrsum,acrusum,intreservesum,year,month,day)}
 
val inputDF = input.toDF("Count","LedgerSum","IntAcruSum","IntReserveSum","Year","Month","Day")
 
val countDF = inputDF.select("Count","Year","Month","Day")
 
//Count LR model:
//===============
val assemblercnt = new VectorAssembler().setInputCols(Array("Year","Month","Day")).setOutputCol("Features")
val transCount = assemblercnt.transform(countDF)
 
// Split data into training (60%) and test (40%)
val splitsCnt = transCount.randomSplit(Array(0.8, 0.2), seed = 3L)
val (trainingCnt, testCnt) = (splitsCnt(0), splitsCnt(1))
 
// Fit the model
val lrCnt = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFeaturesCol("Features").setLabelCol("Count").setPredictionCol("predicted_count")
val lrModelCnt = lrCnt.fit(trainingCnt)
 
// Print info about the fitted model
println(s"Coefficients: ${lrModelCnt.coefficients} Intercept: ${lrModelCnt.intercept}")
val trainingSummaryCnt = lrModelCnt.summary
println(s"numIterations: ${trainingSummaryCnt.totalIterations}")
println(s"objectiveHistory: ${trainingSummaryCnt.objectiveHistory.toList}")
trainingSummaryCnt.residuals.show()
println(s"RMSE: ${trainingSummaryCnt.rootMeanSquaredError}")
println(s"r2: ${trainingSummaryCnt.r2}")
 
lrModelCnt.transform(test).show()
 
trainingCnt.take(3)
df.select("DurationSince").show(2)
 
val model_result_Cnt = lrModelCnt.transform(testCnt)
val result_df_Cnt = model_result_Cnt.select("Count","predicted_count")
 
//Ledger Balance LR model:
//========================
val ldgrDF = inputDF.select("LedgerSum","Year","Month","Day")
val assemblerldgr = new VectorAssembler().setInputCols(Array("Year","Month","Day")).setOutputCol("Features")
val ldgrSum = assemblerldgr.transform(ldgrDF)
 
val splitsLdgr = ldgrSum.randomSplit(Array(0.8, 0.2), seed = 3L)
val (trainingLdgr, testLdgr) = (splitsLdgr(0), splitsLdgr(1))
 
// Fit the model
val lrLdgr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFeaturesCol("Features").setLabelCol("LedgerSum").setPredictionCol("predicted_LedgerSum")
val lrModelLdgr = lrLdgr.fit(trainingLdgr)
 
// Print info about the fitted model
println(s"Coefficients: ${lrModelLdgr.coefficients} Intercept: ${lrModelLdgr.intercept}")
val trainingSummaryLdgr = lrModelLdgr.summary
println(s"numIterations: ${trainingSummaryLdgr.totalIterations}")
println(s"objectiveHistory: ${trainingSummaryLdgr.objectiveHistory.toList}")
trainingSummaryLdgr.residuals.show()
println(s"RMSE: ${trainingSummaryLdgr.rootMeanSquaredError}")
println(s"r2: ${trainingSummaryLdgr.r2}")
 
lrModelLdgr.transform(testLdgr).show()
 
trainingLdgr.take(3)
df.select("DurationSince").show(2)
 
val model_result_Ldgr = lrModelLdgr.transform(testLdgr)
val result_df_Ldgr = model_result_Ldgr.select("LedgerSum","predicted_LedgerSum")
 
//Interest Acru LR model:
//=======================
val interestacruDF = inputDF.select("IntAcruSum","Year","Month","Day")
val assemblerintacru = new VectorAssembler().setInputCols(Array("Year","Month","Day")).setOutputCol("Features")
val interestacruSum = assemblerintacru.transform(interestacruDF)
 
val splitsIntAcru = interestacruSum.randomSplit(Array(0.8, 0.2), seed = 3L)
val (trainingIntAcru, testIntAcru) = (splitsIntAcru(0), splitsIntAcru(1))
 
// Fit the model
val lrIntAcru = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFeaturesCol("Features").setLabelCol("IntAcruSum").setPredictionCol("predicted_IntAcruSum")
val lrModelIntAcru = lrIntAcru.fit(trainingIntAcru)
 
// Print info about the fitted model
println(s"Coefficients: ${lrModelIntAcru.coefficients} Intercept: ${lrModelIntAcru.intercept}")
val trainingSummaryIntAcru = lrModelIntAcru.summary
println(s"numIterations: ${trainingSummaryIntAcru.totalIterations}")
println(s"objectiveHistory: ${trainingSummaryIntAcru.objectiveHistory.toList}")
trainingSummaryIntAcru.residuals.show()
println(s"RMSE: ${trainingSummaryIntAcru.rootMeanSquaredError}")
println(s"r2: ${trainingSummaryIntAcru.r2}")
 
lrModelIntAcru.transform(testIntAcru).show()
 
trainingIntAcru.take(3)
df.select("DurationSince").show(2)
 
val model_result_IntAcru = lrModelIntAcru.transform(testIntAcru)
val result_df_IntAcru = model_result_IntAcru.select("IntAcruSum","predicted_IntAcruSum")
 
//Interest reserve balance LR model:
//==================================
val intreserveDF = inputDF.select("IntReserveSum","Year","Month","Day")
val assemblerintreserve = new VectorAssembler().setInputCols(Array("Year","Month","Day")).setOutputCol("Features")
val intreserveSum = assemblerintreserve.transform(intreserveDF)
 
val splitsIntRes = intreserveSum.randomSplit(Array(0.8, 0.2), seed = 3L)
val (trainingIntRes, testIntRes) = (splitsIntRes(0), splitsIntRes(1))
 
// Fit the model
val lrIntRes = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFeaturesCol("Features").setLabelCol("IntReserveSum").setPredictionCol("predicted_IntReserveSum")
val lrModelIntRes = lrIntRes.fit(trainingIntRes)
 
// Print info about the fitted model
println(s"Coefficients: ${lrModelIntRes.coefficients} Intercept: ${lrModelIntRes.intercept}")
val trainingSummaryIntRes = lrModelIntRes.summary
println(s"numIterations: ${trainingSummaryIntRes.totalIterations}")
println(s"objectiveHistory: ${trainingSummaryIntRes.objectiveHistory.toList}")
trainingSummaryIntRes.residuals.show()
println(s"RMSE: ${trainingSummaryIntRes.rootMeanSquaredError}")
println(s"r2: ${trainingSummaryIntRes.r2}")
 
lrModelIntRes.transform(testIntRes).show()
 
trainingIntRes.take(3)
df.select("DurationSince").show(2)
 
val model_result_IntRes = lrModelIntRes.transform(testIntRes)
val result_df_IntRes = model_result_IntRes.select("IntReserveSum","predicted_IntReserveSum")
 
//===============================================================================
 
val numIterations = 100
val model = LinearRegressionWithSGD.train(conv, numIterations)
 
// Evaluate model on training examples and compute training error
val valuesAndPreds = conv.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
 
val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
println("training Mean Squared Error = " + MSE)
 
val training = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
 
val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
 
val lrModel = lr.fit(conv.toDF)
 
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
 
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
 
//XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 
val summary = sqlContext.sql("select count(*) as COUNT,sum(acc_ldgr_bl) as ACC_LDGR_SUM,sum(int_acru_am) as INT_ACRU_SUM,sum(intreservebal) as INTERESERVEBAL_SUM,YEAR(asofdate) as YEAR,MONTH(asofdate) as MONTH,DAYOFMONTH(asofdate) as DAY from ale_homeloans_dev1.accountlevel group by asofdate sort by YEAR,MONTH,DAY")
 
+--------+--------------------+--------------------+--------------------+----+-----+---+
|   COUNT|        ACC_LDGR_SUM|        INT_ACRU_SUM|  INTERESERVEBAL_SUM|YEAR|MONTH|DAY|
+--------+--------------------+--------------------+--------------------+----+-----+---+
|25047183|610370395915.5100...|2030263367.710000...|-1391507758.08000...|2016|    9| 30|
|25169222|608687324595.5600...|2046425715.030000...|-1359913123.87000...|2016|   10| 31|
|25047183|610370395915.5100...|2030263367.710000...|-1391507758.08000...|2014|   12| 31|
|24727133|629988120939.3200...|2222252771.220000...|-1540104033.99000...|2016|    6| 30|
|24370332|641399506336.4900...|2319019525.710000...|-1649932431.63000...|2016|    5| 31|
|24370332|641399506336.4900...|2319019525.710000...|-1649932431.63000...|2016|    4| 30|
|24307508|643473817562.5500...|2362048508.450000...|-1702538068.98000...|2016|    3| 31|
+--------+--------------------+--------------------+--------------------+----+-----+---+
 
val summary = sqlContext.sql("select count(*) as COUNT,sum(acc_ldgr_bl) as ACC_LDGR_SUM,sum(int_acru_am) as INT_ACRU_SUM,sum(intreservebal) as INTERESERVEBAL_SUM,YEAR(asofdate) as YEAR,MONTH(asofdate) as MONTH,DAYOFMONTH(asofdate) as DAY from ale_homeloans_dev1.accountlevel group by asofdate")
 
+--------+--------------------+--------------------+--------------------+----+-----+---+
|   COUNT|        ACC_LDGR_SUM|        INT_ACRU_SUM|  INTERESERVEBAL_SUM|YEAR|MONTH|DAY|
+--------+--------------------+--------------------+--------------------+----+-----+---+
|25047183|610370395915.5100...|2030263367.710000...|-1391507758.08000...|2016|    9| 30|
|25169222|608687324595.5600...|2046425715.030000...|-1359913123.87000...|2016|   10| 31|
|25047183|610370395915.5100...|2030263367.710000...|-1391507758.08000...|2014|   12| 31|
|24727133|629988120939.3200...|2222252771.220000...|-1540104033.99000...|2016|    6| 30|
|24370332|641399506336.4900...|2319019525.710000...|-1649932431.63000...|2016|    5| 31|
|24370332|641399506336.4900...|2319019525.710000...|-1649932431.63000...|2016|    4| 30|
|24307508|643473817562.5500...|2362048508.450000...|-1702538068.98000...|2016|    3| 31|
+--------+--------------------+--------------------+--------------------+----+-----+---+
val predicted_result = model_result.select("Predicted_Count")
val actual_result = model_result.select("Count")
val result_df = model_result.select("Count","Predicted_Count")
 
test.take(3)
val labelPredRDD = result_df.rdd.map(row => (row.apply(0).asInstanceOf[Double], row.apply(1).asInstanceOf[Double]))
val rm = new RegressionMetrics(labelPredRDD)
println(Math.sqrt(rm.meanSquaredError))
 
val X = model_result.rdd.map(row => (row(0).asInstanceOf[Double]))
val Y = model_result.rdd.map(row => (row(1).asInstanceOf[Double]))
val correlation = Statistics.corr(X,Y,"pearson")
#correlation: Double = 0.9938199821798208
 