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
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import java.util.Calendar
val sqlContext = new HiveContext(sc)
 
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
val RFindexer = new StringIndexer().setInputCol("CustomerType").setOutputCol("CustomerTypeIndex").fit(df2016)
val RFindexed = RFindexer.transform(df2016)
val RFcoder = new OneHotEncoder().setInputCol("CustomerTypeIndex").setOutputCol("CustomerTypeVec")
val RFcoded = RFcoder.transform(RFindexed)
val InputCols = RFcoded.select($"CustomerTypeIndex",$"FICOScore",$"Delinquency")
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
 
val labeledPointsRF = InputCols.rdd.map(parts => LabeledPoint(parts.getAs[Double](0), Vectors.dense(parts.getAs[Double](1),parts.getAs[Double](2))))
val Array(trainingRF, testRF) = labeledPointsRF.randomSplit(Array(0.7, 0.3), seed = 5043l)
 
val numClasses = 6
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 20 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 4
val maxBins = 32
val seed = 5043
 
val startRF=System.nanoTime()
val RFmodel = RandomForest.trainClassifier(trainingRF, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
val endRF = System.nanoTime()
println("Time elapsed: " + (endRF-startRF)/1000 + " microsecs")
Time elapsed: 34620706 microsecs
 
//prediction on training data
val labeledPredictionsRFTrain = trainingRF.map { labeledPoint =>
    val predictions = RFmodel.predict(labeledPoint.features)
    (labeledPoint.label, predictions)
}
val predictionAndLabelsRFTrain = labeledPredictionsRFTrain.map { x => (x._1.toDouble,x._2.toDouble) }
 
val metricsRFTrain = new MulticlassMetrics(predictionAndLabelsRFTrain)
println("Confusion matrix:")
println(metricsRFTrain.confusionMatrix)
5.7118876E7  0.0          0.0        0.0        0.0        0.0
0.0          2.7456565E7  0.0        0.0        0.0        0.0
0.0          0.0          7810327.0  0.0        0.0        0.0
0.0          0.0          0.0        4341585.0  977.0      0.0
0.0          628610.0     0.0        0.0        2965057.0  0.0
613484.0     0.0          5864.0     0.0        0.0        2664176.0
 
val labelsRFTrain = metricsRFTrain.labels
labelsRFTrain.foreach { l =>
  println(s"Precision($l) = " + metricsRFTrain.precision(l))
}
Precision(0.0) = 0.9893736545673865
Precision(1.0) = 0.977617728926382
Precision(2.0) = 0.9992497624482308
Precision(3.0) = 1.0
Precision(4.0) = 0.9996706039108115
Precision(5.0) = 1.0
 
// Recall by label
labelsRFTrain.foreach { l =>
  println(s"Recall($l) = " + metricsRFTrain.recall(l))
}
Recall(0.0) = 1.0
Recall(1.0) = 1.0
Recall(2.0) = 1.0
Recall(3.0) = 0.9997750176048149
Recall(4.0) = 0.8250783948540585
Recall(5.0) = 0.8113770449066308
 
// False positive rate by label
labelsRFTrain.foreach { l =>
  println(s"FPR($l) = " + metricsRFTrain.falsePositiveRate(l))
}
FPR(0.0) = 0.013196994534666892
FPR(1.0) = 0.00825500483552263
FPR(2.0) = 6.121392686985947E-5
FPR(3.0) = 0.0
FPR(4.0) = 9.768842001469145E-6
FPR(5.0) = 0.0
 
// F-measure by label
labelsRFTrain.foreach { l =>
  println(s"F1-Score($l) = " + metricsRFTrain.fMeasure(l))
}
F1-Score(0.0) = 0.9946584466883751
F1-Score(1.0) = 0.9886822054908615
F1-Score(2.0) = 0.999624740457215
F1-Score(3.0) = 0.9998874961467142
F1-Score(4.0) = 0.9040219973440863
F1-Score(5.0) = 0.8958676463170637
 
// Weighted stats
println(s"Weighted precision: ${metricsRFTrain.weightedPrecision}")
Weighted precision: 0.9881420554012593
println(s"Weighted recall: ${metricsRFTrain.weightedRecall}")
Weighted recall: 0.9879452852710426
println(s"Weighted F1 score: ${metricsRFTrain.weightedFMeasure}")
Weighted F1 score: 0.9873934826319446
println(s"Weighted false positive rate: ${metricsRFTrain.weightedFalsePositiveRate}")
Weighted false positive rate: 0.009468267410103502
 
//prediction on test data
val labeledPredictionsRFTest = testRF.map { labeledPoint =>
    val predictions = RFmodel.predict(labeledPoint.features)
    (labeledPoint.label, predictions)
}
val predictionAndLabelsRFTest = labeledPredictionsRFTest.map { x => (x._1.toDouble,x._2.toDouble) }
 
val metricsRFTest = new MulticlassMetrics(predictionAndLabelsRFTest)
println("Confusion matrix:")
println(metricsRFTest.confusionMatrix)
2.44676E7  0.0          0.0        0.0        0.0        0.0
0.0        1.1762358E7  0.0        0.0        0.0        0.0
0.0        0.0          3349093.0  0.0        0.0        0.0
0.0        0.0          0.0        1861096.0  405.0      0.0
0.0        268644.0     0.0        0.0        1271888.0  0.0
261928.0   0.0          2571.0     0.0        0.0        1140606.0
 
val labelsRFTest = metricsRFTest.labels
labelsRFTest.foreach { l =>
  println(s"Precision($l) = " + metricsRFTest.precision(l))
}
Precision(0.0) = 0.9894082895557085
Precision(1.0) = 0.9776706877781252
Precision(2.0) = 0.9992329183354894
Precision(3.0) = 1.0
Precision(4.0) = 0.9996816770979641
Precision(5.0) = 1.0
 
// Recall by label
labelsRFTest.foreach { l =>
  println(s"Recall($l) = " + metricsRFTest.recall(l))
}
Recall(0.0) = 1.0
Recall(1.0) = 1.0
Recall(2.0) = 1.0
Recall(3.0) = 0.9997824336382306
Recall(4.0) = 0.8256160858716339
Recall(5.0) = 0.8117585518519969
 
// False positive rate by label
labelsRFTest.foreach { l =>
  println(s"FPR($l) = " + metricsRFTest.falsePositiveRate(l))
}
FPR(0.0) = 0.013149927437129207
FPR(1.0) = 0.00823459390774799
FPR(2.0) = 6.265063200378506E-5
FPR(3.0) = 0.0
FPR(4.0) = 9.45253331043564E-6
FPR(5.0) = 0.0
 
// F-measure by label
labelsRFTest.foreach { l =>
  println(s"F1-Score($l) = " + metricsRFTest.fMeasure(l))
}
F1-Score(0.0) = 0.9946759493765571
F1-Score(1.0) = 0.9887092869607319
F1-Score(2.0) = 0.9996163120077328
F1-Score(3.0) = 0.9998912049840474
F1-Score(4.0) = 0.9043491863162478
F1-Score(5.0) = 0.8961001464816706
 
// Weighted stats
println(s"Weighted precision: ${metricsRFTest.weightedPrecision}")
Weighted precision: 0.9881751841059351
println(s"Weighted recall: ${metricsRFTest.weightedRecall}")
Weighted recall: 0.9879794140470135
println(s"Weighted F1 score: ${metricsRFTest.weightedFMeasure}")
Weighted F1 score: 0.9874307068430429
println(s"Weighted false positive rate: ${metricsRFTest.weightedFalsePositiveRate}")
Weighted false positive rate: 0.009436038557205005
//save Model
val now = Calendar.getInstance()
val modelName = "mlinput/RandomForestModel_"+now.get(Calendar.YEAR).toString+(now.get(Calendar.MONTH)+1).toString+now.get(Calendar.DAY_OF_MONTH).toString+now.get(Calendar.HOUR).toString+now.get(Calendar.MINUTE).toString+now.get(Calendar.SECOND).toString
modelName: String = mlinput/RandomForestModel_201711211257
RFmodel.save(sc, modelName)
val sameModel = RandomForestModel.load(sc, modelName)
sameModel: org.apache.spark.mllib.tree.model.RandomForestModel =
TreeEnsembleModel classifier with 20 trees
 
/////////////////
Outlier detection
path <- "C:\\Users\\zkdzgop\\Desktop\\Algorithm\\OutliersDetection"
setwd(path)
data <- read.csv("ozone.csv",header = TRUE, stringsAsFactors = FALSE)
str(data)
#colnames(data) <- c("LOANNUM","LDGRBAL","RECOVERYAMT","BANKRUPTCYFLAG","CLOSECODE","BANKRUPTSTATCD","ASOFDATE")
#input <- data.frame(subset(data,select=-LOANNUM,-ASOFDATE))
install.packages("forecast")
library(forecast)
t_s<- ts(data$ozone_reading)
plot(t_s)
tsoutliers(t_s)
t_s_new<-tsclean(t_s)
plot(t_s_new)
tsoutliers(data)
mean(data$ozone_reading)
boxplot(data$ozone_reading)
identify(rep(1, length(data$ozone_reading)), data$ozone_reading, labels = seq_along(data$ozone_reading))
 
\\\\\\\\\\\\\\\\\\\\\\\\\\\
path <- "C:\\Users\\zkdzgop\\Desktop\\Algorithm\\"
setwd(path)
getwd()
 
library(rpart)
input <- read.csv("Features.csv",header = TRUE, sep=",",stringsAsFactors = FALSE)
shuffle_input <- input[sample(nrow(input),nrow(input)),]
dim(shuffle_input)
table(shuffle_input$PredClassify)
sample_size <- floor(0.8 * nrow(input))
set.seed(123)
colSums(is.na(shuffle_input))
table(is.na(shuffle_input))
train_ind <- sample(seq_len(nrow(input)),size = sample_size)
train_shuff <- shuffle_input[train_ind,5:44]
test_shuff <- shuffle_input[-train_ind,5:44]
shuff_train_labels <- shuffle_input[train_ind,5]
shuff_test_labels <- shuffle_input[-train_ind,5]
table(shuff_train_labels)
table(shuff_test_labels)
fit <- rpart(PredClassify ~ ., method="class", data=train_shuff)
#fit <- rpart(ActClassify ~ ., method="class", data=train_shuff)
printcp(fit)
plotcp(fit)
summary(fit)
plot(fit, uniform=TRUE,main="Classification Tree for Transformation Classification")
text(fit, use.n=TRUE, all=TRUE, cex=.8)
pred <- predict(fit,  newdata = test_shuff,type='class')
table(pred == shuff_test_labels)
pred
confusionMatrix(pred,shuff_test_labels)
 
Confusion Matrix and Statistics
 
                  Reference
Prediction         concatenate datederived datereformat derived masking numeric reformat
  concatenate             1122           0            0       0       0                0
  datederived                0         124            0       0       0                0
  datereformat               0           0          500       0       0                0
  derived                    0           0            0       0       0                0
  masking                    0           0            0       0    1152                0
  numeric reformat           0           0            0       0       0              479
  sm                         0           2            0      73       0                3
  stringreformat             0           0            0       0       0                0
                  Reference
Prediction           sm stringreformat
  concatenate         0              0
  datederived         0              0
  datereformat        0              0
  derived             0              0
  masking             0              0
  numeric reformat    0              0
  sm               1129              2
  stringreformat      0              0
 
Overall Statistics
                                         
               Accuracy : 0.9826         
                 95% CI : (0.9783, 0.9861)
    No Information Rate : 0.2512         
    P-Value [Acc > NIR] : < 2.2e-16      
                                          
                  Kappa : 0.9779         
 Mcnemar's Test P-Value : NA             
 
Statistics by Class:
 
                     Class: concatenate Class: datederived Class: datereformat
Sensitivity                      1.0000            0.98413               1.000
Specificity                      1.0000            1.00000               1.000
Pos Pred Value                   1.0000            1.00000               1.000
Neg Pred Value                   1.0000            0.99955               1.000
Prevalence                       0.2447            0.02747               0.109
Detection Rate                   0.2447            0.02704               0.109
Detection Prevalence             0.2447            0.02704               0.109
Balanced Accuracy                1.0000            0.99206               1.000
                     Class: derived Class: masking Class: numeric reformat Class: sm
Sensitivity                 0.00000         1.0000                  0.9938    1.0000
Specificity                 1.00000         1.0000                  1.0000    0.9769
Pos Pred Value                  NaN         1.0000                  1.0000    0.9338
Neg Pred Value              0.98408         1.0000                  0.9993    1.0000
Prevalence                  0.01592         0.2512                  0.1051    0.2462
Detection Rate              0.00000         0.2512                  0.1044    0.2462
Detection Prevalence        0.00000         0.2512                  0.1044    0.2636
Balanced Accuracy           0.50000         1.0000                  0.9969    0.9884
                     Class: stringreformat
Sensitivity                      0.0000000
Specificity                      1.0000000
Pos Pred Value                         NaN
Neg Pred Value                   0.9995639
Prevalence                       0.0004361
Detection Rate                   0.0000000
Detection Prevalence             0.0000000
Balanced Accuracy                0.5000000
