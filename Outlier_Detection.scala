import scala.math.BigDecimal
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
 
val data = sqlContext.sql("select loannum,acc_ldgr_bl,rpt_ldgr_bl,ast_clc_cd,ast_clc_cd_cls,prp_typ_cd,obl_lien_cd,acc_blok_in_cls,acc_lgl_stat_cd,bookvalamt,finactiveflag,finstatuscd,investtypecd,lienpositioncd,repurchtypecd,recoverymtdamt,bnkrptcyflag,closedcd,cus_bnkrpt_stat_cd,int_acru_am,prinbalcontrabal,asofdate from ale_homeloans_dev1.accountlevel")
 
def getRddPercentile(inputScore: RDD[Double], percentile: Double): Double = {
    val numEntries = inputScore.count().toDouble
    val retrievedEntry = (percentile * numEntries / 100.0 )
    inputScore
      .sortBy { case (score) => score }
      .zipWithIndex()
      .filter { case (score, index) => index == retrievedEntry }
      .map { case (score, index) => score }
      .collect()(0)
}
def computeQuantileSorted(variable: RDD[Double], quantile: Double): Double = {
  val nb = variable.count
  if (nb == 1) variable.first
  else {
    val n = (quantile / 100.0) * nb
    val k = math.round(n).toLong
    val d = n - k
    if (k <= 0) variable.first()
    else {
      val index = variable.zipWithIndex().map(_.swap)
      if (k >= nb) {
        index.lookup(nb - 1).head
      }
      else {
        index.lookup(k - 1).head // + d * (index.lookup(k).head - index.lookup(k - 1).head)
      }
    }
  }
}
def hampelID(sortedinput: RDD[Double]) : (RDD[Double],Double,Double) = {
  val mean = sortedinput.sum / (sortedinput.count * 1.0)
  val absValDiff = sortedinput.map(values => Math.abs(values - mean))
  val MAD = absValDiff.sum / (sortedinput.count * 1.0)
  val median = computeQuantileSorted(sortedinput,50)
  val lowlimit = median - (3 * MAD)
  val highlimit = median + (3 * MAD)
  val outliers = sortedinput.filter(row => (row < lowlimit) || (row > highlimit))
  return (outliers, median, MAD)
}
def standardboxplot(input: RDD[Double], level: String) : (RDD[Double],Double,Double,Double,Double) = {
  val q1 = computeQuantileSorted(input,25)
  val q2 = computeQuantileSorted(input,50)
  val q3 = computeQuantileSorted(input,75)
  val iqd = q3-q1
  val range = level.toLowerCase match {
      case "low" => ( q1 - 1.5 * (q3-q1), q1 + 1.5 *(q3-q1) )
      case "strong" => ( q1 - 3 * (q3-q1), q1 + 3 *(q3-q1) )
  }
  val outliers = input.filter(row => ((row < range._1) || (row > range._2)) )
  return (outliers,q1,q2,q3,iqd)
}
 
def esd(input: RDD[Double]): (RDD[Double],RDD[Double],Double,Double) = {
  val count = input.count
  val mean = input.sum /(count * 1.0)
  val devs = input.map( l => (l - mean) * (l - mean) )
  val stddev = Math.sqrt( devs.sum / (count) )
  val minlimit = (mean - (3*stddev))
  val maxlimit = (mean + (3*stddev))
  val outliers = input.filter(row => row > minlimit).filter(row => row < maxlimit)
  val inrange = input.filter(row => row < minlimit || row > maxlimit)
  return (outliers,inrange,mean,stddev)
}
 
val formatLDGRMAR = data.rdd.map{row =>
  val temp1 = row.apply(0).asInstanceOf[String]
  val temp2 = row.apply(1).asInstanceOf[java.math.BigDecimal]
  (temp1.toString,temp2.setScale(30, BigDecimal.RoundingMode.HALF_UP).toDouble)
}
 
val datardd = data.select($"acc_ldgr_bl").distinct.sort("acc_ldgr_bl").map{ row =>
val temp = row.getAs[java.math.BigDecimal](0)
(temp.setScale(10,BigDecimal.ROUND_HALF_DOWN).toPlainString().toDouble)
}
 
val (outliers,median,mad) = hampelID(datardd)
outliers.count
res7: Long = 175971
median: Double = 99265.74
mad: Double = 110538.21915012132
 
val (outliers,inrange,mean,stddev) = esd(datardd)
outliers: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[139] at filter at <console>:52
inrange: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[140] at filter at <console>:53
mean: Double = 148814.97200044178
stddev: Double = 234513.20402190727
 
val (outliers,q1,q2,q3,iqd) = standardboxplot(datardd,"strong")
outliers: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[153] at filter at <console>:56
q1: Double = 45514.12
q2: Double = 99265.74
q3: Double = 184554.97
iqd: Double = 139040.85
 
//sample – code for trial
accntlvl2016mar.select(avg($"acc_ldgr_bl")).show()
val statsmar = accntlvl2016mar.describe()
val stddevmar2016 = accntlvl2016mar.select(stddev($"acc_ldgr_bl"),avg($"acc_ldgr_bl"))
val test = stddevmar2016.select(stddevmar2016("stddev_samp(acc_ldgr_bl,0,0)") - stddevmar2016("avg(acc_ldgr_bl)"))
val stddevmar2016 = accntlvl2016mar.select(stddev($"acc_ldgr_bl"))
val avgmar2016 = accntlvl2016mar.select(avg($"acc_ldgr_bl"))
val minlimit = (avgmar2016 - (3*stddevmar2016))
val maxlimit = (avgmar2016 + (3*stddevmar2016))
val mar2016df = accntlvl2016mar.select($"acc_ldgr_bl")
val mar2016df = accntlvl2016mar.select($"acc_ldgr_bl").distinct.sort("acc_ldgr_bl")
//