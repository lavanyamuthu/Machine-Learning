import org.apache.spark.sql.{Row, SQLContext}

import org.apache.spark.sql.types._

import java.text.DateFormat

import java.text.SimpleDateFormat

import java.text.ParseException

import java.math.BigDecimal

import java.util.Calendar

import org.apache.spark.sql._

import org.apache.spark.sql.types._

import org.apache.spark.sql.functions._

import org.apache.spark.sql.hive.HiveContext

import scala.util.control.Breaks._

import scala.collection.mutable.HashMap

import org.apache.spark.mllib.feature.StandardScaler

import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.{SparkConf, SparkContext}

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

 

val data = sc.textFile("Input_Large.csv").map(_.split(",")).map(values => Row(values(0).toString,values(1).toString,values(2).toString,values(3).toString))

data.count

val fields = Array(StructField("InputCol1",StringType),

                   StructField("InputCol2",StringType),

                   StructField("OutputCol",StringType),

                   StructField("Classification",StringType))

val schema = StructType(fields)

val df = sqlContext.createDataFrame(data,schema)

val dfextend = df.withColumn("InputString1",df("InputCol1")).

                withColumn("InputString2",df("InputCol2")).

                withColumn("OutputString",df("OutputCol")).

                withColumn("ActClassify",df("Classification")).

                withColumn("PredClassify",lit(null).cast(StringType)).

                withColumn("DeleteString",lit(null).cast(StringType)).

                withColumn("InsertString",lit(null).cast(StringType)).

                withColumn("MoveString",lit(null).cast(StringType)).

                withColumn("DelSpBegFlag",lit(0.0)).

                withColumn("DelSpBetFlag",lit(0.0)).

                withColumn("DelSpEndFlag",lit(0.0)).

                withColumn("DelzBegFlag",lit(0.0)).

                withColumn("DelzBetFlag",lit(0.0)).

                withColumn("DelzEndFlag",lit(0.0)).

                withColumn("DelBegFlag",lit(0.0)).

                withColumn("DelBetFlag",lit(0.0)).

                withColumn("DelEndFlag",lit(0.0)).

                withColumn("InsSpBegFlag",lit(0.0)).

                withColumn("InsSpBetFlag",lit(0.0)).

                withColumn("InsSpEndFlag",lit(0.0)).

                withColumn("InsFlag",lit(0.0)).

                withColumn("InsSameFlag",lit(0.0)).

                withColumn("InszBegFlag",lit(0.0)).

                withColumn("InsDecEndFlag",lit(0.0)).

                withColumn("DelDecEndFlag",lit(0.0)).

                withColumn("MovFlag",lit(0.0)).

                withColumn("ConcatFlag",lit(0.0)).

                withColumn("YYYYChngFlag",lit(0.0)).

                withColumn("YYChngFlag",lit(0.0)).

                withColumn("MMChngFlag",lit(0.0)).

                withColumn("DDChngFlag",lit(0.0)).

                withColumn("TSChngFlag",lit(0.0)).

                withColumn("HrChngFlag",lit(0.0)).

                withColumn("MinChngFlag",lit(0.0)).

                withColumn("SecChngFlag",lit(0.0)).

                withColumn("TZChngFlag",lit(0.0)).

                withColumn("YYYYFrmtChngFlag",lit(0.0)).

                withColumn("YYFrmtChngFlag",lit(0.0)).

                withColumn("MMFrmtChngFlag",lit(0.0)).

                withColumn("DDFrmtChngFlag",lit(0.0)).

                withColumn("DelimFrmtChngFlag",lit(0.0)).

                withColumn("TSFrmtChngFlag",lit(0.0)).

                withColumn("HrFrmtChngFlag",lit(0.0)).

                withColumn("MinFrmtChngFlag",lit(0.0)).

                withColumn("SecFrmtChngFlag",lit(0.0)).

                withColumn("TZFrmtChngFlag",lit(0.0)).

                withColumn("DateFrmtChngFlag",lit(0.0)).

                drop(df("InputCol1")).

                drop(df("InputCol2")).

                drop(df("OutputCol")).

                drop(df("Classification"))

 

dfextend.count

 

val df_sm = dfextend.withColumn("PredClassify",when($"InputString1" === $"OutputString","sm").otherwise("not sm"))

val df_concat = df_sm.withColumn("PredClassify",when((concat($"InputString1",$"InputString2") === $"OutputString") && ($"InputString1" !== "") && ($"InputString2" !== "") && ($"OutputString" !== "") && ($"PredClassify" !== "sm"),"concatenate").otherwise(df_sm.col("PredClassify"))).

                      withColumn("ConcatFlag",when($"PredClassify" === "concatenate",1.0).otherwise(df_sm.col("ConcatFlag")))

val df_nonconcat = df_concat.withColumn("PredClassify",when(($"InputString1" !== "") && ($"InputString2" !== "") && ($"OutputString" !== "") && (concat($"InputString1",$"InputString2") !== $"OutputString"),"derivedconcat").otherwise(df_concat.col("PredClassify")))

val intorfloat = """(?=.)[+-]?([0-9]+)?(\.)?([0-9]+)?""".r

 

def toFloat(s: String): Float = {

  try {

    s.toFloat

  } catch {

    case e: Exception => 0

  }

}

def toInt(s: String): Int = {

  try {

    s.toInt

  } catch {

    case e: Exception => 0

  }

}

 

def validateNumber(ip:String,op:String) : (String,String,String,Double,Double,Double,Double) = {

  var op1 = ""

  var op2 = ""

  var classifyip = "defip"

  var classifyop = "defop"

  var classify = "def"

  var delzbeg = 0.0

  var inszbeg = 0.0

  var deldecend = 0.0

 var insdecend = 0.0

  var ipi:String = null

  var ipd:String = null

  var ipde:String = null

  var opi:String = null

  var opd:String = null

  var opde:String = null

  ip match {

     case intorfloat(i,d,de) => {

       ipi = i

       ipd = d

      ipde = de

       if(i == null && (d != null && de != null)) {

         classifyip = "float"

       }

       if(i != null && (d != null && de != null)) {

         classifyip = "float"

       }

       if(i != null && (d == null && de == null)) {

         classifyip = "intonly"        

       }

       if(i == null && (d != null && de == null)) {

         classifyip = "ip is not valid float"

       }

       if(i != null && (d != null && de == null)) {

         classifyip = "ip is not valid float"

      }

     }

     case _ => classifyip = "ip is not numeric"

  }

  op match {

     case intorfloat(i,d,de) => {

       opi = i

       opd = d

       opde = de

       if(i == null && (d != null && de != null)) {

         classifyop = "float"

       }

       if(i != null && (d != null && de != null)) {

         classifyop = "float"

       }

       if(i != null && (d == null && de == null)) {

         classifyop = "intonly"        

       }

       if(i == null && (d != null && de == null)) {

         classifyop = "op is not valid float"

       }

       if(i != null && (d != null && de == null)) {

         classifyop = "op is not valid float"

       }

     }

     case _ => classifyop = "op is not numeric"

  }

  if(classifyip == "float" && classifyop == "float") {

    if(toFloat(ip) == toFloat(op)) {

      if(ip != op) {

        classify = "float reformat"

        if( ipi != null && opi != null && (ipi.length > opi.length) ) { delzbeg = 1.0}

        if( ipi != null && opi != null && (ipi.length < opi.length) ) { inszbeg = 1.0}

        if( ipi != null && opi == null ) { delzbeg = 1.0}

        if( ipi == null && opi != null ) {

          println("In step1")

          inszbeg = 1.0}

        if( ipde != null && opde != null && (ipde.length > opde.length) ){ deldecend = 1.0}

        if( ipde != null && opde != null && (ipde.length < opde.length) ) { insdecend = 1.0}

        if( ipde != null && opde == null ) { deldecend = 1.0}

        if( ipde == null && opde != null ) { insdecend = 1.0}

      }

      else {

        classify = "sm"

      }

    }

    else {

      classify = "derived"

    }

  }

  if(classifyip == "intonly" && classifyop == "intonly") {

    if(toInt(ip) == toInt(op)) {

      if(ip != op) {

        classify = "int reformat"

        if( ip != null && opi != null && (ip.length > op.length) )

          delzbeg = 1.0

        if( ip != null && opi != null && (ip.length < op.length) )

          inszbeg = 1.0

      }

      else {

        classify = "sm"

      }

    }

    else {

      classify = "int derived"

    }

  }

  if((classifyip == "float" && classifyop == "intonly") || (classifyip == "intonly" && classifyop == "float")){

    if(toFloat(ip) == toFloat(op)) {

      if(classifyip == "float") {

       if(ip != op) {

          classify = "float reformat"

          if( ipi != null && opi != null && (ipi.length > opi.length) ) { delzbeg = 1.0}

          if( ipi != null && opi != null && (ipi.length < opi.length) ) { inszbeg = 1.0}

          if( ipi != null && opi == null ) { delzbeg = 1.0}

          if( ipi == null && opi != null ) { inszbeg = 1.0}

          if( ipde != null && opde != null && (ipde.length > opde.length) ){ deldecend = 1.0}

          if( ipde != null && opde != null && (ipde.length < opde.length) ){ insdecend = 1.0}

          if( ipde != null && opde == null ) { deldecend = 1.0}

          if( ipde == null && opde != null ) { insdecend = 1.0}

        }

        else {

          classify = "sm"

        }

      }

      if(classifyip == "intonly") {

        if(ip != op) {

          classify = "int reformat"

          if( ipi != null && opi != null && (ipi.length > opi.length) ) { delzbeg = 1.0}

          if( ipi != null && opi != null && (ipi.length < opi.length) ) { inszbeg = 1.0}

          if( ipi != null && opi == null ) { delzbeg = 1.0}

          if( ipi == null && opi != null ) { inszbeg = 1.0}

          if( ipde != null && opde != null && (ipde.length > opde.length) ) { deldecend = 1.0}

          if( ipde != null && opde != null && (ipde.length < opde.length) ) { insdecend = 1.0}

          if( ipde != null && opde == null ) { deldecend = 1.0}

          if( ipde == null && opde != null ) { insdecend = 1.0}

        }

        else {

          classify = "sm"

        }

        //classify = "int reformat"

      }

    }

    else {

      classify = "intorfloat derived"

    }

  }

  (classify,classifyip,classifyop,inszbeg,delzbeg,insdecend,deldecend)

}

 

val toIntorFloat = udf[(String,String,String,Double,Double,Double,Double),String,String]((x1,x2) => validateNumber(x1,x2))

 

val df_intorfloat = df_nonconcat.withColumn("IntorFloat",toIntorFloat(df_concat.col("InputString1"),df_concat.col("OutputString")))

 

val df_assignintfloat = df_intorfloat.withColumn("PredClassify",when($"IntorFloat._1" !== "def",df_intorfloat.col("IntorFloat._1")).otherwise(df_intorfloat.col("PredClassify"))).

                        withColumn("InszBegFlag",when($"IntorFloat._1" !== "def",df_intorfloat.col("IntorFloat._4")).otherwise(df_intorfloat.col("InszBegFlag"))).

                        withColumn("DelzBegFlag",when($"IntorFloat._1" !== "def",df_intorfloat.col("IntorFloat._5")).otherwise(df_intorfloat.col("DelzBegFlag"))).

                        withColumn("InsDecEndFlag",when($"IntorFloat._1" !== "def",df_intorfloat.col("IntorFloat._6")).otherwise(df_intorfloat.col("InsDecEndFlag"))).

                       withColumn("DelDecEndFlag",when($"IntorFloat._1" !== "def",df_intorfloat.col("IntorFloat._7")).otherwise(df_intorfloat.col("DelDecEndFlag"))).

                        drop(df_intorfloat("IntorFloat"))

 

val Dateyyyymmdd = """(?:(1[6-9]|[2-9]\d)?(\d\d))([-./])(?:(?:(0?[1|3|5|7|8]|10|12)\3(31))|(?:(0?[3-9]|1[0-2]|0?[1])\3(30))|(?:(0?[1-9]|1[012])\3(0?[1-9]|1[0-9]|2[0-9])))(?:(?=\x20\d)\x20|$)?(?:(?:((0?[0-9]|1[012])(?::(0?[0-5]?\d))(?::(0?[0-5]?\d))?(?:(?:\x20)([AP]M)|(?:\x20)([ap]m))?)|([01]\d|2[0-3])(?::(0?[0-5]?\d))(?::(0?[0-5]?\d))?)?)""".r

val Datemmddyyyy = """(?:(?:(0?[1|3|5|7|8]|10|12)([-./])(31))|(?:(0?[3-9]|1[0-2]|0?[1])([-./])(30))|(?:(0?[1-9]|1[012])([-./])(0?[1-9]|1[0-9]|2[0-9])))(\2|\5|\8)(?:(1[6-9]|[2-9]\d)?(\d\d))(?:(?=\x20\d)\x20|$)?(?:(?:(?:(0?[0-9]|1[012])(?::(0?[0-5]?\d))(?::(0?[0-5]?\d))?(?:(?:\x20)([AP]M)|(?:\x20)([ap]m))?)|([01]\d|2[0-3])(?::(0?[0-5]?\d))(?::(0?[0-5]?\d))?)?)""".r

val Dateddmmyyyy = """(?:((?:31(?!.(?:0?[2469]|11))|(?:30|29)(?!.0?2)|29(?=.0?2.(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00)))(?:\x20|$)))|(?:2[0-8]|1\d|0?[1-9]))([-./])(0?[1-9]|1[012])\2(1[6-9]|[2-9]\d)?(\d\d)(?:(?=\x20\d)\x20|$))?(?:(((0?[0-9]|1[012])(?::(0?[0-5]?\d))(?::(0?[0-5]?\d))?(?:(?:\x20)([AP]M)|(?:\x20)([ap]m))?)|([01]\d|2[0-3])(?::(0?[0-5]?\d))(?::(0?[0-5]?\d))?)?)""".r

val Dateyyyyddmm = """(?:(1[6-9]|[2-9]\d)?(\d\d))([-./])(?:(?:(31)\3(0?[1|3|5|7|8]|10|12))|(?:(30)\3(0?[3-9]|1[0-2]|0?[1]))|(?:(0?[1-9]|1[0-9]|2[0-9])\3(0?[1-9]|1[012])))(?:(?=\x20\d)\x20|$)?(?:(?:(?:(0?[0-9]|1[012])(?::(0?[0-5]?\d))(?::(0?[0-5]?\d))?(?:(?:\x20)([AP]M)|(?:\x20)([ap]m))?)|([01]\d|2[0-3])(?::(0?[0-5]?\d))(?::(0?[0-5]?\d))?)?)""".r

 

def validateAndCompareDate(ip:String,op:String) : (String,String,String,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double) = {

  var classifyip = "defip"

  var classifyop = "defop"

  var classify = "def"

  var ipyyyy : String = null

  var ipyy : String = null

  var ipdelim : String = null

  var ipmm : String = null

  var ipdd : String = null

  var iphr : String = null

  var ipmin : String = null

  var ipsec : String = null

  var iptz : String = null

  var opyyyy : String = null

  var opyy : String = null

  var opdelim : String = null

  var opmm : String = null

  var opdd : String = null

  var ophr : String = null

  var opmin : String = null

  var opsec : String = null

  var optz : String = null

  var yyyyChng = 0.0

  var yyChng = 0.0

  var mmChng = 0.0

  var ddChng = 0.0

  var hrChng = 0.0

  var minChng = 0.0

  var secChng = 0.0

  var tzChng = 0.0

  var yyyyFrmtChng = 0.0

  var yyFrmtChng = 0.0

  var mmFrmtChng = 0.0

  var ddFrmtChng = 0.0

  var hrFrmtChng = 0.0

  var minFrmtChng = 0.0

  var secFrmtChng = 0.0

  var tzFrmtChng = 0.0

  var delimFrmtChng = 0.0

  var dateFrmtChng = 0.0

  ip match {

    case Dateyyyymmdd(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18) => {

      classifyip = g1 + "," + g2 + "," + g3 + "," + g4 + "," + g5 + "," + g6 + "," + g7 + "," + g8 + "," + g9 + "," + g10 + "," + g11 + "," + g12 + "," + g13 + "," + g14 + "," + g15 + "," + g16 + "," + g17 + "," + g18

      if(g1 != null) {

        ipyyyy = g1 + g2

      }

      else {

        ipyy = g2

      }

      ipdelim = g3

      if(g4 != null) { ipmm = g4 }

      if(g6 != null) { ipmm = g6 }

      if(g8 != null) { ipmm = g8 }

      if(g5 != null) { ipdd = g5 }

      if(g7 != null) { ipdd = g7 }

      if(g9 != null) { ipdd = g9 }

      if(g11 != null) { iphr = g11 }

      else { iphr = g16 }

      if( g12 != null ) { ipmin = g12 }

      else { ipmin = g17 }

      if( g13 != null ) { ipsec = g13 }

      else { ipsec = g18 }

      if( g14 != null ) { iptz = g14 }

      else if( g15 != null ) { iptz = g15 }

    }

    case Datemmddyyyy(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20) => {

      classifyip = g1 + "," + g2 + "," + g3 + "," + g4 + "," + g5 + "," + g6 + "," + g7 + "," + g8 + "," + g9 + "," + g10 + "," + g11 + "," + g12 + "," + g13 + "," + g14 + "," + g15 + "," + g16 + "," + g17 + "," + g18 + "," + g19 + "," + g20

      if(g1 != null) { ipmm = g1 }

      if(g4 != null) { ipmm = g4 }

      if(g7 != null) { ipmm = g7 }

      if(g2 != null) { ipdelim = g2 }

      if(g5 != null) { ipdelim = g5 }

      if(g8 != null) { ipdelim = g8 }

      if(g3 != null) { ipdd = g3 }

      if(g6 != null) { ipdd = g6 }

      if(g9 != null) { ipdd = g9 }

      if(g11 != null) {

        ipyyyy = g11 + g12

      }

      else {

        ipyy = g12

      }

      if( g13 != null) { iphr = g13 }

      else { iphr = g18 }

      if( g14 != null ) { ipmin = g14 }

      else { ipmin = g19 }

      if( g15 != null ) { ipsec = g15 }

      else { ipsec = g20 }

      if( g16 != null ) { iptz = g16 }

      else if( g17 != null ) { iptz = g17 }

    }

    case Dateddmmyyyy(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15) => {

      classifyip = g1 + "," + g2 + "," + g3 + "," + g4 + "," + g5 + "," + g6 + "," + g7 + "," + g8 + "," + g9 + "," + g10 + "," + g11 + "," + g12 + "," + g13 + "," + g14 + "," + g15

      ipdd = g1

      ipdelim = g2

      ipmm = g3

      if(g4 != null) {

        ipyyyy = g4 + g5

      }

      else

        ipyy = g5

      if( g8 != null) { iphr = g8 }

      else { iphr = g13 }

      if( g9 != null ) { ipmin = g9 }

      else { ipmin = g14 }

      if( g10 != null ) { ipsec = g10 }

      else { ipsec = g15 }

      if( g11 != null ) { iptz = g11 }

      else if( g12 != null ) { iptz = g12 }

    }

    case Dateyyyyddmm(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17) => {

      classifyip = g1 + "," + g2 + "," + g3 + "," + g4 + "," + g5 + "," + g6 + "," + g7 + "," + g8 + "," + g9 + "," + g10 + "," + g11 + "," + g12 + "," + g13 + "," + g14 + "," + g15 + "," + g16 + "," + g17

      if(g1 != null) {

        ipyyyy = g1 + g2

      }

      else

        ipyy = g2

      ipdelim = g3

      if(g5 != null) { ipmm = g5 }

      if(g7 != null) { ipmm = g7 }

      if(g9 != null) { ipmm = g9 }

      if(g4 != null) { ipdd = g4 }

      if(g6 != null) { ipdd = g6 }

      if(g8 != null) { ipdd = g8 }

      if( g10 != null) { iphr = g10 }

      else { iphr = g15 }

      if( g11 != null ) { ipmin = g11 }

      else { ipmin = g16 }

      if( g12 != null ) { ipsec = g12 }

      else { ipsec = g17 }

      if( g13 != null ) { iptz = g13 }

      else if( g14 != null ) { iptz = g14 }

    }

    case _  => classifyip = "Input string not in date format"

  }

  op match {

    case Dateyyyymmdd(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18) => {

      classifyop = g1 + "," + g2 + "," + g3 + "," + g4 + "," + g5 + "," + g6 + "," + g7 + "," + g8 + "," + g9 + "," + g10 + "," + g11 + "," + g12 + "," + g13 + "," + g14 + "," + g15 + "," + g16 + "," + g17 + "," + g18

      if(g1 != null) {

        opyyyy = g1 + g2

      }

      else {

        opyy = g2

      }

      opdelim = g3

      if(g4 != null) { opmm = g4 }

      if(g6 != null) { opmm = g6 }

      if(g8 != null) { opmm = g8 }

      if(g5 != null) { opdd = g5 }

      if(g7 != null) { opdd = g7 }

      if(g9 != null) { opdd = g9 }

      if(g11 != null) { ophr = g11 }

      else { ophr = g16 }

      if( g12 != null ) { opmin = g12 }

      else { opmin = g17 }

      if( g13 != null ) { opsec = g13 }

      else { opsec = g18 }

      if( g14 != null ) { optz = g14 }

      else if( g15 != null ) { optz = g15 }

    }

    case Datemmddyyyy(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20) => {

      classifyop = g1 + "," + g2 + "," + g3 + "," + g4 + "," + g5 + "," + g6 + "," + g7 + "," + g8 + "," + g9 + "," + g10 + "," + g11 + "," + g12 + "," + g13 + "," + g14 + "," + g15 + "," + g16 + "," + g17 + "," + g18 + "," + g19 + "," + g20

      if(g1 != null) { opmm = g1 }

      if(g4 != null) { opmm = g4 }

      if(g7 != null) { opmm = g7 }

      if(g2 != null) { opdelim = g2 }

      if(g5 != null) { opdelim = g5 }

      if(g8 != null) { opdelim = g8 }

      if(g3 != null) { opdd = g3 }

      if(g6 != null) { opdd = g6 }

      if(g9 != null) { opdd = g9 }

      if(g11 != null) {

        opyyyy = g11 + g12

      }

      else {

        opyy = g12

      }

      if( g13 != null) { ophr = g13 }

      else { ophr = g18 }

      if( g14 != null ) { opmin = g14 }

      else { opmin = g19 }

      if( g15 != null ) { opsec = g15 }

      else { opsec = g20 }

      if( g16 != null ) { optz = g16 }

      else if( g17 != null ) { optz = g17 }

    }

    case Dateddmmyyyy(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15) => {

      classifyop = g1 + "," + g2 + "," + g3 + "," + g4 + "," + g5 + "," + g6 + "," + g7 + "," + g8 + "," + g9 + "," + g10 + "," + g11 + "," + g12 + "," + g13 + "," + g14 + "," + g15

      opdd = g1

      opdelim = g2

      opmm = g3

      if(g4 != null) {

        opyyyy = g4 + g5

      }

      else

        opyy = g5

      if( g8 != null) { ophr = g8 }

      else { ophr = g13 }

      if( g9 != null ) { opmin = g9 }

      else { opmin = g14 }

      if( g10 != null ) { opsec = g10 }

      else { opsec = g15 }

      if( g11 != null ) { optz = g11 }

      else if( g12 != null ) { optz = g12 }

    }

    case Dateyyyyddmm(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17) => {

      classifyop = g1 + "," + g2 + "," + g3 + "," + g4 + "," + g5 + "," + g6 + "," + g7 + "," + g8 + "," + g9 + "," + g10 + "," + g11 + "," + g12 + "," + g13 + "," + g14 + "," + g15 + "," + g16 + "," + g17

      if(g1 != null) {

        opyyyy = g1 + g2

      }

      else

        opyy = g2

      opdelim = g3

      if(g5 != null) { opmm = g5 }

      if(g7 != null) { opmm = g7 }

      if(g9 != null) { opmm = g9 }

      if(g4 != null) { opdd = g4 }

      if(g6 != null) { opdd = g6 }

      if(g8 != null) { opdd = g8 }

      if( g10 != null) { ophr = g10 }

      else { ophr = g15 }

      if( g11 != null ) { opmin = g11 }

      else { opmin = g16 }

      if( g12 != null ) { opsec = g12 }

      else { opsec = g17 }

      if( g13 != null ) { optz = g13 }

      else if( g14 != null ) { optz = g14 }

    }

    case _  => classifyop = "Output string not in date format"

  }

 

  if((classifyip != "Input string not in date format" && classifyop == "Output string not in date format")

      || (classifyip == "Input string not in date format" && classifyop != "Output string not in date format")) {

    classify = "datederived"

  }

  else if((classifyip != "Input string not in date format") && (classifyop != "Output string not in date format")) {

     if(ipyyyy != null && opyyyy != null) {

       if(toInt(ipyyyy) != toInt(opyyyy)) {

         yyyyChng = 1.0

         classify = "datederived"

       }

     }

     if((ipyyyy != null && opyyyy == null && opyy == null && ipyy == null) || (ipyyyy == null && opyyyy != null && ipyy == null && opyy == null)) {

       if(toInt(ipyyyy) != toInt(opyyyy)) {

         yyyyChng = 1.0

         classify = "datederived"

       }

       else {

         yyyyFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

       }

     }

     if(ipyy != null && opyy != null) {

       if(toInt(ipyy) != toInt(opyy)) {

         yyChng = 1.0

         classify = "datederived"

       }

     }

     if((ipyy != null && opyy == null && opyyyy == null && ipyyyy == null) || (ipyy == null && opyy != null && ipyyyy == null && opyyyy == null)) {

       if(toInt(ipyy) != toInt(opyy)) {

         yyChng = 1.0

         classify = "datederived"

       }

       else {

         yyFrmtChng = 1.0

         classify = "datereformat"

       }

     }

     if((ipyyyy != null && opyyyy == null && opyy == null && ipyy == null)) {

       if(ipyyyy.trim.substring(ipyyyy.length-2,ipyyyy.length) == opyy.trim) {

         yyyyFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

       }

       else {

         yyyyChng = 1.0

         classify = "datederived"

       }

     }

     if((ipyyyy != null && opyyyy == null && opyy != null && ipyy == null)) {

       if(ipyyyy.trim.substring(ipyyyy.length-2,ipyyyy.length) == opyy.trim) {

         yyyyFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

       }

       else {

         yyyyChng = 1.0

         classify = "datederived"

       }

     }

     if((ipyyyy == null && opyyyy != null) && (ipyy == null && opyy == null)) {

      if(opyyyy.trim.substring(opyyyy.length-2,opyyyy.length) == ipyy.trim) {

         yyFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

       }

       else {

         yyChng = 1.0

         classify = "datederived"

       }

     }

     if((ipyyyy == null && opyyyy != null) && (ipyy != null && opyy == null)) {

       if(opyyyy.trim.substring(opyyyy.length-2,opyyyy.length) == ipyy.trim) {

         yyFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

       }

       else {

         yyChng = 1.0

         classify = "datederived"

       }

     }

     if(toInt(iphr) == toInt(ophr) && ((iphr != null && ophr == null) || (iphr == null && ophr != null))) {

         hrFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

     }

     if(toInt(iphr) == toInt(ophr) && (iphr != null && ophr != null) && (iphr.trim.length != ophr.trim.length)) {

         hrFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

     }

     if(toInt(iphr) != toInt(ophr) && iphr != null && ophr != null) {

       if( (toInt(ophr) - toInt(iphr)) == 12 || (toInt(iphr) - toInt(ophr)) == 12) {

         hrFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

       }

       else {

         hrChng = 1.0

         classify = "datederived"

       }

     }

     if((toInt(ipmm) == toInt(opmm)) && ((ipmm != null && opmm == null) || (ipmm == null && opmm != null))) {

       mmFrmtChng = 1.0

       if(classify == "def") {

         classify = "datereformat"

       }

     }

     if((toInt(ipmm) == toInt(opmm)) && ipmm != null && opmm != null && (ipmm.trim.length != opmm.trim.length)) {

       mmFrmtChng = 1.0

       if(classify == "def") {

         classify = "datereformat"

       }

     }

     if((toInt(ipmm) != toInt(opmm)) && ipmm != null && opmm != null) {

       mmChng = 1.0

       classify = "datederived"

     }

    if((toInt(ipdd) == toInt(opdd)) && ((ipdd != null && opdd == null) || (ipdd == null && opdd != null))) {

       ddFrmtChng = 1.0

       if(classify == "def") {

         classify = "datereformat"

       }

     }

     if((toInt(ipdd) == toInt(opdd)) && ipdd != null && opdd != null && (ipdd.trim.length != opdd.trim.length)) {

         ddFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

     }

     if((toInt(ipdd) != toInt(opdd)) && ipdd != null && opdd != null) {

       ddChng = 1.0

       classify = "datederived"

     }

     if((toInt(ipmin) == toInt(opmin)) && ((ipmin != null && opmin == null) || (ipmin == null && opmin != null))) {

       minFrmtChng = 1.0

       if(classify == "def") {

         classify = "datereformat"

       }

     }

     if((toInt(ipmin) == toInt(opmin)) && ipmin != null && opmin != null && (ipmin.trim.length != opmin.trim.length)) {

       minFrmtChng = 1.0

       if(classify == "def") {

         classify = "datereformat"

       }

     }

     if((toInt(ipmin) != toInt(opmin)) && ipmin != null && opmin != null) {

       minChng = 1.0

      classify = "datederived"

     }

     if((toInt(ipsec) == toInt(opsec)) && ((ipsec != null && opsec == null) || (ipsec == null && opsec != null))) {

       secFrmtChng = 1.0

       if(classify == "def") {

         classify = "datereformat"

       }

     }

     if((toInt(ipsec) == toInt(opsec)) && ipsec != null && opsec != null && (ipsec.trim.length != opsec.trim.length)) {

       secFrmtChng = 1.0

       if(classify == "def") {

         classify = "datereformat"

       }

     }

     if((toInt(ipsec) != toInt(opsec)) && ipsec != null && opsec != null) {

       secChng = 1.0

       classify = "datederived"

     }

     if(ipdelim != opdelim) {

       delimFrmtChng = 1.0

       if(classify == "def") {

         classify = "datereformat"

       }

     }

     if((ipyyyy == opyyyy) && (ipmm == opmm) && (ipdd == opdd) && (ipdelim == opdelim) && (ipyy == opyy) && (iphr == ophr) && (ipmin == opmin) && (ipsec == opsec)) {

       if(ip.trim == op.trim) {

         classify = "sm"

       }

       else {

         dateFrmtChng = 1.0

         if(classify == "def") {

           classify = "datereformat"

         }

       }

     }

  }

  (classify,classifyip,classifyop,yyyyChng,yyChng,mmChng,ddChng,hrChng,minChng,secChng,yyyyFrmtChng,yyFrmtChng,mmFrmtChng,ddFrmtChng,delimFrmtChng,hrFrmtChng,minFrmtChng,secFrmtChng,dateFrmtChng)

}

 

val toDate = udf[(String,String,String,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double,Double),String,String]((x1,x2) => validateAndCompareDate(x1,x2))

val df_date = df_assignintfloat.withColumn("DateOP",when($"InputString1" !== $"OutputString",toDate(df_assignintfloat.col("InputString1"),df_assignintfloat.col("OutputString"))))

val df_assigndate = df_date.withColumn("PredClassify",when(($"PredClassify" === "not sm") && ($"DateOP._1" !== "def"),df_date.col("DateOP._1")).otherwise(df_date.col("PredClassify"))).

                            withColumn("YYYYChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._4")).otherwise(df_date.col("YYYYChngFlag"))).

                            withColumn("YYChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._5")).otherwise(df_date.col("YYChngFlag"))).

                            withColumn("MMChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._6")).otherwise(df_date.col("MMChngFlag"))).

                            withColumn("DDChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._7")).otherwise(df_date.col("DDChngFlag"))).

                            withColumn("HrChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._8")).otherwise(df_date.col("HrChngFlag"))).

                            withColumn("MinChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._9")).otherwise(df_date.col("MinChngFlag"))).

                            withColumn("SecChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._10")).otherwise(df_date.col("SecChngFlag"))).

                            withColumn("YYYYFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._11")).otherwise(df_date.col("YYYYFrmtChngFlag"))).

                            withColumn("YYFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._12")).otherwise(df_date.col("YYFrmtChngFlag"))).

                            withColumn("MMFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._13")).otherwise(df_date.col("MMFrmtChngFlag"))).

                            withColumn("DDFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._14")).otherwise(df_date.col("DDFrmtChngFlag"))).

                            withColumn("DelimFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._15")).otherwise(df_date.col("DelimFrmtChngFlag"))).

                            withColumn("HrFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._16")).otherwise(df_date.col("HrFrmtChngFlag"))).

                            withColumn("MinFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._17")).otherwise(df_date.col("MinFrmtChngFlag"))).

                            withColumn("SecFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._18")).otherwise(df_date.col("SecFrmtChngFlag"))).

                            withColumn("DateFrmtChngFlag",when(($"DateOP._1" !== "def"),df_date.col("DateOP._19")).otherwise(df_date.col("DateFrmtChngFlag"))).

                            drop(df_date.col("DateOP"))

 

//String reformat logic

 

def validateString(ip:String, op:String) : (String,Double,Double,Double,Double) = {

  var classify = ""

  var insspbeg = 0.0

  var delspbeg = 0.0

  var insspend = 0.0

  var delspend = 0.0

  if( ip.trim == op.trim) {

    classify = "stringreformat"

    var ltrimip = ip.replaceAll("""^\s+""","")

    var rtrimip = ip.replaceAll("""\s+$""","")

    var ltrimop = op.replaceAll("""^\s+""","")

    var rtrimop = op.replaceAll("""\s+$""","")

    var ltrimiplen = ip.length - ltrimip.length

    var rtrimiplen = ip.length - rtrimip.length

    var ltrimoplen = op.length - ltrimop.length

    var rtrimoplen = op.length - rtrimop.length

    if(ltrimiplen < ltrimoplen) {

      insspbeg = 1.0

    }

    if(ltrimiplen > ltrimoplen) {

      delspbeg = 1.0

    }

    if(rtrimiplen < rtrimoplen) {

      insspend = 1.0

    }

    if(rtrimiplen > rtrimoplen) {

      delspend = 1.0

    }

  }

  else {

    classify = "stringderived"

  }

  (classify,insspbeg,delspbeg,insspend,delspend)

}

 

val stringReformat = udf[(String,Double,Double,Double,Double),String,String]((x1,x2) => validateString(x1,x2))

val df_string = df_assigndate.withColumn("StringOP",when($"PredClassify" === "not sm",stringReformat(df_assigndate.col("InputString1"),df_assigndate.col("OutputString"))))

 

val df_assignstring = df_string.withColumn("InsSpBegFlag",when($"PredClassify" === "not sm",df_string.col("StringOP._2")).otherwise(df_string.col("InsSpBegFlag"))).

                                withColumn("DelSpBegFlag",when($"PredClassify" === "not sm",df_string.col("StringOP._3")).otherwise(df_string.col("DelSpBegFlag"))).

                                withColumn("InsSpEndFlag",when($"PredClassify" === "not sm",df_string.col("StringOP._4")).otherwise(df_string.col("InsSpEndFlag"))).

                                withColumn("DelSpEndFlag",when($"PredClassify" === "not sm",df_string.col("StringOP._5")).otherwise(df_string.col("DelSpEndFlag"))).

                                withColumn("PredClassify",when($"PredClassify" === "not sm",df_string.col("StringOP._1")).otherwise(df_string.col("PredClassify"))).

                                drop(df_string.col("StringOP"))

 

def findMasking(ip: String, op:String) : Double = {

  var ipList : List[Char] = ip.toList

  var opList : List[Char] = op.toList

  var insList : List[Char] = opList.diff(ipList)

  var sameFlag : Double = 0.0

  var tempList : List[Char] = List()

  var prevChar : Char = ' '

  var currChar : Char = ' '

  var keymap = scala.collection.mutable.HashMap.empty[Char,List[Int]]

  breakable {

  for(i <- 0 to insList.length-1) {

    if(i == 0) {

      prevChar = insList(i)

    }

    else {

      currChar = insList(i)

      if((currChar == prevChar) && (currChar == 'X' || currChar == 'Y' || currChar == 'Z')) {

        sameFlag = 1.0

      }

      else {

        sameFlag = 0.0

        break

      }

    }

  }

  }

  return sameFlag

}

 

val toMask = udf[(Double),String,String]((x1,x2) => findMasking(x1,x2))

val df_mask = df_assignstring.withColumn("InsSameFlag",when($"PredClassify" === "stringderived",toMask(df_assignstring.col("InputString1"),df_assignstring.col("OutputString"))).otherwise(0.0)).

                              withColumn("PredClassify",when($"InsSameFlag" === 1.0,"masking").otherwise(df_assignstring.col("PredClassify")))

val df_inter1 = df_mask.withColumn("PredClassify",when(($"PredClassify" === "stringderived") || ($"PredClassify" === "int derived") || ($"PredClassify" === "intorfloat derived") || ($"PredClassify" === "datederived"),"derived").otherwise(df_mask.col("PredClassify")))

val df_inter2 = df_inter1.withColumn("PredClassify",when(($"PredClassify" === "float reformat") || ($"PredClassify" === "int reformat"),"numeric reformat").otherwise(df_inter1.col("PredClassify")))

//df_inter2.rdd.coalesce(1,true).saveAsTextFile("Features.csv")

 

//val final_rdd=df_inter2.map(row=> row(0)+","+row(1)+","+row(2)+","+row(3)+","+row(4)+","+row(5)+","+row(6)+","+row(7)+","+row(8)+","+row(9)+","+row(10)+

//                           ","+row(11)+","+row(12)+","+row(13)+","+row(14)+","+row(15)+","+row(16)+","+row(17)+","+row(18)+","+row(19)+","+row(20)+

//                           ","+row(21)+","+row(22)+","+row(23)+","+row(24)+","+row(25)+","+row(26)+","+row(27)+","+row(28)+","+row(29)+","+row(30)+

//                           ","+row(31)+","+row(32)+","+row(33)+","+row(34)+","+row(35)+","+row(36)+","+row(37)+","+row(38)+","+row(39)+","+row(40)+

//                           ","+row(41)+","+row(42)+","+row(43)+","+row(44)+","+row(45)+","+row(46))

//final_rdd.coalesce(1,true).saveAsTextFile("Features.csv")

 

val indexer = new StringIndexer().setInputCol("PredClassify").setOutputCol("ClassifyIndex").fit(df_inter2)

val indexed = indexer.transform(df_inter2)

val coder = new OneHotEncoder().setInputCol("ClassifyIndex").setOutputCol("ClassifyVec")

val coded = coder.transform(indexed)

val InputCols = coded.select($"ClassifyIndex",$"DelSpBegFlag",$"DelSpBetFlag",$"DelSpEndFlag",

                             $"DelzBegFlag",$"DelzBetFlag",$"DelzEndFlag",

                             $"DelBegFlag",$"DelBetFlag",$"DelEndFlag",

                             $"InsSpBegFlag",$"InsSpBetFlag",$"InsSpEndFlag",

                             $"InsFlag",$"InsSameFlag",$"InszBegFlag",

                             $"InsDecEndFlag",$"DelDecEndFlag",$"MovFlag",

                             $"ConcatFlag",$"YYYYChngFlag",$"YYChngFlag",

                             $"MMChngFlag",$"DDChngFlag",$"TSChngFlag",

                             $"HrChngFlag",$"MinChngFlag",$"SecChngFlag",

                             $"TZChngFlag",$"YYYYFrmtChngFlag",$"YYFrmtChngFlag",

                             $"MMFrmtChngFlag",$"DDFrmtChngFlag",$"DelimFrmtChngFlag",

                             $"TSFrmtChngFlag",$"HrFrmtChngFlag",$"MinFrmtChngFlag",

                             $"SecFrmtChngFlag",$"TZFrmtChngFlag",$"DateFrmtChngFlag")

val Array(trainingKM,testKM) = InputCols.randomSplit(Array(0.7, 0.3), seed = 11L)

val vectors = trainingKM.rdd.map(r => Vectors.dense(r.getDouble(1), r.getDouble(2),r.getDouble(3), r.getDouble(4),

                                                    r.getDouble(5), r.getDouble(6),r.getDouble(7), r.getDouble(8),

                                                    r.getDouble(9), r.getDouble(10),r.getDouble(11), r.getDouble(12),

                                                    r.getDouble(13), r.getDouble(14),r.getDouble(15), r.getDouble(16),

                                                    r.getDouble(17), r.getDouble(18),r.getDouble(19), r.getDouble(20),

                                                    r.getDouble(21), r.getDouble(22),r.getDouble(23), r.getDouble(24),

                                                    r.getDouble(25), r.getDouble(26),r.getDouble(27), r.getDouble(28),

                                                    r.getDouble(29), r.getDouble(30),r.getDouble(31), r.getDouble(32),

                                                    r.getDouble(33), r.getDouble(34),r.getDouble(35), r.getDouble(36),

                                                    r.getDouble(37), r.getDouble(38),r.getDouble(39)))

vectors.cache

val kMeansModel = KMeans.train(vectors, 7, 20)

kMeansModel.clusterCenters.foreach(println)

//Model on training data

val predictionsKMTrain = trainingKM.rdd.map{r => (r.getDouble(0), kMeansModel.predict(Vectors.dense(r.getDouble(1), r.getDouble(2),r.getDouble(3), r.getDouble(4),

                                                    r.getDouble(5), r.getDouble(6),r.getDouble(7), r.getDouble(8),

                                                    r.getDouble(9), r.getDouble(10),r.getDouble(11), r.getDouble(12),

                                                    r.getDouble(13), r.getDouble(14),r.getDouble(15), r.getDouble(16),

                                                    r.getDouble(17), r.getDouble(18),r.getDouble(19), r.getDouble(20),

                                                    r.getDouble(21), r.getDouble(22),r.getDouble(23), r.getDouble(24),

                                                    r.getDouble(25), r.getDouble(26),r.getDouble(27), r.getDouble(28),

                                                    r.getDouble(29), r.getDouble(30),r.getDouble(31), r.getDouble(32),

                                                    r.getDouble(33), r.getDouble(34),r.getDouble(35), r.getDouble(36),

                                                    r.getDouble(37), r.getDouble(38),r.getDouble(39) ) ) ) }

 

val predictionAndLabelsKMTrain = predictionsKMTrain.map { x => (x._1.toDouble,x._2.toDouble) }

val metricsKMTrain = new MulticlassMetrics(predictionAndLabelsKMTrain)

println("Confusion matrix:")

println(metricsKMTrain.confusionMatrix)

4018.0  0.0     0.0     0.0     0.0     0.0    0.0

0.0     4089.0  0.0     0.0     0.0     0.0    0.0

0.0     0.0     0.0     1563.0  0.0     0.0    0.0

0.0     0.0     0.0     0.0     1552.0  0.0    0.0

0.0     0.0     4062.0  2.0     16.0    267.0  8.0

0.0     0.0     0.0     0.0     0.0     438.0  0.0

0.0     0.0     0.0     192.0   0.0     0.0    0.0

val labelsKMTrain = metricsKMTrain.labels

labelsKMTrain.foreach { l =>

  println(s"Precision($l) = " + metricsKMTrain.precision(l))

}

Precision(0.0) = 1.0

Precision(1.0) = 1.0

Precision(2.0) = 0.0

Precision(3.0) = 0.0

Precision(4.0) = 0.01020408163265306

Precision(5.0) = 0.6212765957446809

Precision(6.0) = 0.0

// Recall by label

labelsKMTrain.foreach { l =>

  println(s"Recall($l) = " + metricsKMTrain.recall(l))

}

Recall(0.0) = 1.0

Recall(1.0) = 1.0

Recall(2.0) = 0.0

Recall(3.0) = 0.0

Recall(4.0) = 0.0036739380022962113

Recall(5.0) = 1.0

Recall(6.0) = 0.0

// False positive rate by label

labelsKMTrain.foreach { l =>

  println(s"FPR($l) = " + metricsKMTrain.falsePositiveRate(l))

}

FPR(0.0) = 0.0

FPR(1.0) = 0.0

FPR(2.0) = 0.27738322862605846

FPR(3.0) = 0.11989082224496758

FPR(4.0) = 0.13094836314546068

FPR(5.0) = 0.016931955101781977

FPR(6.0) = 4.995316890415236E-4

// F-measure by label

labelsKMTrain.foreach { l =>

  println(s"F1-Score($l) = " + metricsKMTrain.fMeasure(l))

}

F1-Score(0.0) = 1.0

F1-Score(1.0) = 1.0

F1-Score(2.0) = 0.0

F1-Score(3.0) = 0.0

F1-Score(4.0) = 0.005402667567111261

F1-Score(5.0) = 0.7664041994750657

F1-Score(6.0) = 0.0

// Weighted stats

println(s"Weighted precision: ${metricsKMTrain.weightedPrecision}")

Weighted precision: 0.5197481288607624

println(s"Weighted recall: ${metricsKMTrain.weightedRecall}")

Weighted recall: 0.5282285432220645

println(s"Weighted F1 score: ${metricsKMTrain.weightedFMeasure}")

Weighted F1 score: 0.5223800614934812

println(s"Weighted false positive rate: ${metricsKMTrain.weightedFalsePositiveRate}")

Weighted false positive rate: 0.07388244402937477

 

//Test data prediction

val predictionsKMTest = testKM.rdd.map{r => (r.getDouble(0), kMeansModel.predict(Vectors.dense(r.getDouble(1), r.getDouble(2),r.getDouble(3), r.getDouble(4),

                                                    r.getDouble(5), r.getDouble(6),r.getDouble(7), r.getDouble(8),

                                                    r.getDouble(9), r.getDouble(10),r.getDouble(11), r.getDouble(12),

                                                    r.getDouble(13), r.getDouble(14),r.getDouble(15), r.getDouble(16),

                                                    r.getDouble(17), r.getDouble(18),r.getDouble(19), r.getDouble(20),

                                                    r.getDouble(21), r.getDouble(22),r.getDouble(23), r.getDouble(24),

                                                    r.getDouble(25), r.getDouble(26),r.getDouble(27), r.getDouble(28),

                                                    r.getDouble(29), r.getDouble(30),r.getDouble(31), r.getDouble(32),

                                                    r.getDouble(33), r.getDouble(34),r.getDouble(35), r.getDouble(36),

                                                    r.getDouble(37), r.getDouble(38),r.getDouble(39) ) ) ) }

//Print the center of each cluster

 

val predictionAndLabelsKMTest = predictionsKMTest.map { x => (x._1.toDouble,x._2.toDouble) }

val metricsKMTest = new MulticlassMetrics(predictionAndLabelsKMTest)

println("Confusion matrix:")

println(metricsKMTest.confusionMatrix)

1712.0  0.0     0.0     0.0    0.0    0.0    0.0

0.0     1641.0  0.0     0.0    0.0    0.0    0.0

0.0     0.0     0.0     644.0  0.0    0.0    0.0

0.0     0.0     0.0     0.0    681.0  0.0    0.0

0.0     0.0     1667.0  0.0    6.0    110.0  2.0

0.0     0.0     0.0     0.0    0.0    187.0  0.0

0.0     0.0     0.0     72.0   0.0    0.0    0.0

val labelsKMTest = metricsKMTest.labels

labelsKMTest.foreach { l =>

  println(s"Precision($l) = " + metricsKMTest.precision(l))

}

Precision(0.0) = 1.0

Precision(1.0) = 1.0

Precision(2.0) = 0.0

Precision(3.0) = 0.0

Precision(4.0) = 0.008733624454148471

Precision(5.0) = 0.6296296296296297

Precision(6.0) = 0.0

// Recall by label

labelsKMTest.foreach { l =>

  println(s"Recall($l) = " + metricsKMTest.recall(l))

}

 

// False positive rate by label

labelsKMTest.foreach { l =>

  println(s"FPR($l) = " + metricsKMTest.falsePositiveRate(l))

}

 

// F-measure by label

labelsKMTest.foreach { l =>

  println(s"F1-Score($l) = " + metricsKMTest.fMeasure(l))

}

 

// Weighted stats

println(s"Weighted precision: ${metricsKMTest.weightedPrecision}")

println(s"Weighted recall: ${metricsKMTest.weightedRecall}")

println(s"Weighted F1 score: ${metricsKMTest.weightedFMeasure}")

println(s"Weighted false positive rate: ${metricsKMTest.weightedFalsePositiveRate}")