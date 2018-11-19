/* file: SamplePCA.scala */
//==============================================================================
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,  and
// your use of  them is governed  by the express  license under which  they were
// provided to you (License).  Unless  the License  provides otherwise,  you may
// not use,  modify,  copy,  publish,  distribute,   disclose  or transmit  this
// software or the related documents without Intel's prior written permission.
//
// This software and the related documents are provided  as is,  with no express
// or implied  warranties,  other than  those that are  expressly stated  in the
// License.
//
// License:
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-ag
// reement/
//==============================================================================

package DAAL

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

//import org.apache.spark.mllib.feature.{PCA, PCAModel}
import daal_for_mllib.{PCA, PCAModel}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import java.io._

object SamplePCA extends App {
    val conf = new SparkConf().setAppName("Spark PCA")
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Spark/PCA/data/PCA.txt")
    val dataRDD = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val model = new PCA(10).fit(dataRDD)

    println("Eigen vectors:")
    println(model.pc.toString())
    println("")
    println("Eigen values:")
    println(model.explainedVariance.toString())

    sc.stop()
}
