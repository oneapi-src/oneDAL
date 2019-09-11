/* file: SampleLinearRegression.scala */
//==============================================================================
// Copyright 2017-2019 Intel Corporation.
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
import org.apache.spark.rdd.RDD

import daal_for_mllib.LinearRegression

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel

import com.intel.daal.algorithms.linear_regression.training.TrainingMethod

object SampleLinearRegression extends App {

    def calculateMSE(model: LinearRegressionModel, data: RDD[LabeledPoint]) : Double = {  
        val valuesAndPreds = data.map( point => {
             val prediction = model.predict(point.features)
             (point.label, prediction)
           }
        )
        valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    }

    val conf = new SparkConf().setAppName("Spark Linear Regression")
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Spark/LinearRegression/data/LinearRegression.txt")
    val parsedData = data.map { line => {
        val parts = line.split(',')
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }}.cache()

    val model = LinearRegression.train(parsedData, TrainingMethod.normEqDense)

    val MSE = calculateMSE(model, parsedData)
    println("Mean Squared Error = " + MSE)

    sc.stop()
}
