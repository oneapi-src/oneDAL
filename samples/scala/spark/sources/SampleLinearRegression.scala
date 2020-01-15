/* file: SampleLinearRegression.scala */
//==============================================================================
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
