/* file: SampleCorrelation.scala */
//==============================================================================
// Copyright 2017-2022 Intel Corporation
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

//import org.apache.spark.mllib.stat.Statistics
import daal_for_mllib.Statistics

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import java.io._

object SampleCorrelation extends App {
    val conf = new SparkConf().setAppName("Spark Correlation")
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Spark/Correlation/data/Correlation.txt")
    val dataRDD = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val correlMatrix: Matrix = Statistics.corr(dataRDD)

    println(correlMatrix.toString)

    sc.stop()
}
