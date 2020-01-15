/* file: SampleKMeans.scala */
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

//import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import daal_for_mllib.{KMeans, DAALKMeansModel => KMeansModel}
import org.apache.spark.storage.StorageLevel

import org.apache.spark.mllib.linalg.Vectors

object SampleKMeans extends App {
    val conf = new SparkConf().setAppName("Spark KMeans")
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Spark/KMeans/data/KMeans.txt")
    val dataRDD = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).persist(StorageLevel.MEMORY_AND_DISK)

    val nClusters = 20
    val nIterations = 10
    val clusters = KMeans.train(dataRDD, nClusters, nIterations, 1, "random")

    val cost = clusters.computeCost(dataRDD)
    println("Sum of squared errors = " + cost)

    sc.stop()
}
