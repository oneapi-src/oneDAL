/* file: SampleSVD.scala */
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

import daal_for_mllib.SVD

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.storage.StorageLevel

import java.io._

object SampleSVD extends App {
    val conf = new SparkConf().setAppName("Spark SVD")
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Spark/SVD/data/SVD.txt")
    val dataRDD = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).persist(StorageLevel.MEMORY_AND_DISK)

    val nRows = data.count()
    val nCols = dataRDD.first.size

    val rowMatrix = new RowMatrix(dataRDD, nRows, nCols)
    rowMatrix.rows.persist(StorageLevel.MEMORY_AND_DISK)

    val result = SVD.computeSVD(rowMatrix)

    val U: RowMatrix = result.U
    val s: Vector = result.s
    val V: Matrix = result.V

    sc.stop()
}
