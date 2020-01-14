/* file: SVD.scala */
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

package daal_for_mllib

import java.nio.DoubleBuffer

import org.apache.spark.SparkContext
import org.apache.spark.api.java._
import org.apache.spark.api.java.function._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.Tuple2
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.SingularValueDecomposition

import com.intel.daal.algorithms.svd._
import com.intel.daal.data_management.data._
import com.intel.daal.services._

object SVD {

    def computeSVD(matrix: RowMatrix) : SingularValueDecomposition[RowMatrix, Matrix] = {
        val context = new DaalContext()

        val dataRDD = getAsPairRDD(matrix)

        val resultsFromStep1 = computeStep1Local(dataRDD)
        dataRDD.unpersist()

        val resultsFromStep2 = computeStep2Master(context, resultsFromStep1._1)

        val resultsFromStep3 = computeStep3Local(resultsFromStep1._2, resultsFromStep2._2)

        val U = resultsFromStep3.mapPartitions(
            (it: Iterator[HomogenNumericTable]) => {
                val res = new ArrayBuffer[Vector]()
                while (it.hasNext) {
                    val collection = it.next
                    val contextLocal = new DaalContext()
                    collection.unpack(contextLocal)
                    val nRows = collection.getNumberOfRows().toInt
                    val nCols = collection.getNumberOfColumns().toInt
                    val i: Int = 0
                    for (i <- 0 to (nRows - 1)) {
                        val arr = new Array[Double](nCols)
                        var buffer = DoubleBuffer.allocate(nCols)
                        buffer = collection.getBlockOfRows(i, 1, buffer)
                        buffer.get(arr)
                        res += Vectors.dense(arr)
                    }
                    collection.pack()
                    contextLocal.dispose()
                }

                res.iterator
            })

        val ntS = resultsFromStep2._1._1
        val ntV = resultsFromStep2._1._2

        val nValues = ntS.getNumberOfColumns().toInt

        var s = new Array[Double](nValues)
        var V = new Array[Double](nValues * nValues)

        var sBuffer = DoubleBuffer.allocate(nValues)
        var vBuffer = DoubleBuffer.allocate(nValues * nValues)

        sBuffer = ntS.getBlockOfRows(0, 1, sBuffer)
        vBuffer = ntV.getBlockOfRows(0, nValues, vBuffer)

        sBuffer.get(s)
        vBuffer.get(V)

        context.dispose()

        new SingularValueDecomposition[RowMatrix, Matrix](new RowMatrix(U, matrix.numRows, matrix.numCols.toInt),
                                                          Vectors.dense(s),
                                                          Matrices.dense(nValues, nValues, V))
    }


    private def getAsPairRDD(matrix: RowMatrix): RDD[HomogenNumericTable] = {


        val tablesRDD = matrix.rows.mapPartitions(
            (it: Iterator[Vector]) => {

                val tables = new ArrayBuffer[HomogenNumericTable]()
                var arrays = new ArrayBuffer[Array[Double]]()
                it.foreach{curVector =>
                    val rowData = curVector.toArray
                    arrays += rowData
                }
                val numCols: Int = arrays(0).length
                val numRows: Int = arrays.length

                val arrData = new Array[Double](numRows * numCols)
                var i: Int = 0
                for (i <- 0 to (numRows - 1)) {
                    arrays(i).copyToArray(arrData, i * numCols)
                }
                val context = new DaalContext()
                val table = new HomogenNumericTable(context, arrData, numCols, numRows)
                table.pack()
                tables += table

                context.dispose()
                arrays.clear
                tables.iterator
            }
        ).persist(StorageLevel.MEMORY_AND_DISK)
        tablesRDD
    }

    private def computeStep1Local(dataRDD: RDD[HomogenNumericTable]): Tuple2[RDD[Tuple2[Long, DataCollection]], RDD[Tuple2[Long, DataCollection]]] = {

        /* Creating RDD containing both partial results for step2 and step3 */
        val dataFromStep1RDD = dataRDD.mapPartitions(
            (it: Iterator[HomogenNumericTable]) => {

                val table = it.next
                val contextLocal = new DaalContext()

                /* Create algorithm to compute SVD decomposition on local node */
                val svdStep1Local = new DistributedStep1Local(contextLocal, classOf[java.lang.Double], Method.defaultDense)
                table.unpack(contextLocal)
                svdStep1Local.input.set( InputId.data, table )

                /* Compute SVD Step 1  */
                val pres = svdStep1Local.compute()
                val dataFromStep1ForStep2 = pres.get( PartialResultId.outputOfStep1ForStep2 )
                dataFromStep1ForStep2.pack()
                val dataFromStep1ForStep3 = pres.get( PartialResultId.outputOfStep1ForStep3 )
                dataFromStep1ForStep3.pack()

                contextLocal.dispose()
                Iterator(Tuple2(dataFromStep1ForStep2, dataFromStep1ForStep3))

            }).cache()

        /* Extracting partial results for step2 */
        val dataFromStep1ForStep2RDD = dataFromStep1RDD.mapPartitionsWithIndex(
            (index, it: Iterator[Tuple2[DataCollection, DataCollection]]) => {
                val tup = it.next()
                Iterator(Tuple2(index.asInstanceOf[Long], tup._1))
            })

        /* Extracting partial results for step3 */
        val dataFromStep1ForStep3RDD = dataFromStep1RDD.mapPartitionsWithIndex(
            (index, it: Iterator[Tuple2[DataCollection, DataCollection]]) => {
                val tup = it.next()
                Iterator(Tuple2(index.asInstanceOf[Long], tup._2))
            })

        new Tuple2(dataFromStep1ForStep2RDD, dataFromStep1ForStep3RDD)
    }

    private def computeStep2Master(context: DaalContext, dataFromStep1ForStep2RDD: RDD[Tuple2[Long, DataCollection]])
        : Tuple2[Tuple2[HomogenNumericTable, HomogenNumericTable], RDD[Tuple2[Long, DataCollection]]] = {

        val svdStep2Master = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense)

        val dataFromStep1ForStep2List = dataFromStep1ForStep2RDD.collect()
        val nBlocks = dataFromStep1ForStep2List.size

        val i: Int = 0
        for (i <- 0 to (nBlocks - 1)) {
            val value = dataFromStep1ForStep2List(i)
            value._2.unpack(context)
            svdStep2Master.input.add(DistributedStep2MasterInputId.inputOfStep2FromStep1, value._1.toInt, value._2)
        }

        val pres = svdStep2Master.compute()

        val inputForStep3FromStep2 = pres.get(DistributedPartialResultCollectionId.outputOfStep2ForStep3)

        val list = new Array[Tuple2[Long, DataCollection]](nBlocks)
        for (i <- 0 to (nBlocks - 1)) {
            val value = dataFromStep1ForStep2List(i)
            val dc = inputForStep3FromStep2.get(value._1.toInt).asInstanceOf[DataCollection]
            dc.pack()
            list(i) = new Tuple2[Long, DataCollection](value._1, dc)
        }

        val sc = dataFromStep1ForStep2RDD.sparkContext

        val dataFromStep2ForStep3RDD = sc.parallelize(list, nBlocks)

        val res = svdStep2Master.finalizeCompute()

        val ntS = res.get(ResultId.singularValues).asInstanceOf[HomogenNumericTable]
        val ntV = res.get(ResultId.rightSingularMatrix).asInstanceOf[HomogenNumericTable]

        new Tuple2(new Tuple2(ntS, ntV), dataFromStep2ForStep3RDD)
    }

    private def computeStep3Local(dataFromStep1ForStep3RDD: RDD[Tuple2[Long, DataCollection]], dataFromStep2ForStep3RDD: RDD[Tuple2[Long, DataCollection]])
        : RDD[HomogenNumericTable] = {

        val dataForStep3RDD = dataFromStep1ForStep3RDD.cogroup(dataFromStep2ForStep3RDD)

        val ntURDD = dataForStep3RDD.mapPartitions(
            (it: Iterator[Tuple2[Long, Tuple2[Iterable[DataCollection], Iterable[DataCollection]]]]) => {
                val tup = it.next()

                val contextLocal = new DaalContext()

                val ntQPi = tup._2._1.iterator.next()
                ntQPi.unpack(contextLocal)
                val ntPi  = tup._2._2.iterator.next()
                ntPi.unpack(contextLocal)

                /* Create algorithm to compute SVD decomposition on master node */
                val svdStep3Local = new DistributedStep3Local(contextLocal, classOf[java.lang.Double], Method.defaultDense)
                svdStep3Local.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep1, ntQPi)
                svdStep3Local.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep2, ntPi)

                /* Compute SVD Step 3 */
                svdStep3Local.compute()
                val result = svdStep3Local.finalizeCompute()

                val Ui = result.get(ResultId.leftSingularMatrix).asInstanceOf[HomogenNumericTable]
                Ui.pack()

                contextLocal.dispose()

                Iterator(Ui)
            })

        ntURDD
    }
}