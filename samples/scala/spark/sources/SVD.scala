/* file: SVD.scala */
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

package daal_for_mllib

import java.nio.DoubleBuffer

import org.apache.spark.SparkContext
import org.apache.spark.api.java._
import org.apache.spark.api.java.function._
import org.apache.spark.rdd.RDD

import scala.Tuple2
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

import com.intel.daal.algorithms.svd._
import com.intel.daal.data_management.data._
import com.intel.daal.services._

object SVD {

    def computeSVD(matrix: RowMatrix) : RowMatrix = {
        val context = new DaalContext()

        val dataRDD = getAsPairRDD(matrix)

        val resultsFromStep1 = computeStep1Local(dataRDD)

        val resultsFromStep2 = computeStep2Master(context, resultsFromStep1._1)

        val resultsFromStep3 = computeStep3Local(resultsFromStep1._2, resultsFromStep2._2)

        dataRDD.unpersist()

        val resultRDD = resultsFromStep3.mapPartitions(
            (it: Iterator[Tuple2[Long, HomogenNumericTable]]) => {
                val res = new ArrayBuffer[Vector]()
                while (it.hasNext) {
                    val tup = it.next
                    val contextLocal = new DaalContext()
                    tup._2.unpack(contextLocal)
                    val nRows = tup._2.getNumberOfRows().toInt
                    val nCols = tup._2.getNumberOfColumns().toInt
                    val i: Int = 0
                    for (i <- 0 to (nRows - 1)) {
                        val arr = new Array[Double](nCols)
                        var buffer = DoubleBuffer.allocate(nCols)
                        buffer = tup._2.getBlockOfRows(i, 1, buffer)
                        buffer.get(arr)
                        res += Vectors.dense(arr)
                    }
                    tup._2.pack()
                    contextLocal.dispose()
                }

                res.iterator
            })
        
        context.dispose()

        new RowMatrix(resultRDD, matrix.numRows, matrix.numCols.toInt)
    }

    private def getAsPairRDD(matrix: RowMatrix): RDD[Tuple2[HomogenNumericTable, Long]] = {
        val nCols = matrix.numCols.toInt

        val tablesRDD = matrix.rows.mapPartitions(
            (it: Iterator[Vector]) => {
                val maxRows: Int = 200000
                var curRow: Int = 0
                val tables = new ArrayBuffer[HomogenNumericTable]()
                var arrays = new ArrayBuffer[Array[Double]]()
                var hasNext = it.hasNext
                while (hasNext) {
                    val curVector = it.next
                    hasNext = it.hasNext
                    val rowData = curVector.toArray
                    arrays += rowData

                    curRow += 1
                    if (curRow == maxRows || !hasNext) {
                        val arrData = new Array[Double](curRow * nCols)
                        var i: Int = 0
                        for (i <- 0 to (curRow - 1)) {
                            arrays(i).copyToArray(arrData, i * nCols)
                        }
                        val context = new DaalContext()
                        val table = new HomogenNumericTable(context, arrData, nCols, curRow)

                        table.pack()
                        tables += table

                        context.dispose()
                        curRow = 0
                        arrays.clear
                    }
                }
                tables.iterator
            }
        )

        tablesRDD.zipWithIndex.cache
    }

    private def computeStep1Local(dataRDD: RDD[Tuple2[HomogenNumericTable, Long]]): Tuple2[RDD[Tuple2[Long, DataCollection]], RDD[Tuple2[Long, DataCollection]]] = {

        /* Creating RDD containing both partial results for step2 and step3 */
        val dataFromStep1RDD = dataRDD.mapPartitions(
            (it: Iterator[Tuple2[HomogenNumericTable, Long]]) => {

                val res = new ArrayBuffer[Tuple2[Long, Tuple2[DataCollection, DataCollection]]]()

                while(it.hasNext) {
                    val tup = it.next

                    val contextLocal = new DaalContext()

                    /* Create algorithm to compute SVD decomposition on local node */
                    val svdStep1Local = new DistributedStep1Local(contextLocal, classOf[java.lang.Double], Method.defaultDense)
                    tup._1.unpack(contextLocal)
                    svdStep1Local.input.set( InputId.data, tup._1 )

                    /* Compute SVD Step 1  */
                    val pres = svdStep1Local.compute()
                    val dataFromStep1ForStep2 = pres.get( PartialResultId.outputOfStep1ForStep2 )
                    dataFromStep1ForStep2.pack()
                    val dataFromStep1ForStep3 = pres.get( PartialResultId.outputOfStep1ForStep3 )
                    dataFromStep1ForStep3.pack()

                    contextLocal.dispose()
                    res += new Tuple2(tup._2, new Tuple2(dataFromStep1ForStep2, dataFromStep1ForStep3))
                }
                res.iterator
            }).cache()

        /* Extracting partial results for step2 */
        val dataFromStep1ForStep2RDD = dataFromStep1RDD.mapPartitions(
            (it: Iterator[Tuple2[Long, Tuple2[DataCollection, DataCollection]]]) => {
                val res = new ArrayBuffer[Tuple2[Long, DataCollection]]()
                while(it.hasNext)
                {
                    val tup = it.next()
                    res += new Tuple2(tup._1, tup._2._1)
                }
                res.iterator
            })

        /* Extracting partial results for step3 */
        val dataFromStep1ForStep3RDD = dataFromStep1RDD.mapPartitions(
            (it: Iterator[Tuple2[Long, Tuple2[DataCollection, DataCollection]]]) => {
                val res = new ArrayBuffer[Tuple2[Long, DataCollection]]()
                while(it.hasNext)
                {
                    val tup = it.next()
                    res += new Tuple2(tup._1, tup._2._2)
                }
                res.iterator
            })

        new Tuple2(dataFromStep1ForStep2RDD, dataFromStep1ForStep2RDD)
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
        : RDD[Tuple2[Long, HomogenNumericTable]] = {

        val dataForStep3RDD = dataFromStep1ForStep3RDD.cogroup(dataFromStep2ForStep3RDD)

        val ntURDD = dataForStep3RDD.mapPartitions(
            (it: Iterator[Tuple2[Long, Tuple2[Iterable[DataCollection], Iterable[DataCollection]]]]) => {
                val res = new ArrayBuffer[Tuple2[Long, HomogenNumericTable]]()
                while(it.hasNext) {
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

                    res += new Tuple2[Long, HomogenNumericTable](tup._1, Ui)
                }
                
                res.iterator
            })

        ntURDD
    }
}
