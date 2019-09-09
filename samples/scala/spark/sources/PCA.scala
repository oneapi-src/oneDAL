/* file: PCA.scala */
//==============================================================================
// Copyright 2017-2019 Intel Corporation
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

import org.apache.spark.api.java._
import org.apache.spark.api.java.function._
import org.apache.spark.rdd.RDD

import scala.Tuple2
import scala.collection.mutable.ArrayBuffer

import com.intel.daal.data_management.data._
import com.intel.daal.services._

import org.apache.spark.mllib.linalg._

import com.intel.daal.algorithms.covariance._
import com.intel.daal.algorithms.pca.{Batch => PCABatch, Method => PCAMethod, InputId => PCAInputId, Result => PCAResult, ResultId => PCAResultId}
import com.intel.daal.services._

class PCAModel (
    val k: Int,
    val pc: DenseMatrix,
    val explainedVariance: DenseVector) {

    def transform(vector: DenseVector): Vector = {
        pc.transpose.multiply(vector)
    }
}

class PCA(val k: Int) {

    def fit(data: RDD[Vector]): PCAModel = {
        compute(data, k)
    }

    def fit(data: JavaRDD[Vector]): PCAModel = fit(data.rdd)

    private def compute(data: RDD[Vector], rank: Int): PCAModel = {

        val context = new DaalContext()

        val tablesRDD = getAsTablesRDD(data)
        val partsRDD = computeStep1Local(tablesRDD)
        tablesRDD.unpersist()
        val result = finalizeMergeOnMasterNode(context, partsRDD)
        partsRDD.unpersist()

        var nRows = result._1.getNumberOfRows().asInstanceOf[Int]
        var nCols = result._1.getNumberOfColumns().asInstanceOf[Int]

        if (nCols > rank) {
            nCols = rank
        }

        val pcFull = result._1.getDoubleArray()
        val pcArray = new Array[Double](nRows * nCols)

        var i: Int = 0
        for (i <- 0 to (nRows - 1)) {
            var j: Int = 0
            for (j <- 0 to (nCols - 1)) {
                pcArray(i * nCols + j) = pcFull(i * nRows + j)
            }
        }

        val evFull = result._2.getDoubleArray()
        val evArray = new Array[Double](nCols)

        var sum: Double = 0 
        for (i <- 0 to (nCols - 1)) {
            evArray(i) = evFull(i)
            sum = sum + evArray(i)
        }

        for (i <- 0 to (nCols - 1)) {
            evArray(i) = evArray(i) / sum
        }

        val pc = Matrices.dense(nRows, nCols, pcArray).asInstanceOf[DenseMatrix]
        val explainedVariance = Vectors.dense(evArray).asInstanceOf[DenseVector]

        context.dispose()

        new PCAModel(rank, pc, explainedVariance)
    }

    private def getAsTablesRDD(data: RDD[Vector]): RDD[HomogenNumericTable] = {

        val tablesRDD = data.mapPartitions(
            (it: Iterator[Vector]) => {
                val maxRows: Int = 100000
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
                        val numCols: Int = rowData.length
                        val arrData = new Array[Double](curRow * numCols)
                        var i: Int = 0
                        for (i <- 0 to (curRow - 1)) {
                            arrays(i).copyToArray(arrData, i * numCols)
                        }
                        val context = new DaalContext()
                        val table = new HomogenNumericTable(context, arrData, numCols, curRow)

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

        tablesRDD.cache
    }

    private def computeStep1Local(tablesRDD: RDD[HomogenNumericTable]): RDD[PartialResult] = {

        val partsRDD = tablesRDD.mapPartitions(
            (it: Iterator[HomogenNumericTable]) => {
                val tables = new ArrayBuffer[PartialResult]()
                val arrList = new ArrayBuffer[Array[Double]]()

                var hasNext = it.hasNext
                val context = new DaalContext()
                val covarianceOnline = new Online(context, classOf[java.lang.Double], Method.defaultDense)
                var partialRes = new PartialResult(context)

                while (hasNext) {
                    val table = it.next
                    hasNext = it.hasNext
                    val contextLocal = new DaalContext()
                    table.unpack(contextLocal)
                    covarianceOnline.input.set(InputId.data, table)
                    partialRes = covarianceOnline.compute()
                    contextLocal.dispose()
                }

                partialRes.pack()
                context.dispose()

                val prList = new ArrayBuffer[PartialResult]()
                prList += partialRes

                prList.iterator
            }
        )

        partsRDD.cache
    }

    private def finalizeMergeOnMasterNode(context: DaalContext, partsRDD: RDD[PartialResult]): Tuple2[HomogenNumericTable, HomogenNumericTable] = {

        val reducedpr = partsRDD.treeReduce(
            (pr1, pr2) => {

                val context = new DaalContext()
                val covarianceMaster = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense)
                pr1.unpack(context)
                pr2.unpack(context)
                covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, pr1)
                covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, pr2)
                val redpr = covarianceMaster.compute()
                redpr.pack()
                context.dispose()

                redpr
            }, 4
        )

        val covarianceMaster = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense)
        reducedpr.unpack(context)
        covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, reducedpr)
        covarianceMaster.compute()
        val covarianceRes = covarianceMaster.finalizeCompute()

        val pcaBatch = new PCABatch(context, classOf[java.lang.Double], PCAMethod.correlationDense)
        pcaBatch.input.set(PCAInputId.correlation, covarianceRes.get(ResultId.covariance).asInstanceOf[HomogenNumericTable])
        val res = pcaBatch.compute()

        new Tuple2(res.get(PCAResultId.eigenVectors).asInstanceOf[HomogenNumericTable],
                   res.get(PCAResultId.eigenValues).asInstanceOf[HomogenNumericTable])
    }
}
