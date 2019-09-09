/* file: Correlation.scala */
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

import java.nio.DoubleBuffer

import org.apache.spark.api.java._
import org.apache.spark.api.java.function._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf

import scala.Tuple2
import scala.collection.mutable.ArrayBuffer

import com.intel.daal.data_management.data._
import com.intel.daal.data_management.data_source._
import com.intel.daal.services._

import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

import com.intel.daal.algorithms.covariance._
import com.intel.daal.services._

object Statistics {

    def corr(data: RDD[Vector]) : Matrix = {
        val context = new DaalContext()

        val tablesRDD = getAsTablesRDD(data)

        val partsRDD = computeStep1Local(tablesRDD)

        val result = finalizeMergeOnMasterNode(context, partsRDD)

        val nRows = result.getNumberOfRows().toInt
        val nCols = result.getNumberOfColumns().toInt
        var buffer = DoubleBuffer.allocate(nRows * nCols)
        buffer = result.getBlockOfRows(0, nRows, buffer)
        val arr = new Array[Double](nRows * nCols)
        buffer.get(arr)
        val resultMatrix = Matrices.dense(nRows, nCols, arr)

        context.dispose()

        resultMatrix
    }

    private def getAsTablesRDD(data: RDD[Vector]): RDD[HomogenNumericTable] = {

        val tablesRDD = data.mapPartitions(
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
                        val nCols = rowData.size
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

        return tablesRDD
    }

    private def computeStep1Local(tablesRDD: RDD[HomogenNumericTable]): RDD[PartialResult] = {

        val partsRDD = tablesRDD.mapPartitions(
            (it: Iterator[HomogenNumericTable]) => {
                val tables = new ArrayBuffer[PartialResult]()
                val arrList = new ArrayBuffer[Array[Double]]()

                var hasNext = it.hasNext
                val context = new DaalContext()
                val covarianceOnline = new Online(context, classOf[java.lang.Double], Method.defaultDense)
                covarianceOnline.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix)
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

    private def finalizeMergeOnMasterNode(context: DaalContext, partsRDD: RDD[PartialResult]): HomogenNumericTable = {

        val reducedpr = partsRDD.treeReduce(
            (pr1, pr2) => {
                val context = new DaalContext()
                val covarianceMaster = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense)
                covarianceMaster.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix)
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
        covarianceMaster.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix)
        reducedpr.unpack(context)
        covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, reducedpr)
        covarianceMaster.compute()
        val res = covarianceMaster.finalizeCompute()

        res.get(ResultId.covariance).asInstanceOf[HomogenNumericTable]
    }
}
