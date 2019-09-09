/* file: LinearRegression.scala */
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

/*
 //  Content:
 //     Java sample of the implicit alternating least squares (ALS) algorithm.
 //
 //     The program trains the implicit ALS trainedModel on a supplied training data
 //     set.
 ////////////////////////////////////////////////////////////////////////////////
 */

package daal_for_mllib

import java.nio.DoubleBuffer

import org.apache.spark.api.java._
import org.apache.spark.api.java.function._
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel

import scala.Tuple2
import scala.collection.mutable.ArrayBuffer

import com.intel.daal.algorithms.linear_regression._
import com.intel.daal.algorithms.linear_regression.training._
import com.intel.daal.algorithms.linear_regression.prediction._
import com.intel.daal.data_management.data._
import com.intel.daal.services._

object LinearRegression {

    def train(dataRDD: RDD[LabeledPoint]) : LinearRegressionModel = {
        train(dataRDD, TrainingMethod.normEqDense)
    }

    def train(dataRDD: RDD[LabeledPoint], method: TrainingMethod) : LinearRegressionModel = {
        val partsRDD = computeStep1Local(dataRDD, method);
        computeStep2Master(partsRDD, method)
    }

    private def computeStep1Local(dataRDD: RDD[LabeledPoint], method: TrainingMethod) = {
        val internRDD = getFromLabeledPoint(dataRDD)

        val methodbc = dataRDD.sparkContext.broadcast(method.getValue())

        internRDD.map(tup => {
            val context = new DaalContext()
            var methodLocal: TrainingMethod = null
            if (methodbc.value == TrainingMethod.normEqDense.getValue()) {
                methodLocal = TrainingMethod.normEqDense
            } else {
                methodLocal = TrainingMethod.qrDense                
            }
            /* Create algorithm to train linear regression model using normal equations method on local nodes */
            val linRegLocal = new TrainingDistributedStep1Local(context, classOf[java.lang.Double], methodLocal)

            /* Set input data on local node */
            tup._1.unpack(context)
            tup._2.unpack(context)
            linRegLocal.input.set( TrainingInputId.data, tup._1 )
            linRegLocal.input.set( TrainingInputId.dependentVariable, tup._2 )

            /* Compute partial results on local node */
            val pres = linRegLocal.compute()
            pres.pack()

            context.dispose()

            pres
        })
    }

    private def computeStep2Master(partsRDD: RDD[PartialResult], method: TrainingMethod) : LinearRegressionModel = {
        val context = new DaalContext()

        val methodbc = partsRDD.sparkContext.broadcast(method.getValue())

        val reducedpr = partsRDD.treeReduce(
            (pr1, pr2) => {
                val contextLocal = new DaalContext()

                var methodLocal: TrainingMethod = null
                if (methodbc.value == TrainingMethod.normEqDense.getValue()) {
                    methodLocal = TrainingMethod.normEqDense
                } else {
                    methodLocal = TrainingMethod.qrDense                
                }
                /* Create algorithm to merge partial results on local node */
                val linRegMaster = new TrainingDistributedStep2Master(contextLocal, classOf[java.lang.Double], methodLocal)
                pr1.unpack(contextLocal)
                pr2.unpack(contextLocal)

                /* Set input models on local node */
                linRegMaster.input.add( MasterInputId.partialModels, pr1 )
                linRegMaster.input.add( MasterInputId.partialModels, pr2 )

                /* Compute partial result on local node */
                val redpr = linRegMaster.compute().asInstanceOf[PartialResult]
                redpr.pack()
                contextLocal.dispose()

                redpr
            }, 3)

        /* Create algorithm to train linear regression model using normal equations method on master node */
        val linRegMaster = new TrainingDistributedStep2Master(context, classOf[java.lang.Double], method)

        /* Add partial results computed on local nodes to the algorithm on master node */
        reducedpr.unpack(context)
        linRegMaster.input.add( MasterInputId.partialModels, reducedpr )

        /* Compute result on master node */
        linRegMaster.compute()

        /* Finalize the computations and retrieve linear regression results */
        val res = linRegMaster.finalizeCompute()

        val model = res.get(TrainingResultId.model)
        val beta = model.getBeta()

        val nRows = beta.getNumberOfRows().toInt
        val nCols = beta.getNumberOfColumns().toInt
        var buffer = DoubleBuffer.allocate(nRows * nCols)
        buffer = beta.getBlockOfRows(0, nRows, buffer)
        val arr = new Array[Double](nCols - 1)
        val b0 = buffer.get(0)
        var i: Int = 0
        for (i <- 1 to (nCols - 1)) {
            arr(i - 1) = buffer.get(i)
        }
        val resultVector = Vectors.dense(arr)

        context.dispose()

        new LinearRegressionModel(resultVector, b0)
    }

    private def getFromLabeledPoint (inputRDD: RDD[LabeledPoint]) : RDD[Tuple2[HomogenNumericTable, HomogenNumericTable]] = {
        inputRDD.mapPartitions(
            it => {
                val maxRows: Int = 200000
                var nCols: Int = 0
                var curRow: Int = 0
                val tables = new ArrayBuffer[Tuple2[HomogenNumericTable, HomogenNumericTable]]()
                var arrays = new ArrayBuffer[Tuple2[Array[Double], Double]]()
                var hasNext = true
                while (hasNext) {
                    val curPoint = it.next
                    hasNext = it.hasNext
                    arrays += new Tuple2(curPoint.features.toArray, curPoint.label)
                    nCols = curPoint.features.toArray.size

                    curRow += 1
                    if (curRow == maxRows || !hasNext) {
                        val arrData = new Array[Double](curRow * nCols)
                        val arrLabels = new Array[Double](curRow)
                        var i: Int = 0
                        for (i <- 0 to (curRow - 1)) {
                            arrays(i)._1.copyToArray(arrData, i * nCols)
                            arrLabels(i) = arrays(i)._2
                        }
                        val context = new DaalContext()
                        val tableData = new HomogenNumericTable(context, arrData, nCols, curRow)
                        val tableLabels = new HomogenNumericTable(context, arrLabels, 1, curRow)

                        tableData.pack()
                        tableLabels.pack()
                        tables += new Tuple2(tableData, tableLabels)

                        context.dispose()
                        curRow = 0
                        arrays.clear
                    }
                }
                tables.iterator
            })
    }
}
