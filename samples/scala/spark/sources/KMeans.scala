/* file: KMeans.scala */
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

import org.apache.spark.api.java._
import org.apache.spark.api.java.function._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering.KMeansModel

import scala.Tuple2
import scala.collection.mutable.ArrayBuffer

import com.intel.daal.algorithms.kmeans._
import com.intel.daal.algorithms.kmeans.init._
import com.intel.daal.data_management.data._
import com.intel.daal.services._

class DAALKMeansModel(override val clusterCenters: Array[Vector]) extends KMeansModel(clusterCenters) {}

object KMeans {
    def train(data: RDD[Vector], nClusters: Int, nIterations: Int): DAALKMeansModel = {
        new KMeans().setNClusters(nClusters).setNIterations(nIterations).run(data)
    }

    def train(data: RDD[Vector], nClusters: Int, nIterations: Int, dummy: Int, initializationMethod: String): DAALKMeansModel = {
        // initializationMethod = "random" is only available at the moment
        new KMeans().setNClusters(nClusters).setNIterations(nIterations).run(data)
    }
}

class KMeansResult private (var centroids: HomogenNumericTable, var nIterations: Int, var nClusters: Long) {}

class KMeans private (private var nClusters: Int, private var nIterations: Int) {

    def this() = this(10, 10)

    def getNClusters: Int = nClusters

    def setNClusters(nClusters: Int): this.type = {
        this.nClusters = nClusters
        this
    }

    def getNIterations: Int = nIterations

    def setNIterations(nIterations: Int): this.type = {
        this.nIterations = nIterations
        this
    }

    def run(data: RDD[Vector]): DAALKMeansModel = {
        train(data, nClusters, nIterations)
    }

    def train(data: RDD[Vector], nClusters: Int, nIterations: Int) : DAALKMeansModel = {

        val internRDD = getAsPairRDD(data)

        val tupleRes = computeOffsets(internRDD)
        val offsets = tupleRes._1          
        val numVectors = tupleRes._2

        var centroids = computeInit(nClusters, numVectors, offsets, internRDD)

        val it: Int = 0
        for(it <- 1 to nIterations) {
            centroids = computeIter(nClusters, centroids, internRDD)
        }

        internRDD.unpersist(true)

        val context = new DaalContext()
        centroids.unpack(context)
        val resTable = centroids.getDoubleArray()
        val resArray = new Array[Vector](nClusters)
        val numCols = centroids.getNumberOfColumns().asInstanceOf[Int]
        var i: Int = 0
        for (i <- 0 to (nClusters - 1)) {
            val internArray = new Array[Double](numCols)
            var j: Int = 0
            for (j <- 0 to (numCols - 1)) {
                internArray(j) = resTable(i * numCols + j)
            }
            resArray(i) = Vectors.dense(internArray)
        }
        centroids.dispose()

        new DAALKMeansModel(resArray)
    }

    def getAsPairRDD(data: RDD[Vector]): RDD[Tuple2[HomogenNumericTable, Long]] = {

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

        tablesRDD.zipWithIndex.cache
    }

    def computeOffsets(internRDD: RDD[Tuple2[HomogenNumericTable, Long]]) : Tuple2[Array[Long], Long] = {

        val tmpOffsetRDD = internRDD.mapPartitions(
            (it: Iterator[Tuple2[HomogenNumericTable, Long]]) => {
                val res = new ArrayBuffer[Tuple2[Long, Long]]()
                while(it.hasNext) {
                    val context = new DaalContext()
                    val tup = it.next
                    tup._1.unpack(context)
                    val numRows = tup._1.getNumberOfRows()
                    tup._1.pack()
                    context.dispose
                    res += new Tuple2(tup._2, numRows)
                }

                res.iterator
            })

        val numbersOfRows = tmpOffsetRDD.collect

        tmpOffsetRDD.unpersist(true)

        var numVectors: Long = 0

        val i: Int = 0
        for (i <- 0 to (numbersOfRows.size - 1)) {
            numVectors = numVectors + numbersOfRows(i)._2
        }

        val partition = new Array[Long](numbersOfRows.size + 1)
        partition(0) = 0
        for(i <- 0 to (numbersOfRows.size - 1)) {
            partition(i + 1) = partition(i) + numbersOfRows(i)._2
        }
        new Tuple2(partition, numVectors)
    }

    def computeCost(model: KMeansResult, data: RDD[Vector]) : Double = {

        val internRDD = getAsPairRDD(data)

        val partsRDDcompute = computeLocal(model.nClusters.asInstanceOf[Long], model.centroids, internRDD)

        val cost = computeMasterCost(model.nClusters.asInstanceOf[Long], partsRDDcompute)

        internRDD.unpersist(true)
        partsRDDcompute.unpersist(true)

        cost
    }

    def computeInit(nClusters: Int, nV: Long, offsets: Array[Long], internRDD: RDD[Tuple2[HomogenNumericTable, Long]]) : HomogenNumericTable = {
 
        val contextM = new DaalContext()

        /* Create an algorithm to compute k-means on the master node */
        val kmeansMasterInit = new InitDistributedStep2Master(contextM,
                                                              classOf[java.lang.Double],
                                                              InitMethod.randomDense,
                                                              nClusters.asInstanceOf[Long])

        val tmpInitRDD = internRDD.mapPartitions(
            (it: Iterator[Tuple2[HomogenNumericTable, Long]]) => {
                val res = new ArrayBuffer[Tuple2[Int, InitPartialResult]]

                while(it.hasNext) {
                    val tup = it.next
                    val context = new DaalContext()

                    /* Create an algorithm to initialize the K-Means algorithm on local nodes */
                    val kmeansLocalInit = new InitDistributedStep1Local(context,
                                                                        classOf[java.lang.Double],
                                                                        InitMethod.randomDense,
                                                                        nClusters,
                                                                        nV,
                                                                        offsets(tup._2.asInstanceOf[Int]))

                    /* Set the input data on local nodes */
                    tup._1.unpack(context)
                    kmeansLocalInit.input.set(InitInputId.data, tup._1)

                    /* Initialize the K-Means algorithm on local nodes */
                    val pres = kmeansLocalInit.compute()
                    pres.pack()
                    tup._1.pack()
                    
                    context.dispose()

                    res += new Tuple2(tup._2.asInstanceOf[Int], pres)
                }

                res.iterator
            })

        val partsList = tmpInitRDD.collect

        tmpInitRDD.unpersist(true)

        /* Add partial results computed on local nodes to the algorithm on the master node */
        val i: Int = 0
        for (i <- 0 to (partsList.size - 1)) {
            partsList(i)._2.unpack(contextM)
            kmeansMasterInit.input.add(InitDistributedStep2MasterInputId.partialResults, partsList(i)._2)
        }

        /* Compute k-means on the master node */
        kmeansMasterInit.compute()

        /* Finalize computations and retrieve the results */
        val initResult = kmeansMasterInit.finalizeCompute()
        val ret = initResult.get(InitResultId.centroids).asInstanceOf[HomogenNumericTable]
        ret.pack()

        contextM.dispose()
        
        ret
    }

    def computeLocal(nClusters: Long, centroids: HomogenNumericTable, internRDD: RDD[Tuple2[HomogenNumericTable, Long]]) : RDD[Tuple2[Int, PartialResult]] = {

        val partsRDDcompute = internRDD.mapPartitions(
            (it: Iterator[Tuple2[HomogenNumericTable, Long]]) => {
                val res = new ArrayBuffer[Tuple2[Int, PartialResult]];
                while(it.hasNext) {
                    val tup = it.next
                    val context = new DaalContext()

                    /* Create an algorithm to compute k-means on local nodes */
                    val kmeansLocal = new DistributedStep1Local(context, classOf[java.lang.Double], Method.defaultDense, nClusters)
                    kmeansLocal.parameter.setAssignFlag(false)

                    /* Set the input data on local nodes */
                    tup._1.unpack(context)
                    centroids.unpack(context)
                    kmeansLocal.input.set(InputId.data, tup._1)
                    kmeansLocal.input.set(InputId.inputCentroids, centroids)

                    /* Compute k-means on local nodes */
                    val pres = kmeansLocal.compute()
                    pres.pack()

                    tup._1.pack()
                    centroids.pack()
                    context.dispose()

                    res += new Tuple2(tup._2.asInstanceOf[Int], pres)
                }

                res.iterator
            }).cache

        partsRDDcompute
    }

    def computeMaster(nClusters: Long, partsRDDcompute: RDD[Tuple2[Int, PartialResult]]) : HomogenNumericTable = {
        val context = new DaalContext()

        /* Create an algorithm to compute k-means on the master node */
        val kmeansMaster = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense, nClusters)

        val partsList = partsRDDcompute.collect()

        partsRDDcompute.unpersist(true)

        /* Add partial results computed on local nodes to the algorithm on the master node */
        val i: Int = 0
        for (i <- 0 to (partsList.size - 1)) {
            partsList(i)._2.unpack(context)
            kmeansMaster.input.add(DistributedStep2MasterInputId.partialResults, partsList(i)._2)
        }

        /* Compute k-means on the master node */
        kmeansMaster.compute()

        /* Finalize computations and retrieve the results */
        val res = kmeansMaster.finalizeCompute()

        val centroids = res.get(ResultId.centroids).asInstanceOf[HomogenNumericTable]
        centroids.pack()

        context.dispose()

        centroids
    }

    def computeIter(nClusters: Long, centroids: HomogenNumericTable, internRDD: RDD[Tuple2[HomogenNumericTable, Long]]) : HomogenNumericTable = {
        val contextI = new DaalContext()

        /* Create an algorithm to compute k-means on the master node */
        val kmeansMaster = new DistributedStep2Master(contextI, classOf[java.lang.Double], Method.defaultDense, nClusters)

        val tmpIterRDD = internRDD.mapPartitions(
            (it: Iterator[Tuple2[HomogenNumericTable, Long]]) => {
                val res = new ArrayBuffer[PartialResult]
                while(it.hasNext) {
                    val tup = it.next()
                    val context = new DaalContext()

                    /* Create an algorithm to compute k-means on local nodes */
                    val kmeansLocal = new DistributedStep1Local(context, classOf[java.lang.Double], Method.defaultDense, nClusters)
                    kmeansLocal.parameter.setAssignFlag(false)

                    /* Set the input data on local nodes */
                    tup._1.unpack(context)
                    centroids.unpack(context)
                    kmeansLocal.input.set(InputId.data, tup._1)
                    kmeansLocal.input.set(InputId.inputCentroids, centroids)

                    /* Compute k-means on local nodes */
                    val pres = kmeansLocal.compute()

                    pres.pack()
                    tup._1.pack()
                    centroids.pack()

                    context.dispose()

                    res += pres
                }

                res.iterator
            })

        val reducedPartialResult = tmpIterRDD.treeReduce(
            (pr1, pr2) => {
                val context = new DaalContext()

                val kmeansMaster = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense, nClusters)

                pr1.unpack(context)
                pr2.unpack(context)
                kmeansMaster.input.add(DistributedStep2MasterInputId.partialResults, pr1)
                kmeansMaster.input.add(DistributedStep2MasterInputId.partialResults, pr2)

                val pr = kmeansMaster.compute()

                pr.pack()
                context.dispose()

                pr
            }, 3)

        tmpIterRDD.unpersist(true)

        reducedPartialResult.unpack(contextI)
        kmeansMaster.input.add(DistributedStep2MasterInputId.partialResults, reducedPartialResult)
        /* Compute k-means on the master node */

        kmeansMaster.compute()

        /* Finalize computations and retrieve the results */
        val res = kmeansMaster.finalizeCompute()

        val centroidsI = res.get(ResultId.centroids).asInstanceOf[HomogenNumericTable]
        centroidsI.pack()
        
        contextI.dispose()

        centroidsI
    }

    def computeMasterCost(nClusters: Long, partsRDDcompute: RDD[Tuple2[Int, PartialResult]]) : Double = {
        val context = new DaalContext()
        /* Create an algorithm to compute k-means on the master node */
        val kmeansMaster = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense, nClusters)

        val partsList = partsRDDcompute.collect()

        /* Add partial results computed on local nodes to the algorithm on the master node */
        val i: Int = 0
        for (i <- 0 to (partsList.size - 1)) {
            partsList(i)._2.unpack(context)
            kmeansMaster.input.add(DistributedStep2MasterInputId.partialResults, partsList(i)._2)
        }

        /* Compute k-means on the master node */
        kmeansMaster.compute()

        /* Finalize computations and retrieve the results */
        val res = kmeansMaster.finalizeCompute()

        val goalFunc = res.get(ResultId.objectiveFunction).asInstanceOf[HomogenNumericTable]

        val goal = goalFunc.getDoubleArray()
        val cost = goal(0)

        context.dispose()

        cost
    }
}
