/* file: ALS.scala */
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

import java.util.List
import java.nio.DoubleBuffer
import java.nio.IntBuffer
import java.io.IOException
import java.lang.ClassNotFoundException
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

import org.apache.spark.api.java._
import org.apache.spark.api.java.function._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

import scala.Tuple2
import scala.Tuple3
import scala.util.Sorting
import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer

import com.intel.daal.data_management.data._
import com.intel.daal.data_management.data_source._
import com.intel.daal.algorithms.implicit_als._
import com.intel.daal.algorithms.implicit_als.training._
import com.intel.daal.algorithms.implicit_als.training.init._
import com.intel.daal.algorithms.implicit_als.prediction.ratings._
import com.intel.daal.services._

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel

object ALS {
    /**
     * Train the implicit ALS model on a supplied dataset given as an RDD of the (userID, productID, rating) tuples
     *
     * @param user       Number of users
     * @param products   Number of products
     * @param ratings    RDD of (userID, productID, rating) tuples
     * @param rank       Number of factors
     * @param iterations Maximum number of iterations of the implicit ALS training algorithm
     * @param lambda     Regularization parameter
     * @param blocks     Number of input data parts used in implicit ALS initialization algorithm
     * @param alpha      Confidence parameter of the implicit ALS training algorithm
     * @param seed       Value of the seed parameter used in implicit ALS initialization algorithm
     */
    def trainImplicit(
        users: Int,
        products: Int,
        ratings: RDD[Rating],
        rank: Int,
        iterations: Int,
        lambda: Double,
        blocks: Int,
        alpha: Double,
        seed: Long): MatrixFactorizationModel = {
      new ALS(users, products, blocks, rank, iterations, lambda, alpha, seed).run(ratings)
    }

    /**
     * Train the implicit ALS model on a supplied dataset given as an RDD of the (userID, productID, rating) tuples
     *
     * @param user       Number of users
     * @param products   Number of products
     * @param ratings    RDD of (userID, productID, rating) tuples
     * @param rank       Number of factors
     * @param iterations Maximum number of iterations of the implicit ALS training algorithm
     * @param lambda     Regularization parameter
     * @param blocks     Number of input data parts used in implicit ALS initialization algorithm
     * @param alpha      Confidence parameter of the implicit ALS training algorithm
     */
    def trainImplicit(
        users: Int,
        products: Int,
        ratings: RDD[Rating],
        rank: Int,
        iterations: Int,
        lambda: Double,
        blocks: Int,
        alpha: Double): MatrixFactorizationModel = {
      new ALS(users, products, blocks, rank, iterations, lambda, alpha).run(ratings)
    }

    /**
     * Train the implicit ALS model on a supplied dataset given as an RDD of the (userID, productID, rating) tuples
     *
     * @param user       Number of users
     * @param products   Number of products
     * @param ratings    RDD of (userID, productID, rating) tuples
     * @param rank       Number of factors
     * @param iterations Maximum number of iterations of the implicit ALS training algorithm
     * @param lambda     Regularization parameter
     * @param alpha      Confidence parameter of the implicit ALS training algorithm
     */
    def trainImplicit(
        users: Int,
        products: Int,
        ratings: RDD[Rating],
        rank: Int,
        iterations: Int,
        lambda: Double,
        alpha: Double): MatrixFactorizationModel = {
      trainImplicit(users, products, ratings, rank, iterations, lambda, -1, alpha)
    }

    /**
     * Train the implicit ALS model on a supplied dataset given as an RDD of the (userID, productID, rating) tuples
     *
     * @param user       Number of users
     * @param products   Number of products
     * @param ratings    RDD of (userID, productID, rating) tuples
     * @param rank       Number of factors
     * @param iterations Maximum number of iterations of the implicit ALS training algorithm
     */
    def trainImplicit(
        users: Int,
        products: Int,
        ratings: RDD[Rating],
        rank: Int,
        iterations: Int): MatrixFactorizationModel = {
      trainImplicit(users, products, ratings, rank, iterations, 0.01, -1, 40.0)
    }
}

class ALS private (
    private var nUsers: Int,
    private var nProducts: Int,
    private var nBlocks: Int,
    private var nFactors: Int,
    private var nIterations: Int,
    private var lambda: Double,
    private var alpha: Double,
    private var seed: Long = System.nanoTime()) extends Serializable {

    def this() = this(-1, -1, 1, 10, 10, 0.01, 40.0)

    def setNUsers(nUsers: Int): this.type = {
        this.nUsers = nUsers
        this
    }

    def setNProducts(nProducts: Int): this.type = {
        this.nProducts = nProducts
        this
    }

    def setNBlocks(nBlocks: Int): this.type = {
        this.nBlocks = nBlocks
        this
    }

    def setNFactors(nFactors: Int): this.type = {
        this.nFactors = nFactors
        this
    }

    def setNIterations(nIterations: Int): this.type = {
        this.nIterations = nIterations
        this
    }

    def setLambda(lambda: Double): this.type = {
        this.lambda = lambda
        this
    }

    def setAlpha(alpha: Double): this.type = {
        this.alpha = alpha
        this
    }

    def setSeed(seed: Long): this.type = {
        this.seed = seed
        this
    }

    def run (data: RDD[Rating]) : MatrixFactorizationModel = {
        val sc = data.sparkContext

        val nBlocks = if (this.nBlocks == -1) {
            math.max(sc.defaultParallelism, data.partitions.length / 2)
        } else {
            this.nBlocks
        }

        val dataRDD = getAsTableRDD(data, nBlocks)
        val trainedModel = trainModel(dataRDD)
        val productFeatures = getAsFeaturesRDD(trainedModel._1, nBlocks).cache
        val userFeatures = getAsFeaturesRDD(trainedModel._2, nBlocks).cache

        userFeatures.count
        productFeatures.count

        new MatrixFactorizationModel(nFactors, userFeatures, productFeatures)
    }

    private def getAsFeaturesRDD(factors: RDD[Tuple2[Int, DistributedPartialResultStep4]], nBlocks: Int) = {
        val itemsInBlock = (nProducts + nBlocks - 1) / nBlocks

        factors.flatMap(
            (tup: Tuple2[Int, DistributedPartialResultStep4]) => {
                val contextLocal = new DaalContext()
                tup._2.unpack(contextLocal)
                val indices = tup._2.get(DistributedPartialResultStep4Id.outputOfStep4).getIndices()
                val factors = tup._2.get(DistributedPartialResultStep4Id.outputOfStep4).getFactors()
                val featArr = new ArrayBuffer[Tuple2[Int, Array[Double]]]()
                val nRows = factors.getNumberOfRows().asInstanceOf[Int]
                val nCols = factors.getNumberOfColumns().asInstanceOf[Int]
                val startRow: Int = itemsInBlock * tup._1
                var i: Int = 0
                var indicesBuffer = IntBuffer.allocate(nRows)
                var factorsBuffer = DoubleBuffer.allocate(nCols)
                indicesBuffer = indices.getBlockOfRows(0, nRows, indicesBuffer)
                for (i <- 0 to (nRows - 1)) {
                    factorsBuffer = factors.getBlockOfRows(i, 1, factorsBuffer)
                    val tempArr = new Array[Double](nCols)
                    factorsBuffer.get(tempArr)
                    featArr += new Tuple2(indicesBuffer.get(i) + startRow, tempArr)
                }

                tup._2.pack()

                featArr.iterator
            }).cache
    }

    private def getAsTableRDD(data: RDD[Rating], nBlocks: Int) : RDD[Tuple2[Int, CSRNumericTable]] = {
        val cachedData = data.cache
        val itemsInBlock = (nProducts + nBlocks - 1) / nBlocks

        val dataGrouped = cachedData.groupBy(value => value.product / itemsInBlock)

        var bcNUsers = data.sparkContext.broadcast(nUsers)

        val tableRDD = dataGrouped.map(group => {
            var startTime = System.currentTimeMillis()
            val array = group._2.toArray

            var finishTime1 = System.currentTimeMillis()
            val time1 = (finishTime1 - startTime).toDouble / 1000.0

            object ratingsOrdering extends Ordering[Rating] {
                def compare(a: Rating, b: Rating) = {
                    val diff = a.product.compare(b.product)
                    if (diff != 0) {
                        diff
                    } else {
                        a.user.compare(b.user)
                    }
                }
            }

            Sorting.quickSort(array)(ratingsOrdering)

            val nFeatures = bcNUsers.value
            val nVectors = itemsInBlock
            val nValues = array.size

            val rowIndex = new Array[Long](nVectors + 1)

            val startRow: Int = itemsInBlock * group._1

            var pos: Int = 0
            var row: Int = 0
            var length = 0
            for (row <- 0 to (nVectors - 1)) {
                while (pos < nValues && array(pos).product == row + startRow) {
                    if (pos == 0 || array(pos - 1).user != array(pos).user || array(pos - 1).product != array(pos).product) {
                        length = length + 1
                    }
                    pos = pos + 1
                }
            }

            val values = new Array[Double](length)
            val columns = new Array[Long](length)

            var rpos: Int = 0
            pos = 0
            for (row <- 0 to (nVectors - 1)) {
                rowIndex(row) = rpos + 1
                while (pos < nValues && array(pos).product == row + startRow) {
                    if (pos == 0 || array(pos - 1).user != array(pos).user || array(pos - 1).product != array(pos).product) {
                        values(rpos) = array(pos).rating
                        columns(rpos) = array(pos).user + 1
                        rpos = rpos + 1
                    }
                    pos = pos + 1
                }
            }

            rowIndex(nVectors) = rpos + 1

            val contextLocal = new DaalContext()
            val table = new CSRNumericTable(contextLocal, values, columns, rowIndex, nFeatures, nVectors)

            table.pack()

            contextLocal.dispose()

            new Tuple2(group._1, table)
        }).cache

        tableRDD
    }

    private def trainModel(dataRDD: RDD[Tuple2[Int, CSRNumericTable]]) = {
        val usersPartition = new Array[Long](1)
        usersPartition(0) = dataRDD.count()

        var transposedDataRDD: RDD[Tuple2[Int, CSRNumericTable]] = null

        var userOffset: RDD[Tuple2[Int, NumericTable]] = null
        var itemOffset: RDD[Tuple2[Int, NumericTable]] = null

        var userStep3LocalInput: RDD[Tuple2[Int, KeyValueDataCollection]] = null
        var itemStep3LocalInput: RDD[Tuple2[Int, KeyValueDataCollection]] = null

        var usersPartialResultLocal: RDD[Tuple2[Int, DistributedPartialResultStep4]] = null
        var itemsPartialResultLocal: RDD[Tuple2[Int, DistributedPartialResultStep4]] = null

        /* Initialize distributed implicit ALS model */
        val initStep1LocalResult = initializeStep1Local(dataRDD, usersPartition).cache
        val initStep1LocalResultForStep2 = initStep1LocalResult.map(tup => new Tuple2(tup._1, tup._2._4)).cache

        val initStep2LocalInput = getInputForStep2(initStep1LocalResultForStep2).cache

        val initStep2LocalResult = initializeStep2Local(initStep2LocalInput).cache

        itemsPartialResultLocal = initStep1LocalResult.map(tup => new Tuple2(tup._1, tup._2._1)).cache
        transposedDataRDD       = initStep2LocalResult.map(tup => new Tuple2(tup._1, tup._2._1)).cache
        itemStep3LocalInput     = initStep1LocalResult.map(tup => new Tuple2(tup._1, tup._2._2)).cache
        userStep3LocalInput     = initStep2LocalResult.map(tup => new Tuple2(tup._1, tup._2._2)).cache
        userOffset              = initStep1LocalResult.map(tup => new Tuple2(tup._1, tup._2._3)).cache
        itemOffset              = initStep2LocalResult.map(tup => new Tuple2(tup._1, tup._2._3)).cache

        /* Train distributed implicit ALS model */
        var step2MasterResultCopies: RDD[Tuple2[Int, DistributedPartialResultStep2]] = null

        var step1LocalResult: RDD[Tuple2[Int, DistributedPartialResultStep1]] = null
        var step3LocalResult: RDD[Tuple2[Int, Tuple2[Int, PartialModel]]] = null

        var iteration: Int = 0
        for (iteration <- 0 to (nIterations - 1)) {
            step1LocalResult = computeStep1Local(itemsPartialResultLocal)
            step2MasterResultCopies = computeStep2Master(step1LocalResult)
            step3LocalResult = computeStep3Local(itemOffset, itemsPartialResultLocal, itemStep3LocalInput)
            usersPartialResultLocal = computeStep4Local(step2MasterResultCopies, step3LocalResult, transposedDataRDD).cache()

            if(iteration != (nIterations - 1)) {
                itemsPartialResultLocal.unpersist()
            }

            step1LocalResult = computeStep1Local(usersPartialResultLocal)
            step2MasterResultCopies = computeStep2Master(step1LocalResult)
            step3LocalResult = computeStep3Local(userOffset, usersPartialResultLocal, userStep3LocalInput)
            itemsPartialResultLocal = computeStep4Local(step2MasterResultCopies, step3LocalResult, dataRDD).cache()

            if(iteration != (nIterations - 1)) {
                usersPartialResultLocal.unpersist()
            }
        }

        new Tuple2(itemsPartialResultLocal, usersPartialResultLocal)
    }

    private def initializeStep1Local(dataRDD: RDD[Tuple2[Int, CSRNumericTable]], usersPartition: Array[Long]) = {
        var bcNUsers = dataRDD.sparkContext.broadcast(nUsers)
        var bcNFactors = dataRDD.sparkContext.broadcast(nFactors)
        var bcSeed = dataRDD.sparkContext.broadcast(seed)

        dataRDD.map(
            (tup: Tuple2[Int, CSRNumericTable]) => {
                val contextLocal = new DaalContext()
                tup._2.unpack(contextLocal)

                /* Create an algorithm object to initialize the implicit ALS model with the fastCSR method */
                val initAlgorithm = new InitDistributedStep1Local(contextLocal, classOf[java.lang.Double], InitMethod.fastCSR)
                initAlgorithm.parameter.setFullNUsers(bcNUsers.value)
                initAlgorithm.parameter.setNFactors(bcNFactors.value)
                initAlgorithm.parameter.setSeed(bcSeed.value + tup._1)
                initAlgorithm.parameter.setPartition(new HomogenNumericTable(contextLocal, usersPartition, 1, usersPartition.length))

                /* Pass a training data set and dependent values to the algorithm */
                initAlgorithm.input.set(InitInputId.data, tup._2)

                /* Initialize the implicit ALS model */
                val initPartialResult = initAlgorithm.compute()

                val partialModel = initPartialResult.get(InitPartialResultId.partialModel)

                val partialResultStep4 = new DistributedPartialResultStep4(contextLocal)
                partialResultStep4.set(DistributedPartialResultStep4Id.outputOfStep4ForStep1, partialModel)

                val step3LocalInput      = initPartialResult.get(InitPartialResultBaseId.outputOfInitForComputeStep3)
                val offset               = initPartialResult.get(InitPartialResultBaseId.offsets, tup._1.toLong)
                val initStep1LocalResult = initPartialResult.get(InitPartialResultCollectionId.outputOfStep1ForStep2)

                partialResultStep4.pack()
                step3LocalInput.pack()
                offset.pack()
                initStep1LocalResult.pack()
                tup._2.pack()

                val value = new Tuple4(partialResultStep4, step3LocalInput, offset, initStep1LocalResult)

                contextLocal.dispose()

                new Tuple2(tup._1, value)
            })
    }

    private def initializeStep2Local(inputOfStep2FromStep1: RDD[Tuple2[Int, Iterable[Tuple2[Int, NumericTable]]]])
        : RDD[Tuple2[Int, Tuple3[CSRNumericTable, KeyValueDataCollection, NumericTable]]] = {

        inputOfStep2FromStep1.map(
            (tup: Tuple2[Int, Iterable[Tuple2[Int, NumericTable]]]) => {

                val contextLocal = new DaalContext()
                val initStep2LocalInput = new KeyValueDataCollection(contextLocal)

                for (item <- tup._2) {
                    item._2.unpack(contextLocal)
                    initStep2LocalInput.set(item._1, item._2)
                }

                val initAlgorithm = new InitDistributedStep2Local(contextLocal, classOf[java.lang.Double], InitMethod.fastCSR)

                initAlgorithm.input.set(InitStep2LocalInputId.inputOfStep2FromStep1, initStep2LocalInput)

                /* Compute partial results of the second step on local nodes */
                val initPartialResult = initAlgorithm.compute()

                val dataTableTransposed = initPartialResult.get(InitDistributedPartialResultStep2Id.transposedData).asInstanceOf[CSRNumericTable]
                val step3LocalInput     = initPartialResult.get(InitPartialResultBaseId.outputOfInitForComputeStep3)
                val offset              = initPartialResult.get(InitPartialResultBaseId.offsets, tup._1.toInt)

                dataTableTransposed.pack()
                step3LocalInput.pack()
                offset.pack()

                contextLocal.dispose()

                val value = Tuple3(dataTableTransposed, step3LocalInput, offset)

                new Tuple2(tup._1, value)
            })
    }

    private def getInputForStep2(initStep1LocalResultForStep2: RDD[Tuple2[Int, KeyValueDataCollection]])
        : RDD[Tuple2[Int, Iterable[Tuple2[Int, NumericTable]]]] = {

        initStep1LocalResultForStep2.flatMap(
            (tup: Tuple2[Int, KeyValueDataCollection]) => {

                val contextLocal = new DaalContext()
                val collection = tup._2
                collection.unpack(contextLocal)

                val list = new ArrayBuffer[Tuple2[Int, Tuple2[Int, NumericTable]]]()
                var i: Int = 0
                for(i <- 0 to (collection.size().toInt - 1)) {
                    val table = collection.get(i).asInstanceOf[NumericTable]
                    table.pack()
                    val blockToIdWithTuple = new Tuple2(collection.getKeyByIndex(i).toInt, new Tuple2(tup._1, table))
                    list += blockToIdWithTuple
                }

                contextLocal.dispose()

                list.iterator
            }).groupByKey()
    }

    private def computeStep1Local(partialResultLocal: RDD[Tuple2[Int, DistributedPartialResultStep4]])
        : RDD[Tuple2[Int, DistributedPartialResultStep1]] = {

        var bcNFactors = partialResultLocal.sparkContext.broadcast(nFactors)
        var bcAlpha = partialResultLocal.sparkContext.broadcast(alpha)
        var bcLambda = partialResultLocal.sparkContext.broadcast(lambda)

        partialResultLocal.map(
            (tup: Tuple2[Int, DistributedPartialResultStep4]) => {

                val contextLocal = new DaalContext()
                tup._2.unpack(contextLocal)

                /* Create algorithm objects to compute a implisit ALS algorithm in the distributed processing mode using the fastCSR method */
                val algorithm = new DistributedStep1Local(contextLocal, classOf[java.lang.Double], TrainingMethod.fastCSR)
                algorithm.parameter.setNFactors(bcNFactors.value)
                algorithm.parameter.setAlpha(bcAlpha.value)
                algorithm.parameter.setLambda(bcLambda.value)

                /* Set input objects for the algorithm */
                algorithm.input.set(PartialModelInputId.partialModel, tup._2.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1))

                /* Compute partial estimates on local nodes */
                val step1LocalResult = algorithm.compute()

                step1LocalResult.pack()
                tup._2.pack()

                contextLocal.dispose()

                new Tuple2(tup._1, step1LocalResult)
            })
    }

    private def computeStep2Master(step1LocalResult: RDD[Tuple2[Int, DistributedPartialResultStep1]])
        : RDD[Tuple2[Int, DistributedPartialResultStep2]] = {

        val contextLocal = new DaalContext()

        val step1LocalResultList = step1LocalResult.collect()

        /* Create algorithm objects to compute a implisit ALS algorithm in the distributed processing mode using the fastCSR method */
        val algorithm = new DistributedStep2Master(contextLocal, classOf[java.lang.Double], TrainingMethod.fastCSR)
        algorithm.parameter.setNFactors(nFactors)
        algorithm.parameter.setAlpha(alpha)
        algorithm.parameter.setLambda(lambda)
        val nBlocks = step1LocalResultList.size.toInt

        /* Set input objects for the algorithm */
        for (value <- step1LocalResultList) {
            value._2.unpack(contextLocal)
            algorithm.input.add(MasterInputId.inputOfStep2FromStep1, value._2)
        }

        /* Compute a partial estimate on the master node from the partial estimates on local nodes */
        val step2MasterResult = algorithm.compute()

        step2MasterResult.pack()

        val sc = step1LocalResult.sparkContext

        /* Create deep copies of master result:
         * 1) Get serialized step2masterResult as byte array */
        val buffer = serializeObject(step2MasterResult)

        /* 2) Create broadcast value from byte array to avoid duplicate sending on nodes */
        val masterPartArray = sc.broadcast(buffer)

        /* 3) Create dummy list to create rdd with multiplied step2MasterResult objects */
        val list = new ArrayBuffer[Tuple2[Int, Int]]();
        val i: Int = 0
        for(i <- 0 to (nBlocks - 1)) {
            list += new Tuple2(i, 0)
        }

        /* 4) Create rdd with separate copy of step2MasterResult for every block */
        val rdd = sc.parallelize(list, nBlocks).mapValues(
            (masterRes: Int) => {
                val array = masterPartArray.value
                deserializeObject(array).asInstanceOf[DistributedPartialResultStep2]
            })

        contextLocal.dispose()

        rdd
    }

    private def computeStep3Local(
        offset:             RDD[Tuple2[Int, NumericTable]],
        partialResultLocal: RDD[Tuple2[Int, DistributedPartialResultStep4]],
        step3LocalInput:    RDD[Tuple2[Int, KeyValueDataCollection]])
        : RDD[Tuple2[Int, Tuple2[Int, PartialModel]]] = {

        val joined = offset.join(partialResultLocal.join(step3LocalInput))

        var bcNFactors = joined.sparkContext.broadcast(nFactors)
        var bcAlpha = joined.sparkContext.broadcast(alpha)
        var bcLambda = joined.sparkContext.broadcast(lambda)

        joined.flatMap(
            (tup: Tuple2[Int, Tuple2[NumericTable, Tuple2[DistributedPartialResultStep4, KeyValueDataCollection]]]) => {

                val contextLocal = new DaalContext()
                tup._2._1.unpack(contextLocal)
                tup._2._2._1.unpack(contextLocal)
                tup._2._2._2.unpack(contextLocal)

                val algorithm = new DistributedStep3Local(contextLocal, classOf[java.lang.Double], TrainingMethod.fastCSR)

                algorithm.parameter.setNFactors(bcNFactors.value)
                algorithm.parameter.setAlpha(bcAlpha.value)
                algorithm.parameter.setLambda(bcLambda.value)

                algorithm.input.set(PartialModelInputId.partialModel, tup._2._2._1.get(DistributedPartialResultStep4Id.outputOfStep4ForStep3))
                algorithm.input.set(Step3LocalCollectionInputId.partialModelBlocksToNode, tup._2._2._2)
                algorithm.input.set(Step3LocalNumericTableInputId.offset, tup._2._1)

                val partialResult = algorithm.compute()
                tup._2._1.pack()
                tup._2._2._1.pack()

                val collection = partialResult.get(DistributedPartialResultStep3Id.outputOfStep3ForStep4)

                val list = new ArrayBuffer[Tuple2[Int, Tuple2[Int, PartialModel]]]()
                var i: Int = 0
                for(i <- 0 to (collection.size().toInt - 1)) {
                    val partialModel = collection.getValueByIndex(i).asInstanceOf[PartialModel]
                    partialModel.pack()
                    val blockToIdWithTuple = new Tuple2(collection.getKeyByIndex(i).toInt, new Tuple2(tup._1, partialModel))
                    list += blockToIdWithTuple
                }

                contextLocal.dispose()

                list.iterator
            })
    }

    private def computeStep4Local(
        step2MasterResult: RDD[Tuple2[Int, DistributedPartialResultStep2]],
        step3LocalResult: RDD[Tuple2[Int, Tuple2[Int, PartialModel]]],
        dataRDD: RDD[Tuple2[Int, CSRNumericTable]])
        : RDD[Tuple2[Int, DistributedPartialResultStep4]] = {

        val rddToCompute = dataRDD.cogroup(step3LocalResult, step2MasterResult)
        var bcNFactors = rddToCompute.sparkContext.broadcast(nFactors)
        var bcAlpha = rddToCompute.sparkContext.broadcast(alpha)
        var bcLambda = rddToCompute.sparkContext.broadcast(lambda)

        rddToCompute.map(
            (tup: Tuple2[Int, Tuple3[Iterable[CSRNumericTable],
                                     Iterable[Tuple2[Int, PartialModel]],
                                     Iterable[DistributedPartialResultStep2]]]) => {

                val contextLocal = new DaalContext()
                val tuple = tup._2
                val dataTable = tuple._1.iterator.next
                dataTable.unpack(contextLocal)
                val step4LocalInput = new KeyValueDataCollection(contextLocal)
                for (item <- tuple._2) {
                    item._2.unpack(contextLocal)
                    step4LocalInput.set(item._1, item._2)
                }

                val inputOfStep4FromStep2Value = tuple._3.iterator.next
                inputOfStep4FromStep2Value.unpack(contextLocal)

                val algorithm = new DistributedStep4Local(contextLocal, classOf[java.lang.Double], TrainingMethod.fastCSR)
                algorithm.parameter.setNFactors(bcNFactors.value)
                algorithm.parameter.setAlpha(bcAlpha.value)
                algorithm.parameter.setLambda(bcLambda.value)

                algorithm.input.set(Step4LocalPartialModelsInputId.partialModels, step4LocalInput)
                algorithm.input.set(Step4LocalNumericTableInputId.partialData, dataTable)
                algorithm.input.set(Step4LocalNumericTableInputId.inputOfStep4FromStep2,
                                    inputOfStep4FromStep2Value.get(DistributedPartialResultStep2Id.outputOfStep2ForStep4))

                val partialResultLocal = algorithm.compute()

                val nt = partialResultLocal.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1).getFactors()
                partialResultLocal.pack()

                dataTable.pack()
                contextLocal.dispose()

                new Tuple2(tup._1, partialResultLocal)
            })
    }

    private def serializeObject(serializableObject: SerializableBase) : Array[Byte] = {
        /* Create an output stream to serialize the object */
        val outputByteStream = new ByteArrayOutputStream()

        /* Serialize the object into the output stream */
        val outputStream = new ObjectOutputStream(outputByteStream)
        outputStream.writeObject(serializableObject)

        /* Store the serialized data in an array */
        val buffer = outputByteStream.toByteArray()

        buffer
    }

    private def deserializeObject(buffer: Array[Byte]) : SerializableBase = {
        /* Create an input stream to deserialize the object from the array */
        val inputByteStream = new ByteArrayInputStream(buffer)
        val inputStream = new ObjectInputStream(inputByteStream)

        /* Create a numeric table object */
        val restoredObject = inputStream.readObject().asInstanceOf[SerializableBase]

        restoredObject
    }
}
