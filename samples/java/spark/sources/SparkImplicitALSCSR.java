/* file: SparkImplicitALSCSR.java */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 //  Content:
 //     Java sample of the implicit alternating least squares (ALS) algorithm in
 //     the distributed processing mode.
 //
 //     The program trains the implicit ALS model on a supplied training data
 //     set.
 ////////////////////////////////////////////////////////////////////////////////
 */

package DAAL;

import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;
import java.io.IOException;
import java.lang.ClassNotFoundException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.DoubleBuffer;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;
import org.apache.spark.broadcast.Broadcast;

import java.util.Iterator;
import java.util.Collections;
import java.util.Comparator;
import java.nio.IntBuffer;

import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;

import com.intel.daal.algorithms.implicit_als.*;
import com.intel.daal.algorithms.implicit_als.training.*;
import com.intel.daal.algorithms.implicit_als.training.init.*;
import com.intel.daal.algorithms.implicit_als.prediction.ratings.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class SparkImplicitALSCSR {

    public static class TrainingResult {
        JavaPairRDD<Integer, DistributedPartialResultStep4> itemsFactors;
        JavaPairRDD<Integer, DistributedPartialResultStep4> usersFactors;
    }

    static final int nUsers = 46;           /* Full number of users */
    static final int nItems = 21;           /* Full number of items */
    static final int nFactors = 2;          /* Number of factors */
    static final int maxIterations = 5;     /* Number of iterations of the implicit ALS training algorithm */

    public static TrainingResult trainModel(
        JavaSparkContext sc,
        JavaPairRDD<Integer, NumericTable> dataRDD
    )
    throws IOException, ClassNotFoundException {

        long[] usersPartition = { dataRDD.count() };

        JavaPairRDD<Integer, NumericTable> transposedDataRDD = null;

        JavaPairRDD<Integer, NumericTable> userOffset = null;
        JavaPairRDD<Integer, NumericTable> itemOffset = null;

        JavaPairRDD<Integer, KeyValueDataCollection> userStep3LocalInput = null;
        JavaPairRDD<Integer, KeyValueDataCollection> itemStep3LocalInput = null;

        JavaPairRDD<Integer, DistributedPartialResultStep4> usersPartialResultLocal = null;
        JavaPairRDD<Integer, DistributedPartialResultStep4> itemsPartialResultLocal = null;

        /* Initialize distributed implicit ALS model */
        JavaPairRDD<Integer, Tuple4<DistributedPartialResultStep4,
                                    KeyValueDataCollection,
                                    NumericTable,
                                    KeyValueDataCollection>> initStep1LocalResult = initializeStep1Local(dataRDD, usersPartition);
        JavaPairRDD<Integer, KeyValueDataCollection> initStep1LocalResultForStep2 = getRDD4Split(3, initStep1LocalResult, KeyValueDataCollection.class);

        JavaPairRDD<Integer, Iterable<Tuple2<Integer, NumericTable>>> initStep2LocalInput = getInputForStep2(initStep1LocalResultForStep2);

        JavaPairRDD<Integer, Tuple3<NumericTable,
                                    KeyValueDataCollection,
                                    NumericTable>> initStep2LocalResult = initializeStep2Local(initStep2LocalInput);

        itemsPartialResultLocal = getRDD4Split(0, initStep1LocalResult, DistributedPartialResultStep4.class);
        transposedDataRDD       = getRDD3Split(0, initStep2LocalResult, NumericTable.class);

        itemStep3LocalInput     = getRDD4Split(1, initStep1LocalResult, KeyValueDataCollection.class);
        userStep3LocalInput     = getRDD3Split(1, initStep2LocalResult, KeyValueDataCollection.class);

        userOffset              = getRDD4Split(2, initStep1LocalResult, NumericTable.class);
        itemOffset              = getRDD3Split(2, initStep2LocalResult, NumericTable.class);

        /* Train distributed implicit ALS model */
        JavaPairRDD<Integer, DistributedPartialResultStep2> step2MasterResultCopies = null;

        JavaPairRDD<Integer, DistributedPartialResultStep1> step1LocalResult = null;
        JavaPairRDD<Integer, Tuple2<Integer, PartialModel>> step3LocalResult = null;

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            step1LocalResult = computeStep1Local(itemsPartialResultLocal);
            step2MasterResultCopies = computeStep2Master(sc, step1LocalResult);
            step3LocalResult = computeStep3Local(itemOffset, itemsPartialResultLocal, itemStep3LocalInput);
            usersPartialResultLocal = computeStep4Local(step2MasterResultCopies, step3LocalResult, transposedDataRDD).cache();
            if(iteration != (maxIterations - 1)) {
                itemsPartialResultLocal.unpersist();
            }

            step1LocalResult = computeStep1Local(usersPartialResultLocal);
            step2MasterResultCopies = computeStep2Master(sc, step1LocalResult);
            step3LocalResult = computeStep3Local(userOffset, usersPartialResultLocal, userStep3LocalInput);
            itemsPartialResultLocal = computeStep4Local(step2MasterResultCopies, step3LocalResult, dataRDD).cache();
            if(iteration != (maxIterations - 1)) {
                usersPartialResultLocal.unpersist();
            }
        }
        TrainingResult result = new TrainingResult();
        result.itemsFactors = itemsPartialResultLocal.cache();
        result.usersFactors = usersPartialResultLocal.cache();
        return result;
    }

    public static JavaRDD<Tuple3<Integer, Integer, RatingsResult>> testModel(
        JavaPairRDD<Integer, DistributedPartialResultStep4> usersFactors,
        JavaPairRDD<Integer, DistributedPartialResultStep4> itemsFactors
    ) {
        /* Do a trick to avoid using cartesian function:
         * Need to get all pairs where the first  element is from the usersFactors rdd (U_1, U_2, ... , U_nBlocks) and
         *                             the second element is from the itemsFactors rdd (I_1, I_2, ... , I_nBlocks).
         * 1) Create an rdd with pairs <0, U_1>, <0, U_2>, ... , <0, U_nBlocks>
         * 2) Create an rdd with pairs <0, I_1>, <0, I_2>, ... , <0, I_nBlocks>
         * 3) Join users rdd with items rdd to get pairs <0, <U_1, I_1>>,
         *                                               <0, <U_1, I_2>>,
         *                                               <0, <U_2, I_1>>,
         *                                               ... ,
         *                                               <0, <U_nBlocks, I_nBlocks>>
        */

        /* 1) Create an rdd with pairs <0, U_1>, <0, U_2>, ... , <0, U_nBlocks>*/
        JavaPairRDD<Integer, Tuple2<Integer, DistributedPartialResultStep4>> usersFactorsWithEqualKey = usersFactors.mapToPair(
        new PairFunction<Tuple2<Integer, DistributedPartialResultStep4>, Integer, Tuple2<Integer, DistributedPartialResultStep4>>() {
            public Tuple2<Integer, Tuple2<Integer, DistributedPartialResultStep4>> call(Tuple2<Integer, DistributedPartialResultStep4> tup) {
                return new Tuple2<Integer, Tuple2<Integer, DistributedPartialResultStep4>>(0, tup);
            }
        });

        /* 2) Create an rdd with pairs <0, I_1>, <0, I_2>, ... , <0, I_nBlocks>*/
        JavaPairRDD<Integer, Tuple2<Integer, DistributedPartialResultStep4>> itemsFactorsWithEqualKey = itemsFactors.mapToPair(
        new PairFunction<Tuple2<Integer, DistributedPartialResultStep4>, Integer, Tuple2<Integer, DistributedPartialResultStep4>>() {
            public Tuple2<Integer, Tuple2<Integer, DistributedPartialResultStep4>> call(Tuple2<Integer, DistributedPartialResultStep4> tup) {
                return new Tuple2<Integer, Tuple2<Integer, DistributedPartialResultStep4>>(0, tup);
            }
        });

        /* 3) Join users rdd with items rdd to get pairs <0, <U_1, I_1>>,
         *                                               <0, <U_1, I_2>>,
         *                                               <0, <U_2, I_1>>,
         *                                               ... ,
         *                                               <0, <U_nBlocks, I_nBlocks>>
         */
        JavaPairRDD<Integer, Tuple2<Tuple2<Integer, DistributedPartialResultStep4>, Tuple2<Integer, DistributedPartialResultStep4>>> allPairs =
            usersFactorsWithEqualKey.join(itemsFactorsWithEqualKey);

        JavaRDD<Tuple3<Integer, Integer, RatingsResult>> predictedRatings =
            allPairs.map(new Function<Tuple2<Integer,
                         Tuple2<Tuple2<Integer, DistributedPartialResultStep4>,
                         Tuple2<Integer, DistributedPartialResultStep4>>>,
        Tuple3<Integer, Integer, RatingsResult>>() {
            public Tuple3<Integer, Integer, RatingsResult> call(
                Tuple2<Integer, Tuple2<Tuple2<Integer, DistributedPartialResultStep4>, Tuple2<Integer, DistributedPartialResultStep4>>> tup) {
                DaalContext context = new DaalContext();
                Tuple2<Tuple2<Integer, DistributedPartialResultStep4>, Tuple2<Integer, DistributedPartialResultStep4>> t = tup._2;

                DistributedPartialResultStep4 usersPartialResultLocal = t._1._2;
                usersPartialResultLocal.unpack(context);
                DistributedPartialResultStep4 itemsPartialResultLocal = t._2._2;
                itemsPartialResultLocal.unpack(context);

                RatingsDistributed algorithm = new RatingsDistributed(context, Double.class, RatingsMethod.defaultDense);
                algorithm.parameter.setNFactors(nFactors);

                algorithm.input.set(RatingsPartialModelInputId.usersPartialModel,
                                    usersPartialResultLocal.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1));
                algorithm.input.set(RatingsPartialModelInputId.itemsPartialModel,
                                    itemsPartialResultLocal.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1));

                algorithm.compute();
                RatingsResult result = algorithm.finalizeCompute();

                usersPartialResultLocal.pack();
                itemsPartialResultLocal.pack();
                result.pack();

                context.dispose();
                return new Tuple3<Integer, Integer, RatingsResult>(t._1._1, t._2._1, result);
            }
        });
        return predictedRatings;
    }

    static <T, T1, T2, T3> JavaPairRDD<Integer, T> getRDD3Split(
        final long i, JavaPairRDD<Integer, Tuple3<T1, T2, T3>> inputRDD, Class<T> type) {

        return inputRDD.mapToPair(
        new PairFunction<Tuple2<Integer, Tuple3<T1, T2, T3>>, Integer, T>() {
            public Tuple2<Integer, T> call(Tuple2<Integer, Tuple3<T1, T2, T3>> tup) {
                if      (i == 0) { return new Tuple2<Integer, T>(tup._1, (T)tup._2._1()); }
                else if (i == 1) { return new Tuple2<Integer, T>(tup._1, (T)tup._2._2()); }
                else if (i == 2) { return new Tuple2<Integer, T>(tup._1, (T)tup._2._3()); }
                else             { return new Tuple2<Integer, T>(tup._1, null);           }
            }
        });
    }

    static <T, T1, T2, T3, T4> JavaPairRDD<Integer, T> getRDD4Split(
        final long i, JavaPairRDD<Integer, Tuple4<T1, T2, T3, T4>> inputRDD, Class<T> type) {

        return inputRDD.mapToPair(
        new PairFunction<Tuple2<Integer, Tuple4<T1, T2, T3, T4>>, Integer, T>() {
            public Tuple2<Integer, T> call(Tuple2<Integer, Tuple4<T1, T2, T3, T4>> tup) {
                if      (i == 0) { return new Tuple2<Integer, T>(tup._1, (T)tup._2._1()); }
                else if (i == 1) { return new Tuple2<Integer, T>(tup._1, (T)tup._2._2()); }
                else if (i == 2) { return new Tuple2<Integer, T>(tup._1, (T)tup._2._3()); }
                else if (i == 3) { return new Tuple2<Integer, T>(tup._1, (T)tup._2._4()); }
                else             { return new Tuple2<Integer, T>(tup._1, null);           }
            }
        });
    }

    static JavaPairRDD<Integer, Tuple4<DistributedPartialResultStep4,
                                       KeyValueDataCollection,
                                       NumericTable,
                                       KeyValueDataCollection>>
    initializeStep1Local(JavaPairRDD<Integer, NumericTable> dataRDD, final long[] usersPartition) {
        return dataRDD.mapToPair(
        new PairFunction<Tuple2<Integer, NumericTable>,
            Integer, Tuple4<DistributedPartialResultStep4, KeyValueDataCollection, NumericTable, KeyValueDataCollection>>() {
            public Tuple2<Integer, Tuple4<DistributedPartialResultStep4, KeyValueDataCollection, NumericTable, KeyValueDataCollection>>
                call(Tuple2<Integer, NumericTable> tup) {
                DaalContext context = new DaalContext();
                tup._2.unpack(context);

                /* Create an algorithm object to initialize the implicit ALS model with the fastCSR method */
                InitDistributedStep1Local initAlgorithm = new InitDistributedStep1Local(context, Double.class, InitMethod.fastCSR);
                initAlgorithm.parameter.setFullNUsers(nUsers);
                initAlgorithm.parameter.setNFactors(nFactors);
                initAlgorithm.parameter.setSeed(initAlgorithm.parameter.getSeed() + tup._1);
                initAlgorithm.parameter.setPartition(new HomogenNumericTable(context, usersPartition, 1, usersPartition.length));

                /* Pass a training data set and dependent values to the algorithm */
                initAlgorithm.input.set(InitInputId.data, tup._2);

                /* Initialize the implicit ALS model */
                InitPartialResult initPartialResult = initAlgorithm.compute();

                PartialModel partialModel = initPartialResult.get(InitPartialResultId.partialModel);

                DistributedPartialResultStep4 partialResultStep4 = new DistributedPartialResultStep4(context);
                partialResultStep4.set(DistributedPartialResultStep4Id.outputOfStep4ForStep1, partialModel);

                KeyValueDataCollection step3LocalInput      = initPartialResult.get(InitPartialResultBaseId.outputOfInitForComputeStep3);
                NumericTable           offset               = initPartialResult.get(InitPartialResultBaseId.offsets, tup._1.longValue());
                KeyValueDataCollection initStep1LocalResult = initPartialResult.get(InitPartialResultCollectionId.outputOfStep1ForStep2);

                partialResultStep4.pack();
                step3LocalInput.pack();
                offset.pack();
                initStep1LocalResult.pack();
                tup._2.pack();

                Tuple4<DistributedPartialResultStep4, KeyValueDataCollection, NumericTable, KeyValueDataCollection> value =
                    new Tuple4<DistributedPartialResultStep4, KeyValueDataCollection, NumericTable, KeyValueDataCollection>(
                        partialResultStep4, step3LocalInput, offset, initStep1LocalResult);

                context.dispose();
                return new Tuple2<Integer, Tuple4<DistributedPartialResultStep4,
                                                  KeyValueDataCollection,
                                                  NumericTable,
                                                  KeyValueDataCollection>>(tup._1, value);
            }
        });
    }

    static JavaPairRDD<Integer, Tuple3<NumericTable,
                                       KeyValueDataCollection,
                                       NumericTable>>
    initializeStep2Local(JavaPairRDD<Integer, Iterable<Tuple2<Integer, NumericTable>>> inputOfStep2FromStep1) {
        return inputOfStep2FromStep1.mapToPair(
                   new PairFunction<Tuple2<Integer, Iterable<Tuple2<Integer, NumericTable>>>,
        Integer, Tuple3<NumericTable,
                        KeyValueDataCollection,
                        NumericTable>>() {
            public Tuple2<Integer, Tuple3<NumericTable,
                                          KeyValueDataCollection,
                                          NumericTable>> call(Tuple2<Integer, Iterable<Tuple2<Integer, NumericTable>>> tup) {

                DaalContext context = new DaalContext();
                KeyValueDataCollection initStep2LocalInput = new KeyValueDataCollection(context);

                for (Tuple2<Integer, NumericTable> item : tup._2) {
                    item._2.unpack(context);
                    initStep2LocalInput.set(item._1, item._2);
                }

                InitDistributedStep2Local initAlgorithm = new InitDistributedStep2Local(context, Double.class, InitMethod.fastCSR);

                initAlgorithm.input.set(InitStep2LocalInputId.inputOfStep2FromStep1, initStep2LocalInput);

                /* Compute partial results of the second step on local nodes */
                InitDistributedPartialResultStep2 initPartialResult = initAlgorithm.compute();

                NumericTable dataTableTransposed       = initPartialResult.get(InitDistributedPartialResultStep2Id.transposedData);
                KeyValueDataCollection step3LocalInput = initPartialResult.get(InitPartialResultBaseId.outputOfInitForComputeStep3);
                NumericTable offset                    = initPartialResult.get(InitPartialResultBaseId.offsets, tup._1);

                dataTableTransposed.pack();
                step3LocalInput.pack();
                offset.pack();

                Tuple3<NumericTable, KeyValueDataCollection, NumericTable> value =
                    new Tuple3<NumericTable, KeyValueDataCollection, NumericTable>(
                        dataTableTransposed, step3LocalInput, offset);

                return new Tuple2<Integer, Tuple3<NumericTable, KeyValueDataCollection, NumericTable>>(tup._1, value);
            }
        });
    }

    static JavaPairRDD<Integer, Iterable<Tuple2<Integer, NumericTable>>> getInputForStep2(
        JavaPairRDD<Integer, KeyValueDataCollection> initStep1LocalResultForStep2) {

        return initStep1LocalResultForStep2.flatMapToPair(
            new PairFlatMapFunction<Tuple2<Integer, KeyValueDataCollection>, Integer, Tuple2<Integer, NumericTable>>() {
            public Iterator<Tuple2<Integer, Tuple2<Integer, NumericTable>>>
            call(Tuple2<Integer, KeyValueDataCollection> tup) {

                DaalContext context = new DaalContext();
                KeyValueDataCollection collection = tup._2;
                collection.unpack(context);

                List<Tuple2<Integer, Tuple2<Integer, NumericTable>>> list = new LinkedList<Tuple2<Integer, Tuple2<Integer, NumericTable>>>();
                for(int i = 0; i < collection.size(); i++) {
                    NumericTable table = (NumericTable)collection.get(i);
                    table.pack();
                    Tuple2<Integer, NumericTable> blockFromIdWithTable = new Tuple2<Integer, NumericTable>(tup._1, table);
                    Tuple2<Integer, Tuple2<Integer, NumericTable>> blockToIdWithTuple =
                        new Tuple2<Integer, Tuple2<Integer, NumericTable>>((int)collection.getKeyByIndex(i), blockFromIdWithTable);
                    list.add(blockToIdWithTuple);
                }

                context.dispose();
                return list.iterator();
            }
        }).groupByKey();
    }

    static JavaPairRDD<Integer, DistributedPartialResultStep1> computeStep1Local(
        JavaPairRDD<Integer, DistributedPartialResultStep4> partialResultLocal) {
        return partialResultLocal.mapToPair(
        new PairFunction<Tuple2<Integer, DistributedPartialResultStep4>, Integer, DistributedPartialResultStep1>() {
            public Tuple2<Integer, DistributedPartialResultStep1> call(Tuple2<Integer, DistributedPartialResultStep4> tup) {
                DaalContext context = new DaalContext();
                tup._2.unpack(context);

                /* Create algorithm objects to compute a implisit ALS algorithm in the distributed processing mode using the fastCSR method */
                DistributedStep1Local algorithm = new DistributedStep1Local(context, Double.class, TrainingMethod.fastCSR);
                algorithm.parameter.setNFactors(nFactors);

                /* Set input objects for the algorithm */
                algorithm.input.set(PartialModelInputId.partialModel,
                                    tup._2.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1));

                /* Compute partial estimates on local nodes */
                DistributedPartialResultStep1 step1LocalResult = algorithm.compute();

                step1LocalResult.pack();
                tup._2.pack();

                context.dispose();
                return new Tuple2<Integer, DistributedPartialResultStep1>(tup._1, step1LocalResult);
            }
        });
    }

    static JavaPairRDD<Integer, DistributedPartialResultStep2> computeStep2Master(
        JavaSparkContext sc,
        JavaPairRDD<Integer, DistributedPartialResultStep1> step1LocalResult)
    throws IOException, ClassNotFoundException {
        DaalContext context = new DaalContext();

        List<Tuple2<Integer, DistributedPartialResultStep1>> step1LocalResultList = step1LocalResult.collect();

        /* Create algorithm objects to compute a implisit ALS algorithm in the distributed processing mode using the fastCSR method */
        DistributedStep2Master algorithm = new DistributedStep2Master(context, Double.class, TrainingMethod.fastCSR);
        algorithm.parameter.setNFactors(nFactors);
        int nBlocks = (int)step1LocalResultList.size();

        /* Set input objects for the algorithm */
        for (int i = 0; i < nBlocks; i++) {
            step1LocalResultList.get(i)._2.unpack(context);
            algorithm.input.add(MasterInputId.inputOfStep2FromStep1, step1LocalResultList.get(i)._2);
        }

        /* Compute a partial estimate on the master node from the partial estimates on local nodes */
        DistributedPartialResultStep2 step2MasterResult = algorithm.compute();

        step2MasterResult.pack();

        /* Create deep copies of master result:
         * 1) Get serialized step2masterResult as byte array */
        byte[] buffer = serializeObject(step2MasterResult);

        /* 2) Create broadcast value from byte array to avoid duplicate sending on nodes */
        final Broadcast<byte[]> masterPartArray = sc.broadcast(buffer);

        /* 3) Create dummy list to create rdd with multiplied step2MasterResult objects */
        List<Tuple2<Integer, Integer>> list = new ArrayList<Tuple2<Integer, Integer>>(nBlocks);
        for(int i = 0; i < nBlocks; i++) {
            list.add(new Tuple2<Integer, Integer>(i, 0));
        }

        /* 4) Create rdd with separate copy of step2MasterResult for every block */
        JavaPairRDD<Integer, DistributedPartialResultStep2> rdd = sc.parallelizePairs(list, nBlocks).mapValues(
        new Function<Integer, DistributedPartialResultStep2>() {
            public DistributedPartialResultStep2 call(Integer masterRes) throws IOException, ClassNotFoundException {
                byte[] array = masterPartArray.value();
                return (DistributedPartialResultStep2)deserializeObject(array);
            }
        });

        context.dispose();
        return rdd;
    }

    static JavaPairRDD<Integer, Tuple2<Integer, PartialModel>> computeStep3Local(
        JavaPairRDD<Integer, NumericTable> offset,
        JavaPairRDD<Integer, DistributedPartialResultStep4> partialResultLocal,
        JavaPairRDD<Integer, KeyValueDataCollection> step3LocalInput) {

        JavaPairRDD<Integer, Tuple2<NumericTable, Tuple2<DistributedPartialResultStep4, KeyValueDataCollection>>> joined = offset.join(partialResultLocal.join(step3LocalInput));

        return joined.flatMapToPair(
                   new PairFlatMapFunction<Tuple2<Integer, Tuple2<NumericTable, Tuple2<DistributedPartialResultStep4, KeyValueDataCollection>>>,
        Integer, Tuple2<Integer, PartialModel>>() {
            public Iterator<Tuple2<Integer, Tuple2<Integer, PartialModel>>> call(
                Tuple2<Integer, Tuple2<NumericTable, Tuple2<DistributedPartialResultStep4, KeyValueDataCollection>>> tup) {
                DaalContext context = new DaalContext();
                tup._2._1.unpack(context);
                tup._2._2._1.unpack(context);
                tup._2._2._2.unpack(context);

                DistributedStep3Local algorithm = new DistributedStep3Local(context, Double.class, TrainingMethod.fastCSR);
                algorithm.parameter.setNFactors(nFactors);

                algorithm.input.set(PartialModelInputId.partialModel, tup._2._2._1.get(DistributedPartialResultStep4Id.outputOfStep4ForStep3));
                algorithm.input.set(Step3LocalCollectionInputId.partialModelBlocksToNode, tup._2._2._2);
                algorithm.input.set(Step3LocalNumericTableInputId.offset, tup._2._1);

                DistributedPartialResultStep3 partialResult = algorithm.compute();
                tup._2._1.pack();
                tup._2._2._1.pack();

                KeyValueDataCollection collection = partialResult.get(DistributedPartialResultStep3Id.outputOfStep3ForStep4);

                List<Tuple2<Integer, Tuple2<Integer, PartialModel>>> list = new LinkedList<Tuple2<Integer, Tuple2<Integer, PartialModel>>>();
                for(int i = 0; i < collection.size(); i++) {
                    PartialModel partialModel = (PartialModel)collection.getValueByIndex(i);
                    partialModel.pack();
                    Tuple2<Integer, PartialModel> blockFromIdWithModel = new Tuple2<Integer, PartialModel>(tup._1, partialModel);
                    Tuple2<Integer, Tuple2<Integer, PartialModel>> blockToIdWithTuple =
                        new Tuple2<Integer, Tuple2<Integer, PartialModel>>((int)collection.getKeyByIndex(i), blockFromIdWithModel);
                    list.add(blockToIdWithTuple);
                }

                context.dispose();
                return list.iterator();
            }
        });
    }

    static JavaPairRDD<Integer, DistributedPartialResultStep4> computeStep4Local(
        JavaPairRDD<Integer, DistributedPartialResultStep2> step2MasterResult,
        JavaPairRDD<Integer, Tuple2<Integer, PartialModel>> step3LocalResult,
        JavaPairRDD<Integer, NumericTable> dataRDD) {

        JavaPairRDD<Integer, Tuple3<Iterable<NumericTable>,
                                    Iterable<Tuple2<Integer, PartialModel>>,
                                    Iterable<DistributedPartialResultStep2>>> rddToCompute = dataRDD.cogroup(step3LocalResult, step2MasterResult);

        return rddToCompute.mapToPair(
                   new PairFunction<Tuple2<Integer, Tuple3<Iterable<NumericTable>,
                                                           Iterable<Tuple2<Integer, PartialModel>>,
                                                           Iterable<DistributedPartialResultStep2>>>,
        Integer, DistributedPartialResultStep4>() {
            public Tuple2<Integer, DistributedPartialResultStep4> call(
                Tuple2<Integer, Tuple3<Iterable<NumericTable>,
                                       Iterable<Tuple2<Integer, PartialModel>>,
                                       Iterable<DistributedPartialResultStep2>>> tup) {
                DaalContext context = new DaalContext();
                Tuple3<Iterable<NumericTable>,
                       Iterable<Tuple2<Integer, PartialModel>>,
                       Iterable<DistributedPartialResultStep2>> tuple = tup._2;
                NumericTable dataTable = tuple._1().iterator().next();
                dataTable.unpack(context);
                KeyValueDataCollection step4LocalInput = new KeyValueDataCollection(context);
                for (Tuple2<Integer, PartialModel> item : tuple._2()) {
                    item._2.unpack(context);
                    step4LocalInput.set(item._1, item._2);
                }

                DistributedPartialResultStep2 inputOfStep4FromStep2Value = tuple._3().iterator().next();
                inputOfStep4FromStep2Value.unpack(context);

                DistributedStep4Local algorithm = new DistributedStep4Local(context, Double.class, TrainingMethod.fastCSR);
                algorithm.parameter.setNFactors(nFactors);

                algorithm.input.set(Step4LocalPartialModelsInputId.partialModels, step4LocalInput);
                algorithm.input.set(Step4LocalNumericTableInputId.partialData, dataTable);
                algorithm.input.set(Step4LocalNumericTableInputId.inputOfStep4FromStep2,
                                    inputOfStep4FromStep2Value.get(DistributedPartialResultStep2Id.outputOfStep2ForStep4));

                DistributedPartialResultStep4 partialResultLocal = algorithm.compute();

                NumericTable nt = partialResultLocal.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1).getFactors();
                partialResultLocal.pack();

                dataTable.pack();
                context.dispose();
                return new Tuple2<Integer, DistributedPartialResultStep4>(tup._1, partialResultLocal);
            }
        });
    }

    public static Long[] computePartition(JavaPairRDD<Integer, NumericTable> dataRDD) {
        JavaPairRDD<Integer, Long> numbersOfRowsRDD = dataRDD.mapToPair(
        new PairFunction<Tuple2<Integer, NumericTable>, Integer, Long>() {
            public Tuple2<Integer, Long> call(Tuple2<Integer, NumericTable> tup) {
                return new Tuple2<Integer, Long>(tup._1, tup._2.getNumberOfRows());
            }
        });

        List<Tuple2<Integer, Long>> unmodifiableNumbersOfRows = numbersOfRowsRDD.collect();
        Comparator<Tuple2<Integer, Long>> comparator = new Comparator<Tuple2<Integer, Long>>() {
            public int compare(Tuple2<Integer, Long> tupleA,
                               Tuple2<Integer, Long> tupleB) {
                return tupleA._1.compareTo(tupleB._1);
            }
        };

        List<Tuple2<Integer, Long>> numbersOfRows = new ArrayList<Tuple2<Integer, Long>>(unmodifiableNumbersOfRows);

        Collections.sort(numbersOfRows, comparator);

        Long[] partition = new Long[numbersOfRows.size() + 1];
        partition[0] = new Long(0);
        for(int i = 0; i < partition.length - 1; i++) {
            partition[i + 1] = partition[i] + numbersOfRows.get(i)._2;
        }
        return partition;
    }

    public static byte[] serializeObject(SerializableBase serializableObject) throws IOException {
        /* Create an output stream to serialize the object */
        ByteArrayOutputStream outputByteStream = new ByteArrayOutputStream();

        /* Serialize the object into the output stream */
        ObjectOutputStream outputStream = new ObjectOutputStream(outputByteStream);
        outputStream.writeObject(serializableObject);

        /* Store the serialized data in an array */
        byte[] buffer = outputByteStream.toByteArray();
        return buffer;
    }

    public static SerializableBase deserializeObject(byte[] buffer) throws IOException, ClassNotFoundException {
        /* Create an input stream to deserialize the object from the array */
        ByteArrayInputStream inputByteStream = new ByteArrayInputStream(buffer);
        ObjectInputStream inputStream = new ObjectInputStream(inputByteStream);

        /* Create a numeric table object */
        SerializableBase restoredDataTable = (SerializableBase)inputStream.readObject();

        return restoredDataTable;
    }
}
