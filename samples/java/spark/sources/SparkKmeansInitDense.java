/* file: SparkKmeansInitDense.java */
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
//      Java sample of K-Means clustering in the distributed processing mode
////////////////////////////////////////////////////////////////////////////////
*/

package DAAL;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;
import org.apache.spark.broadcast.Broadcast;

import java.lang.ClassNotFoundException;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import scala.Tuple2;
import scala.Tuple3;

import com.intel.daal.algorithms.kmeans.*;
import com.intel.daal.algorithms.kmeans.init.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class SparkKmeansInitDense {
    /* Classes containing the algorithm results */
    static class KmeansInitResult {
        public NumericTable centroids;
    }
    static class KmeansResult {
        public NumericTable centroids;
    }

    private static final long nClusters      = 20;
    private static final int nBlocks         = 4;
    private static final int nIterations     = 5;
    private static final int nVectorsInBlock = 2500;

    public static KmeansInitResult initKmeansPlusPlus(JavaSparkContext sc, DaalContext context, JavaPairRDD<Integer, HomogenNumericTable> dataRDD) throws IOException, ClassNotFoundException {
        /* Numeric table to collect the results */
        RowMergedNumericTable centroids = new RowMergedNumericTable(context);
        final boolean isPlusPlus = true;
        final boolean isForStep5 = false; /* can be true for parallelPlus only */

        /* Perform step 1 */
        JavaRDD<NumericTable> newCentroidsRDD = initStep1Local(dataRDD, isPlusPlus);

        /* Get table that is not null */
        List<NumericTable> newCentroidsList   = newCentroidsRDD.collect();
        NumericTable newCentroids = null;
        for (NumericTable value : newCentroidsList) {
            if(value != null) {
                value.unpack(context);
                centroids.addNumericTable(value);
                newCentroids = value;
            }
        }
        newCentroids.pack();

        /* Create an algorithm object for the step 3 */
        InitDistributedStep3Master step3 = new InitDistributedStep3Master(context, Double.class, InitMethod.plusPlusDense, nClusters);

        /* Perform 1st iteration of step 2 to get localNodeData */
        JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> localNodeDataAndResultFrom2To3RDD = initStep2LocalFirst(newCentroids, dataRDD, isPlusPlus);
        /* Perform steps 3 and 4 first time */
        newCentroids = performSteps34PlusPlus(sc, context, step3, localNodeDataAndResultFrom2To3RDD, dataRDD, centroids, isPlusPlus);

        for (int iCentroid = 2; iCentroid < nClusters; iCentroid++) {
            newCentroids.pack();
            /* Perform step 2  */
            localNodeDataAndResultFrom2To3RDD = initStep2Local(localNodeDataAndResultFrom2To3RDD, newCentroids, dataRDD, isForStep5, isPlusPlus);
            /* Perform steps 3 and 4 */
            newCentroids = performSteps34PlusPlus(sc, context, step3, localNodeDataAndResultFrom2To3RDD, dataRDD, centroids, isPlusPlus);
        }

        KmeansInitResult initResult = new KmeansInitResult();
        initResult.centroids = centroids;

        return initResult;
    }

    public static KmeansInitResult initKmeansParallelPlus(JavaSparkContext sc, DaalContext context, JavaPairRDD<Integer, HomogenNumericTable> dataRDD) throws IOException, ClassNotFoundException {
        final boolean isPlusPlus = false;
        boolean isForStep5 = false;
        /* Perform step 1 */
        JavaRDD<NumericTable> newCentroidsRDD = initStep1Local(dataRDD, isPlusPlus);

        /* Get table that is not null */
        List<NumericTable> newCentroidsList = newCentroidsRDD.collect();
        NumericTable newCentroids = null;
        for(NumericTable value : newCentroidsList) {
            if(value != null) {
                value.unpack(context);
                newCentroids = value;
            }
        }
        /* Create an algorithm object for the step 3 */
        InitDistributedStep3Master step3 = new InitDistributedStep3Master(context, Double.class, InitMethod.parallelPlusDense, nClusters);

        /* Create an algorithm object for the step 5 */
        InitDistributedStep5Master step5 = new InitDistributedStep5Master(context, Double.class, InitMethod.parallelPlusDense, nClusters);
        step5.input.add(InitDistributedStep5MasterPlusPlusInputId.inputCentroids, newCentroids);

        newCentroids.pack();
        /* Perform 1st iteration of step 2 to get localNodeData */
        JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> localNodeDataAndResultFrom2To3RDD = initStep2LocalFirst(newCentroids, dataRDD, isPlusPlus);

        int iRound = 0;
        /* Perform steps 3 and 4 first time */
        newCentroids = performSteps34ParallelPlus(sc, context, step3, step5, localNodeDataAndResultFrom2To3RDD, dataRDD, iRound, isPlusPlus);

        for(iRound = 1; iRound < step5.parameter.getNRounds(); iRound++) {
            newCentroids.pack();
            /* Perform step 2  */
            localNodeDataAndResultFrom2To3RDD = initStep2Local(localNodeDataAndResultFrom2To3RDD, newCentroids, dataRDD, isForStep5, isPlusPlus);
            /* Perform steps 3 and 4 */
            newCentroids = performSteps34ParallelPlus(sc, context, step3, step5, localNodeDataAndResultFrom2To3RDD, dataRDD, iRound, isPlusPlus);
        }

        newCentroids.pack();
        isForStep5 = true;
        JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> newRes = initStep2Local(localNodeDataAndResultFrom2To3RDD, newCentroids, dataRDD, isForStep5, isPlusPlus);

        List<Tuple2<Integer, Tuple2<DataCollection, NumericTable>>> localNodeDataAndResultFrom2To3List = newRes.collect();
        for(Tuple2<Integer, Tuple2<DataCollection, NumericTable>> value : localNodeDataAndResultFrom2To3List) {
            if(value._2._2 != null) {
                value._2._2.unpack(context);
                step5.input.add(InitDistributedStep5MasterPlusPlusInputId.inputOfStep5FromStep2, value._2._2);
            }
        }

        step5.compute();

        KmeansInitResult initResult = new KmeansInitResult();
        initResult.centroids = step5.finalizeCompute().get(InitResultId.centroids);

        return initResult;
    }

    public static KmeansResult runKmeans(DaalContext context, JavaPairRDD<Integer, HomogenNumericTable> dataRDD, NumericTable initCentroids) {
        initCentroids.pack();
        NumericTable resultCentroids = null;

        for(int it = 0; it < nIterations; it++) {
            JavaRDD<PartialResult> partResRDD = computeLocal(context, dataRDD, initCentroids);
            resultCentroids  = computeMaster(context, partResRDD);
        }

        initCentroids.unpack(context);
        KmeansResult result = new KmeansResult();
        result.centroids = resultCentroids;

        return result;
    }

    public static NumericTable performSteps34PlusPlus(JavaSparkContext sc,
                                                      DaalContext context,
                                                      InitDistributedStep3Master step3,
                                                      JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> localNodeDataAndResultFrom2To3RDD,
                                                      JavaPairRDD<Integer, HomogenNumericTable> dataRDD,
                                                      RowMergedNumericTable centroids,
                                                      final boolean isPlusPlus) throws IOException, ClassNotFoundException {
        /* Perform step 3 */
        JavaPairRDD<Integer, InitDistributedStep3MasterPlusPlusPartialResult> step3Pres = initStep3Master(sc, context, step3, localNodeDataAndResultFrom2To3RDD, isPlusPlus);

        /* Perform step 4 */
        JavaRDD<NumericTable> outputOfStep4 = initStep4Local(localNodeDataAndResultFrom2To3RDD, step3Pres, dataRDD, isPlusPlus);

        /* Get table that is not null */
        NumericTable newCentroids = null;
        List<NumericTable> outputOfStep4List = outputOfStep4.collect();
        for (NumericTable value : outputOfStep4List) {
            if(value != null) {
                value.unpack(context);
                centroids.addNumericTable(value);
                newCentroids = value;
            }
        }
        return newCentroids;
    }

    public static NumericTable performSteps34ParallelPlus(JavaSparkContext sc,
                                                          DaalContext context,
                                                          InitDistributedStep3Master step3,
                                                          InitDistributedStep5Master step5,
                                                          JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> localNodeDataAndResultFrom2To3RDD,
                                                          JavaPairRDD<Integer, HomogenNumericTable> dataRDD,
                                                          int iRound,
                                                          final boolean isPlusPlus) throws IOException, ClassNotFoundException {
        /* Perform step 3 */
        JavaPairRDD<Integer, InitDistributedStep3MasterPlusPlusPartialResult> step3Pres = initStep3Master(sc, context, step3, localNodeDataAndResultFrom2To3RDD, isPlusPlus);

        if(iRound + 1 == step5.parameter.getNRounds()) {
            List<Tuple2<Integer, InitDistributedStep3MasterPlusPlusPartialResult>> step3PresList = step3Pres.collect();
            InitDistributedStep3MasterPlusPlusPartialResult pres = step3PresList.get(0)._2;
            pres.unpack(context);
            SerializableBase inputOfStep5FromStep3 = pres.get(InitDistributedStep3MasterPlusPlusPartialResultDataId.outputOfStep3ForStep5);
            step5.input.set(InitDistributedStep5MasterPlusPlusInputDataId.inputOfStep5FromStep3, inputOfStep5FromStep3);
            pres.pack();
        }

        /* Perform step 4 */
        JavaRDD<NumericTable> outputOfStep4 = initStep4Local(localNodeDataAndResultFrom2To3RDD, step3Pres, dataRDD, isPlusPlus);

        /* Get table that is not null */
        RowMergedNumericTable newCentroids = new RowMergedNumericTable(context);
        List<NumericTable> outputOfStep4List = outputOfStep4.collect();
        for (NumericTable value : outputOfStep4List) {
            if(value != null) {
                value.unpack(context);
                newCentroids.addNumericTable(value);
            }
        }

        step5.input.add(InitDistributedStep5MasterPlusPlusInputId.inputCentroids, newCentroids);

        return newCentroids;
    }

    private static JavaRDD<NumericTable> initStep1Local(JavaPairRDD<Integer, HomogenNumericTable> dataRDD,
                                                        final boolean isPlusPlus) {
        return dataRDD.map(new Function<Tuple2<Integer, HomogenNumericTable>, NumericTable>() {
            public NumericTable call(Tuple2<Integer, HomogenNumericTable> tup) {
                DaalContext localContext = new DaalContext();
                NumericTable  dataTable = tup._2;
                dataTable.unpack(localContext);
                InitMethod method = (isPlusPlus == true) ? InitMethod.plusPlusDense : InitMethod.parallelPlusDense;

                /* Create an algorithm to initialize the K-Means algorithm on local nodes */
                InitDistributedStep1Local step1Local = new InitDistributedStep1Local(localContext, Double.class, method, nClusters,
                                                                                     nBlocks * nVectorsInBlock, nVectorsInBlock * tup._1);
                /* Set the input data on local nodes */
                step1Local.input.set(InitInputId.data, dataTable);

                /* Compute K-Means initialization on local nodes */
                InitPartialResult initPres = step1Local.compute();
                NumericTable pNewCenters = initPres.get(InitPartialResultId.partialCentroids);

                dataTable.pack();
                if(pNewCenters != null) {
                    pNewCenters.pack();
                }

                localContext.dispose();
                return pNewCenters;
            }
        });
    }

    private static JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> initStep2LocalFirst(final NumericTable step2Input,
                                                                                                  JavaPairRDD<Integer, HomogenNumericTable> dataRDD,
                                                                                                  final boolean isPlusPlus) {
        return dataRDD.mapToPair(new PairFunction<Tuple2<Integer, HomogenNumericTable>, Integer, Tuple2<DataCollection, NumericTable>>() {
            public Tuple2<Integer, Tuple2<DataCollection, NumericTable>> call(Tuple2<Integer, HomogenNumericTable> tup) {
                DaalContext localContext = new DaalContext();
                InitMethod method = (isPlusPlus == true) ? InitMethod.plusPlusDense : InitMethod.parallelPlusDense;
                final boolean isFirst = true;

                NumericTable  dataTable = tup._2;

                dataTable.unpack(localContext);
                step2Input.unpack(localContext);

                /* Create an algorithm object for the step 2 */
                InitDistributedStep2Local step2Local = new InitDistributedStep2Local(localContext, Double.class, method, nClusters, isFirst);

                /* Set the input data to the algorithm */
                step2Local.input.set(InitInputId.data, dataTable);
                step2Local.input.set(InitDistributedStep2LocalPlusPlusInputId.inputOfStep2, step2Input);

                /* Compute and get the result */
                InitDistributedStep2LocalPlusPlusPartialResult initPres = step2Local.compute();
                DataCollection localNodeData = initPres.get(InitDistributedStep2LocalPlusPlusPartialResultDataId.internalResult);
                NumericTable outputOfStep2ForStep3 = initPres.get(InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep3);

                if(outputOfStep2ForStep3 != null) {
                    outputOfStep2ForStep3.pack();
                }

                step2Input.pack();
                localNodeData.pack();
                dataTable.pack();

                localContext.dispose();

                Tuple2<DataCollection, NumericTable> tuple = new Tuple2<DataCollection, NumericTable>(localNodeData, outputOfStep2ForStep3);
                return new Tuple2<Integer, Tuple2<DataCollection, NumericTable>>(tup._1, tuple);
            }
        });
    }

    private static JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> initStep2Local(JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> localNodeData,
                                                                                             final NumericTable step2Input,
                                                                                             JavaPairRDD<Integer, HomogenNumericTable> dataRDD,
                                                                                             final boolean isForStep5,
                                                                                             final boolean isPlusPlus) {
        /* Cogroup three RDDs */
        JavaPairRDD<Integer, Tuple2<Iterable<HomogenNumericTable>, Iterable<Tuple2<DataCollection, NumericTable>>> > groupedData = dataRDD.cogroup(localNodeData);

        return groupedData.mapToPair(new PairFunction<Tuple2<Integer, Tuple2<Iterable<HomogenNumericTable>, Iterable<Tuple2<DataCollection, NumericTable>>>>, Integer, Tuple2<DataCollection, NumericTable>>() {
            public Tuple2<Integer, Tuple2<DataCollection, NumericTable>> call(Tuple2<Integer, Tuple2<Iterable<HomogenNumericTable>, Iterable<Tuple2<DataCollection, NumericTable>>>> tup) {
                DaalContext localContext = new DaalContext();
                InitMethod method = (isPlusPlus == true) ? InitMethod.plusPlusDense : InitMethod.parallelPlusDense;
                final boolean isFirst = false;

                NumericTable  dataTable = tup._2._1().iterator().next();
                Tuple2<DataCollection, NumericTable> tuple = tup._2._2().iterator().next();
                DataCollection localNodeData = tuple._1;

                dataTable.unpack(localContext);
                localNodeData.unpack(localContext);
                step2Input.unpack(localContext);

                /* Create an algorithm object for the step 2 */
                InitDistributedStep2Local step2Local = new InitDistributedStep2Local(localContext, Double.class, method, nClusters, isFirst);

                /* Set the input data to the algorithm */
                step2Local.input.set(InitInputId.data, dataTable);
                step2Local.input.set(InitDistributedStep2LocalPlusPlusInputId.inputOfStep2, step2Input);
                step2Local.input.set(InitDistributedLocalPlusPlusInputDataId.internalInput, localNodeData);

                /* For ParallelPlusPlus */
                if(isForStep5) {
                    step2Local.parameter.setOutputForStep5Required(true);
                }

                /* Compute and get the result */
                InitDistributedStep2LocalPlusPlusPartialResult initPres = step2Local.compute();

                NumericTable resultTable = null;
                if(!isForStep5) {
                    resultTable = initPres.get(InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep3);
                } else {
                    resultTable = initPres.get(InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep5);
                }

                if(resultTable != null) {
                    resultTable.pack();
                }

                step2Input.pack();
                localNodeData.pack();
                dataTable.pack();

                localContext.dispose();

                Tuple2<DataCollection, NumericTable> rtuple = new Tuple2<DataCollection, NumericTable>(localNodeData, resultTable);
                return new Tuple2<Integer, Tuple2<DataCollection, NumericTable>>(tup._1, rtuple);
            }
        });
    }

    public static JavaPairRDD<Integer, InitDistributedStep3MasterPlusPlusPartialResult> initStep3Master(JavaSparkContext sc,
                                                                                                        DaalContext context,
                                                                                                        InitDistributedStep3Master step3,
                                                                                                        JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> resultFrom2To3RDD,
                                                                                                        final boolean isPlusPlus) throws IOException, ClassNotFoundException {
        InitMethod method = (isPlusPlus == true) ? InitMethod.plusPlusDense : InitMethod.parallelPlusDense;

        /* Set the input data to the algorithm */
        List<Tuple2<Integer, Tuple2<DataCollection, NumericTable>>> resultFrom2To3List = resultFrom2To3RDD.collect();
        for (Tuple2<Integer, Tuple2<DataCollection, NumericTable>> value : resultFrom2To3List) {
            if(value._2._2 != null) {
                value._2._2.unpack(context);
                step3.input.add(InitDistributedStep3MasterPlusPlusInputId.inputOfStep3FromStep2, value._1, value._2._2);
            }
        }

        /* Compute and get the result */
        InitDistributedStep3MasterPlusPlusPartialResult step3Pres = step3.compute();
        step3Pres.pack();

        /* Create deep copies of master result:
         * 1) Get serialized step3Pres as byte array */
        byte[] buffer = serializeObject(step3Pres);

        /* 2) Create broadcast value from byte array to avoid duplicate sending on nodes */
        final Broadcast<byte[]> step3PresArray = sc.broadcast(buffer);

        /* 3) Create dummy list to create rdd with multiplied step3Pres objects */
        List<Tuple2<Integer, Integer>> list = new ArrayList<Tuple2<Integer, Integer>>(nBlocks);
        for(int i = 0; i < nBlocks; i++) {
            list.add(new Tuple2<Integer, Integer>(i, 0));
        }

        /* 4) Create rdd with separate copy of step3Pres for every block */
        JavaPairRDD<Integer, InitDistributedStep3MasterPlusPlusPartialResult> step3PresRDD = sc.parallelizePairs(list, nBlocks).mapValues(
        new Function<Integer, InitDistributedStep3MasterPlusPlusPartialResult>() {
            public InitDistributedStep3MasterPlusPlusPartialResult call(Integer masterRes) throws IOException, ClassNotFoundException {
                byte[] array = step3PresArray.value();
                return (InitDistributedStep3MasterPlusPlusPartialResult)deserializeObject(array);
            }
        });

        return step3PresRDD;
    }

    private static JavaRDD<NumericTable> initStep4Local(JavaPairRDD<Integer, Tuple2<DataCollection, NumericTable>> localNodeData,
                                                        JavaPairRDD<Integer, InitDistributedStep3MasterPlusPlusPartialResult> step3Pres,
                                                        JavaPairRDD<Integer, HomogenNumericTable> dataRDD,
                                                        final boolean isPlusPlus) {
        /* Cogroup three RDDs */
        JavaPairRDD<Integer, Tuple3<Iterable<HomogenNumericTable>, Iterable<Tuple2<DataCollection, NumericTable>>, Iterable<InitDistributedStep3MasterPlusPlusPartialResult>>> groupedData = dataRDD.cogroup(localNodeData, step3Pres);
        return groupedData.map(new Function<Tuple2<Integer, Tuple3<Iterable<HomogenNumericTable>, Iterable<Tuple2<DataCollection, NumericTable>>, Iterable<InitDistributedStep3MasterPlusPlusPartialResult>>>, NumericTable>() {
                    public NumericTable call(Tuple2<Integer, Tuple3<Iterable<HomogenNumericTable>, Iterable<Tuple2<DataCollection, NumericTable>>, Iterable<InitDistributedStep3MasterPlusPlusPartialResult>>> tup) {
                DaalContext localContext = new DaalContext();
                InitMethod method = (isPlusPlus == true) ? InitMethod.plusPlusDense : InitMethod.parallelPlusDense;

                Tuple2<DataCollection, NumericTable> tuple                = tup._2._2().iterator().next();
                DataCollection localNodeData                              = tuple._1;
                NumericTable   dataTable                                  = tup._2._1().iterator().next();
                InitDistributedStep3MasterPlusPlusPartialResult step3Pres = tup._2._3().iterator().next();

                localNodeData.unpack(localContext);
                dataTable.unpack(localContext);
                step3Pres.unpack(localContext);

                /* Get an input for step 4 on this node if any */
                NumericTable step3Output = step3Pres.get(InitDistributedStep3MasterPlusPlusPartialResultId.outputOfStep3ForStep4, tup._1);
                if(step3Output == null) {
                    return null; /* can be null */
                }

                /* Create an algorithm object for the step 4 */
                InitDistributedStep4Local step4 = new InitDistributedStep4Local(localContext, Double.class, method, nClusters);

                /* Set the input data to the algorithm */
                step4.input.set(InitInputId.data, dataTable);
                step4.input.set(InitDistributedLocalPlusPlusInputDataId.internalInput, localNodeData);
                step4.input.set(InitDistributedStep4LocalPlusPlusInputId.inputOfStep4FromStep3, step3Output);

                /* Compute and get the result */
                NumericTable outputOfStep4 = step4.compute().get(InitDistributedStep4LocalPlusPlusPartialResultId.outputOfStep4);

                dataTable.pack();
                step3Pres.pack();
                localNodeData.pack();
                outputOfStep4.pack();

                return outputOfStep4;
            }
        });
    }

    private static JavaRDD<PartialResult> computeLocal(DaalContext context,
                                                       JavaPairRDD<Integer, HomogenNumericTable> dataRDD,
                                                       final NumericTable centroids) {
        return dataRDD.map(new Function<Tuple2<Integer, HomogenNumericTable>, PartialResult>() {
            public PartialResult call(Tuple2<Integer, HomogenNumericTable> tup) {
                DaalContext localContext = new DaalContext();

                /* Create an algorithm to compute k-means on local nodes */
                DistributedStep1Local kmeansLocal = new DistributedStep1Local(localContext, Double.class, Method.defaultDense, nClusters);

                tup._2.unpack(localContext);
                centroids.unpack(localContext);

                /* Set the input data on local nodes */
                kmeansLocal.input.set(InputId.data, tup._2);
                kmeansLocal.input.set(InputId.inputCentroids, centroids);

                /* Compute k-means on local nodes */
                PartialResult pres = kmeansLocal.compute();

                pres.pack();
                tup._2.pack();
                centroids.pack();

                localContext.dispose();

                return pres;
            }
        });
    }

    private static NumericTable computeMaster(DaalContext context, JavaRDD<PartialResult> partResRDD) {
        /* Create an algorithm to compute k-means on the master node */
        DistributedStep2Master kmeansMaster = new DistributedStep2Master(context, Double.class, Method.defaultDense, nClusters);

        /* Set the partial result to the master algorithm to compute the final result */
        List<PartialResult> partResList = partResRDD.collect();
        for (PartialResult value : partResList) {
            if(value != null) {
                value.unpack(context);
                kmeansMaster.input.add(DistributedStep2MasterInputId.partialResults, value);
            }
        }

        /* Compute k-means on the master node */
        kmeansMaster.compute();

        /* Finalize computations and retrieve the results */
        Result res = kmeansMaster.finalizeCompute();
        return res.get(ResultId.centroids);
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
