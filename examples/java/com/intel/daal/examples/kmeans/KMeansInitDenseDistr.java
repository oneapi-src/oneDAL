/* file: KMeansInitDenseDistr.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 //     Java example of dense K-Means clustering in the distributed processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-KMEANSINITDENSEDISTRIBUTED">
 * @example KMeansInitDenseDistr.java
 */

package com.intel.daal.examples.kmeans;

import com.intel.daal.algorithms.kmeans.*;
import com.intel.daal.algorithms.kmeans.init.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.RowMergedNumericTable;
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import java.util.ArrayList;

class KMeansInitDenseDistr {
    /* Input data set parameters */
    private static final String[] datasetFileNames = {
        "../data/distributed/kmeans_dense_1.csv", "../data/distributed/kmeans_dense_2.csv",
        "../data/distributed/kmeans_dense_3.csv", "../data/distributed/kmeans_dense_4.csv"};

    private static final int    nClusters       = 20;
    private static final int    nBlocks         = 4;
    private static final int    nIterations     = 5;
    private static final int    nVectorsInBlock = 2500;

    private static DaalContext context = new DaalContext();
    private static NumericTable[] data = new NumericTable[nBlocks];

    private static void loadData(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        for (int node = 0; node < nBlocks; node++) {
            /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
            FileDataSource dataSource = new FileDataSource(context, datasetFileNames[node],
                    DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                    DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

            /* Retrieve the input data */
            dataSource.loadDataBlock();

            data[node] = dataSource.getNumericTable();
        }
    }

    private static void calculateCentroids(final NumericTable initialCentroids) {

        NumericTable centroids = initialCentroids;
        NumericTable[] assignments = new NumericTable[nBlocks];
        NumericTable objectiveFunction = null;

        /* Create an algorithm for K-Means clustering */
        DistributedStep2Master masterAlgorithm = new DistributedStep2Master(context, Float.class, Method.defaultDense,
                nClusters);

        /* Calculate centroids */
        for (int it = 0; it < nIterations; it++) {
            for (int node = 0; node < nBlocks; node++) {
                /* Create an algorithm object for the K-Means algorithm */
                DistributedStep1Local algorithm = new DistributedStep1Local(context, Float.class, Method.defaultDense,
                        nClusters);

                /* Set the input data to the algorithm */
                algorithm.input.set(InputId.data, data[node]);
                algorithm.input.set(InputId.inputCentroids, centroids);

                PartialResult pres = algorithm.compute();

                masterAlgorithm.input.add(DistributedStep2MasterInputId.partialResults, pres);
            }

            masterAlgorithm.compute();
            Result result = masterAlgorithm.finalizeCompute();

            centroids = result.get(ResultId.centroids);
            objectiveFunction = result.get(ResultId.objectiveFunction);
        }

        /* Calculate assignments */
        for (int node = 0; node < nBlocks; node++) {
            /* Create an algorithm object for the K-Means algorithm */
            Batch algorithm = new Batch(context, Float.class, Method.lloydDense, nClusters, 0);

            algorithm.parameter.setAssignFlag(true);

            /* Set the input data to the algorithm */
            algorithm.input.set(InputId.data, data[node]);
            algorithm.input.set(InputId.inputCentroids, centroids);

            Result result = algorithm.compute();

            assignments[node] = result.get(ResultId.assignments);
        }

        /* Print the results */
        Service.printNumericTable("First 10 cluster assignments from 1st node:", assignments[0], 10);
        Service.printNumericTable("First 10 dimensions of centroids:", centroids, 20, 10);
        Service.printNumericTable("Objective function value:", objectiveFunction);
    }

    private static NumericTable initStep1(final InitMethod method, DataCollection[] localNodeData) {
        for (int node = 0; node < nBlocks; node++) {
            /* Create an algorithm object for the step 1 */
            InitDistributedStep1Local initLocal = new InitDistributedStep1Local(context, Float.class,
                    method, nClusters, nBlocks * nVectorsInBlock, node * nVectorsInBlock);

            /* Set the input data to the algorithm */
            initLocal.input.set(InitInputId.data, data[node]);

            /* Compute and get the result */
            InitPartialResult initPres = initLocal.compute();
            NumericTable pNewCenters = initPres.get(InitPartialResultId.partialCentroids);
            if(pNewCenters != null)
                return pNewCenters;
        }
        return null;
    }

    private static void initStep23(final InitMethod method, DataCollection[] localNodeData,
                                   final NumericTable step2Input, InitDistributedStep3Master step3,
                                   boolean bFirstIteration) {
        for (int node = 0; node < nBlocks; node++) {

            /* Create an algorithm object for the step 2 */
            InitDistributedStep2Local step2 = new InitDistributedStep2Local(context, Float.class,
                    method, nClusters, bFirstIteration);

            /* Set the input data to the algorithm */
            step2.input.set(InitInputId.data, data[node]);
            if(!bFirstIteration)
                step2.input.set(InitDistributedLocalPlusPlusInputDataId.internalInput, localNodeData[node]);
            step2.input.set(InitDistributedStep2LocalPlusPlusInputId.inputOfStep2, step2Input);

            /* Compute and get the result */
            InitDistributedStep2LocalPlusPlusPartialResult initPres = step2.compute();
            if(bFirstIteration)
                localNodeData[node] = initPres.get(InitDistributedStep2LocalPlusPlusPartialResultDataId.internalResult);

            /* Set the result to the input of step 3 */
            step3.input.add(InitDistributedStep3MasterPlusPlusInputId.inputOfStep3FromStep2, node,
                initPres.get(InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep3));
        }
    }

    private static NumericTable initStep4(final InitMethod method, DataCollection[] localNodeData,
                             InitDistributedStep3MasterPlusPlusPartialResult step3Pres) {

        ArrayList<NumericTable> results = new ArrayList<NumericTable>();
        for (int node = 0; node < nBlocks; node++) {
            /* Get an input for step 4 on this node if any */
            NumericTable step3Output = step3Pres.get(InitDistributedStep3MasterPlusPlusPartialResultId.outputOfStep3ForStep4, node);
            if(step3Output == null)
                continue; /* can be null */

            /* Create an algorithm object for the step 4 */
            InitDistributedStep4Local step4 = new InitDistributedStep4Local(context, Float.class, method, nClusters);
            /* Set the input data to the algorithm */
            step4.input.set(InitInputId.data, data[node]);
            step4.input.set(InitDistributedLocalPlusPlusInputDataId.internalInput, localNodeData[node]);
            step4.input.set(InitDistributedStep4LocalPlusPlusInputId.inputOfStep4FromStep3, step3Output);

            /* Compute and get the result */
            results.add(step4.compute().get(InitDistributedStep4LocalPlusPlusPartialResultId.outputOfStep4));
        }
        if(results.size() == 0)
            return null;
        if(results.size() == 1)
            return results.get(0);

        /* For parallelPlus algorithm */
        RowMergedNumericTable result = new RowMergedNumericTable(context);
        for(int i = 0; i < results.size(); i++)
            result.addNumericTable(results.get(i));
        return result;
    }

    private static NumericTable initCentroidsPlusPlus() {
        System.out.println("plusPlusDense");
        final InitMethod method = InitMethod.plusPlusDense;
        /* Internal data to be stored on the local nodes */
        DataCollection[] localNodeData = new DataCollection[nBlocks];
        /* Numeric table to collect the results */
        RowMergedNumericTable centroids = new RowMergedNumericTable(context);
        /* Firs step on the local nodes */
        NumericTable newCentroids = initStep1(method, localNodeData);
        centroids.addNumericTable(newCentroids);

        /* Create an algorithm object for the step 3 */
        InitDistributedStep3Master step3 = new InitDistributedStep3Master(context, Float.class, method, nClusters);
        for (int iCentroid = 1; iCentroid < nClusters; iCentroid++) {
            /* Perform steps 2 and 3 */
            initStep23(method, localNodeData, newCentroids, step3, iCentroid == 1);
            InitDistributedStep3MasterPlusPlusPartialResult initPres = step3.compute();
            /* Perform steps 4 */
            newCentroids = initStep4(method, localNodeData, initPres);
            centroids.addNumericTable(newCentroids);
        }
        return centroids;
    }

    private static NumericTable initCentroidsParallelPlus() {
        System.out.println("parallelPlusDense");
        final InitMethod method = InitMethod.parallelPlusDense;
        /* Internal data to be stored on the local nodes */
        DataCollection[] localNodeData = new DataCollection[nBlocks];
        /* Firs step on the local nodes */
        NumericTable newCentroids = initStep1(method, localNodeData);

        /* Create an algorithm object for the step 5 */
        InitDistributedStep5Master step5 = new InitDistributedStep5Master(context, Float.class, method, nClusters);
        step5.input.add(InitDistributedStep5MasterPlusPlusInputId.inputCentroids, newCentroids);
        /* Create an algorithm object for the step 3 */
        InitDistributedStep3Master step3 = new InitDistributedStep3Master(context, Float.class, method, nClusters);

        SerializableBase inputOfStep5FromStep3 = null;
        for (int iRound = 0; iRound < step5.parameter.getNRounds(); iRound++) {
            /* Perform steps 2 and 3 */
            initStep23(method, localNodeData, newCentroids, step3, iRound == 0);
            InitDistributedStep3MasterPlusPlusPartialResult initPres = step3.compute();
            if(iRound + 1 == step5.parameter.getNRounds())
                inputOfStep5FromStep3 = initPres.get(InitDistributedStep3MasterPlusPlusPartialResultDataId.outputOfStep3ForStep5);
            /* Perform steps 4 */
            newCentroids = initStep4(method, localNodeData, initPres);
            step5.input.add(InitDistributedStep5MasterPlusPlusInputId.inputCentroids, newCentroids);
        }
        /* One more step 2 */
        for (int node = 0; node < nBlocks; node++) {

            /* Create an algorithm object for the step 2 */
            InitDistributedStep2Local step2 = new InitDistributedStep2Local(context, Float.class,
                    method, nClusters, false);
            step2.parameter.setOutputForStep5Required(true);
            /* Set the input data to the algorithm */
            step2.input.set(InitInputId.data, data[node]);
            step2.input.set(InitDistributedLocalPlusPlusInputDataId.internalInput, localNodeData[node]);
            step2.input.set(InitDistributedStep2LocalPlusPlusInputId.inputOfStep2, newCentroids);

            /* Compute and get the result */
            InitDistributedStep2LocalPlusPlusPartialResult initPres = step2.compute();

            /* Add the results to the input of step 5 */
            step5.input.add(InitDistributedStep5MasterPlusPlusInputId.inputOfStep5FromStep2,
                initPres.get(InitDistributedStep2LocalPlusPlusPartialResultId.outputOfStep2ForStep5));
        }
        step5.input.set(InitDistributedStep5MasterPlusPlusInputDataId.inputOfStep5FromStep3, inputOfStep5FromStep3);
        step5.compute();
        return step5.finalizeCompute().get(InitResultId.centroids);
    }

    private static NumericTable initCentroids(final InitMethod method) {
        System.out.print("K-means init parameters: method = ");
        if(method == InitMethod.plusPlusDense)
            return initCentroidsPlusPlus();
        return initCentroidsParallelPlus();
    }

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        loadData(args);
        /* Get initial centroids by plusPlusDense method */
        NumericTable initialCentroids = initCentroids(InitMethod.plusPlusDense);
        /* Calculate centroids */
        calculateCentroids(initialCentroids);

        /* Get initial centroids by parallelPlusDense method */
        initialCentroids = initCentroids(InitMethod.parallelPlusDense);
        /* Calculate centroids */
        calculateCentroids(initialCentroids);

        context.dispose();
    }
}
