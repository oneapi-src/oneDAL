/* file: KMeansCSRDistr.java */
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
 //     Java example of sparse K-Means clustering in the distributed processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-KMEANSCSRDISTRIBUTED">
 * @example KMeansCSRDistr.java
 */

package com.intel.daal.examples.kmeans;

import com.intel.daal.algorithms.kmeans.*;
import com.intel.daal.algorithms.kmeans.init.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class KMeansCSRDistr {
    /* Input data set parameters */
    private static final String[] datasetFileNames = {
        "../data/distributed/kmeans_csr_1.csv", "../data/distributed/kmeans_csr_2.csv",
        "../data/distributed/kmeans_csr_3.csv", "../data/distributed/kmeans_csr_4.csv"};
    private static final int    nClusters       = 20;
    private static final int    nBlocks         = 4;
    private static final int    nIterations     = 5;
    private static final int    nVectorsInBlock = 8000;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Get initial centroids */
        InitDistributedStep2Master initMaster = new InitDistributedStep2Master(context, Float.class,
                InitMethod.randomCSR, nClusters);

        NumericTable[] data = new NumericTable[nBlocks];

        for (int node = 0; node < nBlocks; node++) {
            /* Create Numeric Table for input data */
            data[node] = Service.createSparseTable(context, datasetFileNames[node]);

            /* Create an algorithm object to initialize the K-Means algorithm */
            InitDistributedStep1Local initLocal = new InitDistributedStep1Local(context, Float.class,
                    InitMethod.randomCSR, nClusters, nBlocks * nVectorsInBlock, node * nVectorsInBlock);

            /* Set the input data to the algorithm */
            initLocal.input.set(InitInputId.data, data[node]);

            InitPartialResult initPres = initLocal.compute();
            initMaster.input.add(InitDistributedStep2MasterInputId.partialResults, initPres);
        }

        initMaster.compute();
        InitResult initResult = initMaster.finalizeCompute();

        NumericTable centroids = initResult.get(InitResultId.centroids);
        NumericTable[] assignments = new NumericTable[nBlocks];
        NumericTable objectiveFunction = null;

        /* Create an algorithm for K-Means clustering */
        DistributedStep2Master masterAlgorithm = new DistributedStep2Master(context, Float.class, Method.lloydCSR,
                nClusters);

        /* Calculate centroids */
        for (int it = 0; it < nIterations; it++) {
            for (int node = 0; node < nBlocks; node++) {
                /* Create an algorithm object for the K-Means algorithm */
                DistributedStep1Local algorithm = new DistributedStep1Local(context, Float.class, Method.lloydCSR,
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
            Batch algorithm = new Batch(context, Float.class, Method.lloydCSR, nClusters, 0);

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

        context.dispose();
    }
}
