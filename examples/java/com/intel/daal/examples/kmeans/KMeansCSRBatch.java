/* file: KMeansCSRBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
 //     Java example of sparse K-Means clustering in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-KMEANSCSRBATCH">
 * @example KMeansCSRBatch.java
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

class KMeansCSRBatch {
    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/kmeans_csr.csv";
    private static final int    nClusters       = 20;

    /* K-Means algorithm parameters */
    private static final int maxIterations = 5;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        CSRNumericTable input = Service.createSparseTable(context, datasetFileName);

        /* Calculate initial clusters for K-Means clustering */
        InitBatch init = new InitBatch(context, Double.class, InitMethod.randomCSR, nClusters);
        init.input.set(InitInputId.data, input);
        InitResult initResult = init.compute();
        NumericTable inputCentroids = initResult.get(InitResultId.centroids);

        /* Create an algorithm for K-Means clustering */
        Batch algorithm = new Batch(context, Double.class, Method.lloydCSR, nClusters, maxIterations);

        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);
        algorithm.input.set(InputId.inputCentroids, inputCentroids);

        /* Clusterize the data */
        Result result = algorithm.compute();

        /* Print the results */
        Service.printNumericTable("First 20 cluster assignments:", result.get(ResultId.assignments), 20);
        Service.printNumericTable("First 10 dimensions of centroids:", result.get(ResultId.centroids), 20, 10);
        Service.printNumericTable("Goal function value:", result.get(ResultId.goalFunction));

        context.dispose();
    }
}
