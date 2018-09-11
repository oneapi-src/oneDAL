/* file: KMeansDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of dense K-Means clustering in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-KMEANSDENSEBATCH">
 * @example KMeansDenseBatch.java
 */

package com.intel.daal.examples.kmeans;

import com.intel.daal.algorithms.kmeans.*;
import com.intel.daal.algorithms.kmeans.init.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class KMeansDenseBatch {
    /* Input data set parameters */
    private static final String dataset       = "../data/batch/kmeans_dense.csv";
    private static final int    nClusters     = 20;

    /* K-Means algorithm parameters */
    private static final int maxIterations = 5;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();
        NumericTable input = dataSource.getNumericTable();

        /* Calculate initial clusters for K-Means clustering */
        InitBatch init = new InitBatch(context, Float.class, InitMethod.randomDense, nClusters);
        init.input.set(InitInputId.data, input);
        InitResult initResult = init.compute();
        NumericTable inputCentroids = initResult.get(InitResultId.centroids);

        /* Create an algorithm for K-Means clustering */
        Batch algorithm = new Batch(context, Float.class, Method.lloydDense, nClusters, maxIterations);

        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);
        algorithm.input.set(InputId.inputCentroids, inputCentroids);

        /* Clusterize the data */
        Result result = algorithm.compute();

        /* Print the results */
        Service.printNumericTable("First 10 cluster assignments:", result.get(ResultId.assignments), 10);
        Service.printNumericTable("First 10 dimensions of centroids:", result.get(ResultId.centroids), 20, 10);
        Service.printNumericTable("Objective function value:", result.get(ResultId.objectiveFunction));

        context.dispose();
    }
}
