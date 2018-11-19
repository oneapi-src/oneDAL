/* file: KMeansCSRBatchAssign.java */
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
 //     Java example of sparse K-Means clustering in the batch processing mode
 //     for calculation assignments without centroids update
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-KMEANSCSRBATCHASSIGN">
 * @example KMeansCSRBatchAssign.java
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

class KMeansCSRBatchAssign {
    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/kmeans_csr.csv";
    private static final int    nClusters       = 20;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        CSRNumericTable input = Service.createSparseTable(context, datasetFileName);

        /* Calculate initial clusters for K-Means clustering */
        InitBatch init = new InitBatch(context, Float.class, InitMethod.randomCSR, nClusters);
        init.input.set(InitInputId.data, input);
        InitResult initResult = init.compute();
        NumericTable inputCentroids = initResult.get(InitResultId.centroids);

        /* Create an algorithm for K-Means clustering to calculate only assignments */
        Batch algorithm = new Batch(context, Float.class, Method.lloydCSR, nClusters, 0);

        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);
        algorithm.input.set(InputId.inputCentroids, inputCentroids);

        /* Clusterize the data */
        Result result = algorithm.compute();

        /* Print the results */
        Service.printNumericTable("First 10 cluster assignments:", result.get(ResultId.assignments), 10);

        context.dispose();
    }
}
