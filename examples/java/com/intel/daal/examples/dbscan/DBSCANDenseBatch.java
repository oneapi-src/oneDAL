/* file: DBSCANDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 //     Java example of dense DBSCAN clustering in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DBSCANDENSEBATCH">
 * @example DBSCANDenseBatch.java
 */

package com.intel.daal.examples.dbscan;

import com.intel.daal.algorithms.dbscan.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class DBSCANDenseBatch {
    /* Input data set parameters */
    private static final String dataset       = "../data/batch/dbscan_dense.csv";

    /* DBSCAN algorithm parameters */
    private static final double epsilon = 0.02;
    private static final long   minObservations = 180;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();
        NumericTable input = dataSource.getNumericTable();

        /* Create an algorithm for DBSCAN clustering */
        Batch algorithm = new Batch(context, Float.class, Method.defaultDense, epsilon, minObservations);

        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);

        /* Clusterize the data */
        Result result = algorithm.compute();

        /* Print the results */
        Service.printNumericTable("Number of clusters:", result.get(ResultId.nClusters));
        Service.printNumericTable("Assignments of first 20 observations:", result.get(ResultId.assignments), 20);

        context.dispose();
    }
}
