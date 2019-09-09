/* file: DBSCANDenseBatch.java */
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
