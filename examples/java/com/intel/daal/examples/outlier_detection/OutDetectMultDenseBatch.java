/* file: OutDetectMultDenseBatch.java */
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
 //     Java example of multivariate outlier detection
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-OUTLIERDETECTIONMULTIVARIATEDENSEBATCH">
 * @example OutDetectMultDenseBatch.java
 */

package com.intel.daal.examples.outlier_detection;

import com.intel.daal.algorithms.multivariate_outlier_detection.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;

import com.intel.daal.services.DaalContext;

class OutDetectMultDenseBatch {

    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/outlierdetection.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();

        /* Create an algorithm to detect outliers using the default method */
        Batch alg = new Batch(context, Double.class, Method.defaultDense);

        NumericTable data = dataSource.getNumericTable();
        alg.input.set(InputId.data, data);

        /* Detect outliers */
        Result result = alg.compute();

        NumericTable weights = result.get(ResultId.weights);

        Service.printNumericTables(data,weights,"Input data","Weights","Outlier detection result (Default method)",0);
        context.dispose();
    }
}
