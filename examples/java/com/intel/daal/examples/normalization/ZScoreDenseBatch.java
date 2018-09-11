/* file: ZScoreDenseBatch.java */
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
 //     Java example of Z-score normalization algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.normalization;

import com.intel.daal.algorithms.normalization.zscore.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-ZSCOREBATCH">
 * @example ZScoreDenseBatch.java
 */

class ZScoreDenseBatch {
    private static final String dataset     = "../data/batch/normalization.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                                                       DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                       DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        NumericTable input = dataSource.getNumericTable();

        /* Create an algorithm */
        Batch algorithm = new Batch(context, Float.class, Method.defaultDense);

        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);

        algorithm.parameter.setResultsToCompute(ResultsToComputeId.mean);// | ResultsToComputeId.variance);
        /* Compute Z-score normalization function */
        Result result = algorithm.compute();

        /* Print the results of stage */
        Service.printNumericTable("First 10 rows of the input data:", input, 10);
        Service.printNumericTable("First 10 rows of the z-score normalization result:", result.get(ResultId.normalizedData), 10);
      //Service.printNumericTable("Means:", result.get(ResultId.means));
      //Service.printNumericTable("Variances:", result.get(ResultId.variances));

        context.dispose();
    }
}
