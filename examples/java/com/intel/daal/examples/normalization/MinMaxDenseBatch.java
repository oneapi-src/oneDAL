/* file: MinMaxDenseBatch.java */
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
 //     Java example of Min-max normalization algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.normalization;

import com.intel.daal.algorithms.normalization.minmax.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-MINMAXDENSEBATCH">
 * @example MinMaxDenseBatch.java
 */

class MinMaxDenseBatch {
    private static final String dataset = "../data/batch/normalization.csv";

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

        /* Set lower and upper bounds for the algorithm */
        algorithm.parameter.setLowerBound(-1.0);
        algorithm.parameter.setUpperBound( 1.0);

        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);

        /* Compute Min-max normalization function */
        Result result = algorithm.compute();

        /* Print the results of stage */
        Service.printNumericTable("First 10 rows of the input data:", input, 10);
        Service.printNumericTable("First 10 rows of the Min-max normalization result:", result.get(ResultId.normalizedData), 10);

        context.dispose();
    }
}
