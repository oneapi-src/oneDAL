/* file: CholeskyDenseBatch.java */
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
 //     Java example of Cholesky decomposition
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.cholesky;

import com.intel.daal.algorithms.cholesky.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-CHOLESKYBATCH">
 * @example CholeskyDenseBatch.java
 */

class CholeskyDenseBatch {
    private static final String dataset   = "../data/batch/cholesky.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm */
        Batch choleskyAlgorithm = new Batch(context, Float.class, Method.defaultDense);

        /* Set an input object for the algorithm */
        NumericTable input = dataSource.getNumericTable();
        choleskyAlgorithm.input.set(InputId.data, input);

        /* Compute Cholesky decomposition */
        Result result = choleskyAlgorithm.compute();

        NumericTable choleskyFactor = result.get(ResultId.choleskyFactor);

        Service.printNumericTable("", choleskyFactor);

        context.dispose();
    }
}
