/* file: CorDenseBatch.java */
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
 //     Java example of dense correlation matrix computation in the batch
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-CORRELATIONDENSEBATCH">
 * @example CorDenseBatch.java
 */

package com.intel.daal.examples.covariance;

import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class CorDenseBatch {
    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/covcormoments_dense.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm to compute a correlation matrix using the single-pass method */
        Batch alg = new Batch(context, Float.class, Method.defaultDense);
        NumericTable input = dataSource.getNumericTable();
        alg.input.set(InputId.data, input);

        /* Set the parameter to choose the type of the output matrix */
        alg.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix);

        /* Compute the correlation matrix */
        Result res = alg.compute();

        HomogenNumericTable correlation = (HomogenNumericTable) res.get(ResultId.correlation);
        HomogenNumericTable mean = (HomogenNumericTable) res.get(ResultId.mean);

        Service.printNumericTable("Correlation matrix:", correlation);
        Service.printNumericTable("Mean vector:", mean);

        context.dispose();
    }
}
