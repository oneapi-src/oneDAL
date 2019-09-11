/* file: CorCSRBatch.java */
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
 //     Java example of correlation matrix computation in the batch
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-CORRELATIONCSRBATCH">
 * @example CorCSRBatch.java
 */

package com.intel.daal.examples.covariance;

import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class CorCSRBatch {
    /*
     * Input data set parameters
     * Input matrix is stored in the compressed sparse row format with one-based indexing
     */
    private static final String datasetFileName = "../data/batch/covcormoments_csr.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read a data set from a file and create a numeric table for storing the input data */
        CSRNumericTable dataTable = Service.createSparseTable(context, datasetFileName);

        /* Create an algorithm to compute a correlation matrix using the default method */
        Batch alg = new Batch(context, Float.class, Method.fastCSR);
        alg.input.set(InputId.data, dataTable);

        /* Set the parameter to choose the type of the output matrix */
        alg.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix);

        /* Compute the correlation matrix */
        Result res = alg.compute();

        HomogenNumericTable correlation = (HomogenNumericTable) res.get(ResultId.correlation);
        HomogenNumericTable mean = (HomogenNumericTable) res.get(ResultId.mean);

        Service.printNumericTable("Correlation matrix (upper left square 10*10) :", correlation, 10, 10);
        Service.printNumericTable("Mean vector:", mean, 1, 10);

        context.dispose();
    }
}
