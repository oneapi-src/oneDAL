/* file: CorCSROnline.java */
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
 //     Java example of correlation matrix computation in the online
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-CORRELATIONCSRONLINE">
 * @example CorCSROnline.java
 */

package com.intel.daal.examples.covariance;

import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/*
// Input data set is stored in the compressed sparse row format
*/

class CorCSROnline {

    /* Input data set parameters */
    private static final String datasetFileNames[] = new String[] { "../data/online/covcormoments_csr_1.csv",
            "../data/online/covcormoments_csr_2.csv", "../data/online/covcormoments_csr_3.csv",
            "../data/online/covcormoments_csr_4.csv" };
    private static final int    nBlocks            = 4;

    private static Result result;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Create algorithm objects to compute a correlation matrix in the online processing mode using the default method */
        Online algorithm = new Online(context, Float.class, Method.fastCSR);

        /* Set the parameter to choose the type of the output matrix */
        algorithm.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix);

        for (int i = 0; i < nBlocks; i++) {
            /* Read the input data from a file */
            CSRNumericTable dataTable = Service.createSparseTable(context, datasetFileNames[i]);

            /* Set input objects for the algorithm */
            algorithm.input.set(InputId.data, dataTable);

            /* Compute partial estimates */
            algorithm.compute();
        }

        /* Finalize the result in the online processing mode */
        result = algorithm.finalizeCompute();

        HomogenNumericTable correlation = (HomogenNumericTable) result.get(ResultId.correlation);
        HomogenNumericTable mean = (HomogenNumericTable) result.get(ResultId.mean);

        Service.printNumericTable("Correlation matrix (upper left square 10*10) :", correlation, 10, 10);
        Service.printNumericTable("Mean vector:", mean, 1, 10);

        context.dispose();
    }
}
