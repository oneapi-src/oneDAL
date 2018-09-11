/* file: PivotedQRDenseBatch.java */
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
 //     Java example of computing pivoted QR decomposition
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PIVOTEDQRBATCH">
 * @example PivotedQRDenseBatch.java
 */

package com.intel.daal.examples.pivoted_qr;

import com.intel.daal.algorithms.pivoted_qr.Batch;
import com.intel.daal.algorithms.pivoted_qr.InputId;
import com.intel.daal.algorithms.pivoted_qr.Method;
import com.intel.daal.algorithms.pivoted_qr.Result;
import com.intel.daal.algorithms.pivoted_qr.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class PivotedQRDenseBatch {

    /* Input data set parameters */
    private static final String dataset  = "../data/batch/qr.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();

        NumericTable input = dataSource.getNumericTable();

        /* Create an algorithm to compute pivoted QR decomposition */
        Batch pivotedQRAlgorithm = new Batch(context, Float.class, Method.defaultDense);
        pivotedQRAlgorithm.input.set(InputId.data, input);

        /* Compute pivoted QR decomposition */
        Result res = pivotedQRAlgorithm.compute();

        NumericTable matrixQ = res.get(ResultId.matrixQ);
        NumericTable matrixR = res.get(ResultId.matrixR);
        NumericTable permutationMatrix = res.get(ResultId.permutationMatrix);

        /* Print the results */
        printResults(input, matrixQ, matrixR, permutationMatrix);

        context.dispose();
    }

    private static void printResults(NumericTable data, NumericTable Q, NumericTable R, NumericTable P) {
        Service.printNumericTable("Orthogonal matrix Q:", Q, 10);
        Service.printNumericTable("Triangular matrix R:", R);
        Service.printNumericTable("Permutation matrix P:", P);
    }
}
