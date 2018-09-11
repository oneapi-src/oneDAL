/* file: LowOrderMomsCSROnline.java */
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
 //     Java example of computing low order moments in the online processing
 //     mode.
 //
 //     Input matrix is stored in the compressed sparse row (CSR) format with
 //     one-based indexing.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOWORDERMOMENTSCSRONLINE">
 * @example LowOrderMomsCSROnline.java
 */

package com.intel.daal.examples.moments;

import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/*
// Input data set is stored in the compressed sparse row format
*/

class LowOrderMomsCSROnline {

    /* Input data set parameters */
    private static final String datasetFileNames[] = new String[] { "../data/online/covcormoments_csr_1.csv",
            "../data/online/covcormoments_csr_2.csv", "../data/online/covcormoments_csr_3.csv",
            "../data/online/covcormoments_csr_4.csv" };
    private static final int    nBlocks            = 4;

    private static Result result;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Create algorithm objects to compute low order moments in the online processing mode using the default method */
        Online algorithm = new Online(context, Float.class, Method.fastCSR);

        for (int i = 0; i < nBlocks; i++) {
            /* Read the input data from a file */
            CSRNumericTable dataTable = Service.createSparseTable(context, datasetFileNames[i]);

            /* Set input objects for the algorithm */
            algorithm.input.set(InputId.data, dataTable);

            /* Compute partial low order moments estimates */
            PartialResult pres = algorithm.compute();

            dataTable.dispose();
            pres.dispose();
        }

        /* Finalize the result in the online processing mode */
        result = algorithm.finalizeCompute();

        printResults();

        context.dispose();
    }

    private static void printResults() {
        NumericTable minimum = result.get(ResultId.minimum);
        NumericTable maximum = result.get(ResultId.maximum);
        NumericTable sum = result.get(ResultId.sum);
        NumericTable sumSquares = result.get(ResultId.sumSquares);
        NumericTable sumSquaresCentered = result.get(ResultId.sumSquaresCentered);
        NumericTable mean = result.get(ResultId.mean);
        NumericTable secondOrderRawMoment = result.get(ResultId.secondOrderRawMoment);
        NumericTable variance = result.get(ResultId.variance);
        NumericTable standardDeviation = result.get(ResultId.standardDeviation);
        NumericTable variation = result.get(ResultId.variation);

        Service.printNumericTable("Minimum:", minimum);
        Service.printNumericTable("Maximum:", maximum);
        Service.printNumericTable("Sum:", sum);
        Service.printNumericTable("Sum of squares:", sumSquares);
        Service.printNumericTable("Sum of squared difference from the means:", sumSquaresCentered);
        Service.printNumericTable("Mean:", mean);
        Service.printNumericTable("Second order raw moment:", secondOrderRawMoment);
        Service.printNumericTable("Variance:", variance);
        Service.printNumericTable("Standart deviation:", standardDeviation);
        Service.printNumericTable("Variation:", variation);
    }
}
