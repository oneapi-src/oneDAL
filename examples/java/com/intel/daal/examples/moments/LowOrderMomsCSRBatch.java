/* file: LowOrderMomsCSRBatch.java */
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
//  Java example of computing low order moments in the batch processing mode.
//
//      Input matrix is stored in the compressed sparse row (CSR) format with
//      one-based indexing.
////////////////////////////////////////////////////////////////////////////////
*/

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOWORDERMOMENTSCSRBATCH">
 * @example LowOrderMomsCSRBatch.java
 */

package com.intel.daal.examples.moments;

import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
/*
// Input data set is stored in the compressed sparse row format
*/
import com.intel.daal.services.DaalContext;

class LowOrderMomsCSRBatch {

    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/covcormoments_csr.csv";

    private static CSRNumericTable dataTable;
    private static Result          result;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read the input data from a file */
        dataTable = Service.createSparseTable(context, datasetFileName);

        /* Create algorithm objects to compute low order moments using the default method */
        Batch algorithm = new Batch(context, Float.class, Method.fastCSR);

        /* Set input objects for the algorithm */
        algorithm.input.set(InputId.data, dataTable);

        /* Compute low order moments */
        result = algorithm.compute();

        printResults();

        context.dispose();
    }

    private static void printResults() {
        HomogenNumericTable minimum = (HomogenNumericTable) result.get(ResultId.minimum);
        HomogenNumericTable maximum = (HomogenNumericTable) result.get(ResultId.maximum);
        HomogenNumericTable sum = (HomogenNumericTable) result.get(ResultId.sum);
        HomogenNumericTable sumSquares = (HomogenNumericTable) result.get(ResultId.sumSquares);
        HomogenNumericTable sumSquaresCentered = (HomogenNumericTable) result.get(ResultId.sumSquaresCentered);
        HomogenNumericTable mean = (HomogenNumericTable) result.get(ResultId.mean);
        HomogenNumericTable secondOrderRawMoment = (HomogenNumericTable) result.get(ResultId.secondOrderRawMoment);
        HomogenNumericTable variance = (HomogenNumericTable) result.get(ResultId.variance);
        HomogenNumericTable standardDeviation = (HomogenNumericTable) result.get(ResultId.standardDeviation);
        HomogenNumericTable variation = (HomogenNumericTable) result.get(ResultId.variation);

        Service.printNumericTable("Minimum:", minimum);
        Service.printNumericTable("Maximum:", maximum);
        Service.printNumericTable("Sum:", sum);
        Service.printNumericTable("Sum of squares:", sumSquares);
        Service.printNumericTable("Sum of squared difference from the means:", sumSquaresCentered);
        Service.printNumericTable("Mean:", mean);
        Service.printNumericTable("Second order raw moment:", secondOrderRawMoment);
        Service.printNumericTable("Variance:", variance);
        Service.printNumericTable("Standard deviation:", standardDeviation);
        Service.printNumericTable("Variation:", variation);
    }
}
