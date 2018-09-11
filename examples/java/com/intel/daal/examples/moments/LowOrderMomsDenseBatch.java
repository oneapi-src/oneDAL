/* file: LowOrderMomsDenseBatch.java */
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
 //     Java example of computing low order moments in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOWORDERMOMENTSDENSEBATCH">
 * @example LowOrderMomsDenseBatch.java
 */

package com.intel.daal.examples.moments;

import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class LowOrderMomsDenseBatch {

    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/covcormoments_dense.csv";

    private static DaalContext context = new DaalContext();

    private static Result result;

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create algorithm objects to compute low order moments using the default method */
        Batch algorithm = new Batch(context, Float.class, Method.defaultDense);

        /* Set input objects for the algorithm */
        NumericTable input = dataSource.getNumericTable();
        algorithm.input.set(InputId.data, input);

        /* Compute low order moments */
        result = algorithm.compute();

        printResults();

        context.dispose();
    }

    private static void printResults() {
        Service.printNumericTable("Minimum:", result.get(ResultId.minimum));
        Service.printNumericTable("Maximum:", result.get(ResultId.maximum));
        Service.printNumericTable("Sum:", result.get(ResultId.sum));
        Service.printNumericTable("Sum of squares:", result.get(ResultId.sumSquares));
        Service.printNumericTable("Sum of squared difference from the means:", result.get(ResultId.sumSquaresCentered));
        Service.printNumericTable("Mean:", result.get(ResultId.mean));
        Service.printNumericTable("Second order raw moment:", result.get(ResultId.secondOrderRawMoment));
        Service.printNumericTable("Variance:", result.get(ResultId.variance));
        Service.printNumericTable("Standart deviation:", result.get(ResultId.standardDeviation));
        Service.printNumericTable("Variation:", result.get(ResultId.variation));
    }
}
