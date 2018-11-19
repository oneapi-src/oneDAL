/* file: OutDetectBaconDenseBatch.java */
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
 //     Java example of multivariate outlier detection using the Bacon method
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-OUTDETECTBACONDENSEBATCH">
 * @example OutDetectBaconDenseBatch.java
 */

package com.intel.daal.examples.outlier_detection;

import java.nio.FloatBuffer;
import com.intel.daal.algorithms.bacon_outlier_detection.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;

import com.intel.daal.services.DaalContext;

class OutDetectBaconDenseBatch {

    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/outlierdetection.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();

        /* Create an algorithm to detect outliers using the Bacon method */
        Batch alg = new Batch(context, Double.class, Method.defaultDense);

        NumericTable data = dataSource.getNumericTable();
        alg.input.set(InputId.data, data);

        /* Detect outliers */
        Result result = alg.compute();

        NumericTable weights = result.get(ResultId.weights);

        FloatBuffer dataFloat = FloatBuffer.allocate(0);
        dataFloat = data.getBlockOfRows(0, 20, dataFloat);
        FloatBuffer dataWeights = FloatBuffer.allocate(0);
        dataWeights = weights.getBlockOfRows(0, 20, dataWeights);

        printFloatBuffers("Input data","Weights","Outlier detection result (Bacon method)",dataFloat,dataWeights,3,20);

        context.dispose();
    }

    private static void printFloatBuffers (String header1, String header2, String message, FloatBuffer buf1, FloatBuffer buf2, int nColumns, int nRows) {
        int step = (int)nColumns;
        System.out.println(message);
        System.out.println(header1 + "                  " + header2);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nColumns; j++) {
                System.out.format("%6.3f   ", buf1.get(i * step + j));
            }
            System.out.format("%6.3f   ", buf2.get(i));
            System.out.println("");
        }
        System.out.println("");
    }
}
