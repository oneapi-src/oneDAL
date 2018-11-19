/* file: DataStructuresPackedTriangular.java */
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
 //     Java example of using packed data structures
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DATASTRUCTURESPACKEDTRIANGULAR">
 * @example DataStructuresPackedTriangular.java
 */

package com.intel.daal.examples.datasource;

import java.nio.DoubleBuffer;
import com.intel.daal.data_management.data.PackedTriangularMatrix;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

class DataStructuresPackedTriangular {
    private static final int nDim  = 5;
    private static final int nRead = 5;
    private static final int firstReadRow = 0;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
        System.out.println("Packed triangular matrix example\n");
        int readFeatureIdx;

        float[] data = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f };

        PackedTriangularMatrix dataTable = new PackedTriangularMatrix(context, data, nDim, NumericTable.StorageLayout.lowerPackedTriangularMatrix);

        /* Read a block of rows */
        DoubleBuffer dataDouble = DoubleBuffer.allocate(nRead * nDim);

        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);
        System.out.printf("%d rows are read\n",nRead);
        printDoubleBuffer(dataDouble, nDim, nRead, "Print 3 rows from packed triangular matrix as float:");
        dataTable.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        /* Read a feature (column) */
        DoubleBuffer dataDoubleFeatures = DoubleBuffer.allocate((int) nRead);
        readFeatureIdx = 2;
        dataDoubleFeatures = dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nRead, dataDoubleFeatures);
        printDoubleBuffer(dataDoubleFeatures, 1, nRead, "Print the third feature of packed triangular matrix:");

        /* Set new value to a buffer and release it */
        dataDoubleFeatures.put(readFeatureIdx-1, -1);
        dataDoubleFeatures.put(readFeatureIdx+1, -2);
        dataTable.releaseBlockOfColumnValues(readFeatureIdx, firstReadRow, nRead, dataDoubleFeatures);

        /* Read a block of rows */
        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);
        System.out.printf("%d rows are read\n",nRead);
        printDoubleBuffer(dataDouble, nDim, nRead, "Print 3 rows from packed triangular matrix as float:");
        dataTable.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        context.dispose();
    }

    private static void printDoubleBuffer(DoubleBuffer buf, long nColumns, long nRows, String message) {
        int step = (int) nColumns;
        System.out.println(message);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nColumns; j++) {
                System.out.format("%6.3f   ", buf.get(i * step + j));
            }
            System.out.println("");
        }
        System.out.println("");
    }
}
