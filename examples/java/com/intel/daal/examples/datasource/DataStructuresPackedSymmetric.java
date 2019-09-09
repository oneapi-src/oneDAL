/* file: DataStructuresPackedSymmetric.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of using packed data structures
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DATASTRUCTURESPACKEDSYMMETRIC">
 * @example DataStructuresPackedSymmetric.java
 */

package com.intel.daal.examples.datasource;

import java.nio.DoubleBuffer;
import com.intel.daal.data_management.data.PackedSymmetricMatrix;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

class DataStructuresPackedSymmetric {
    private static final int nDim  = 5;
    private static final int nRead = 5;
    private static final int firstReadRow = 0;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
        System.out.println("Packed symmetric matrix example\n");
        int readFeatureIdx;

        double[] data = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4 };

        PackedSymmetricMatrix dataTable = new PackedSymmetricMatrix(context, data, nDim, NumericTable.StorageLayout.lowerPackedSymmetricMatrix);

        /* Read a block of rows */
        DoubleBuffer dataDouble = DoubleBuffer.allocate(nRead * nDim);

        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);
        System.out.printf("%d rows are read\n", nRead);
        printDoubleBuffer(dataDouble, nDim, nRead, "Print 3 rows from packed symmetric matrix as float:");
        dataTable.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        /* Read a feature (column) */
        DoubleBuffer dataDoubleFeatures = DoubleBuffer.allocate((int) nRead);
        readFeatureIdx = 2;
        dataDoubleFeatures = dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nRead, dataDoubleFeatures);
        printDoubleBuffer(dataDoubleFeatures, 1, nRead, "Print the third feature of packed symmetric matrix:");

        /* Set new value to a buffer and release it */
        dataDoubleFeatures.put(1, 239);
        dataTable.releaseBlockOfColumnValues(readFeatureIdx, firstReadRow, nRead, dataDoubleFeatures);

        /* Read a block of rows */
        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);
        System.out.printf("%d rows are read\n", nRead);
        printDoubleBuffer(dataDouble, nDim, nRead, "Print 3 rows from packed symmetric matrix as float:");
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
