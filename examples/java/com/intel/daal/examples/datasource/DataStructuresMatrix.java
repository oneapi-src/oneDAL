/* file: DataStructuresMatrix.java */
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
 //     Java example of using matrix data structures
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DATASTRUCTURESMATRIX">
 * @example DataStructuresMatrix.java
 */

package com.intel.daal.examples.datasource;

import java.nio.DoubleBuffer;
import com.intel.daal.data_management.data.Matrix;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

class DataStructuresMatrix {
    private static final int nVectorsMatrix  = 10;
    private static final int nFeaturesMatrix = 11;
    private static final int firstReadRow     = 0;
    private static final int nRead            = 5;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
        System.out.println("Matrix numeric table example\n");
        int readFeatureIdx;

        double[] data = {
            0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1,
            1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2,
            2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3,
            3.0f, 3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f, 3.7f, 3.8f, 3.9f, 4,
            4.0f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f, 4.7f, 4.8f, 4.9f, 5,
            5.0f, 5.1f, 5.2f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f, 5.9f, 1,
            6.0f, 6.1f, 6.2f, 6.3f, 6.4f, 6.5f, 6.6f, 6.7f, 6.8f, 6.9f, 2,
            7.0f, 7.1f, 7.2f, 7.3f, 7.4f, 7.5f, 7.6f, 7.7f, 7.8f, 7.9f, 3,
            8.0f, 8.1f, 8.2f, 8.3f, 8.4f, 8.5f, 8.6f, 8.7f, 8.8f, 8.9f, 4,
            9.0f, 9.1f, 9.2f, 9.3f, 9.4f, 9.5f, 9.6f, 9.7f, 9.8f, 9.9f, 5
        };

        Matrix dataTable = new Matrix(context, data, nFeaturesMatrix, nVectorsMatrix);

        /* Read a block of rows */
        DoubleBuffer dataDouble = DoubleBuffer.allocate(nRead * nFeaturesMatrix);

        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);
        System.out.printf("%d  rows are read\n", nRead);
        printDoubleBuffer(dataDouble, nFeaturesMatrix, nRead, "Print 5 rows from matrix data array as float:");
        dataTable.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        readFeatureIdx = 2;

        /* Set new values in Matrix */
        dataTable.set(firstReadRow, readFeatureIdx, (double)-1);
        dataTable.set(firstReadRow + 1, readFeatureIdx, (double)-2);
        dataTable.set(firstReadRow + 2, readFeatureIdx, (double)-3);

        /* Read a feature (column) */
        DoubleBuffer dataDoubleFeatures = DoubleBuffer.allocate((int) nVectorsMatrix);
        dataDoubleFeatures = dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nVectorsMatrix, dataDoubleFeatures);
        printDoubleBuffer(dataDoubleFeatures, 1, nVectorsMatrix, "Print the third feature of matrix data:");
        dataTable.releaseBlockOfColumnValues(readFeatureIdx, firstReadRow, nVectorsMatrix, dataDoubleFeatures);

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
