/* file: DataStructuresMatrix.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
    private static final int nRead            = 3;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
        System.out.println("Matrix example");
        int readFeatureIdx;

        double[] data = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 2,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 3,
            80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 4,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 5
        };

        Matrix dataTable = new Matrix(context, data, nFeaturesMatrix, nVectorsMatrix);

        /* Read a block of rows */
        DoubleBuffer dataDouble = DoubleBuffer.allocate(nRead * nFeaturesMatrix);

        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);
        printDoubleBuffer(dataDouble, nFeaturesMatrix, nRead, "Print 3 rows from matrix as double:");
        dataTable.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        readFeatureIdx = 2;

        /* Set new values in Matrix */
        dataTable.set(firstReadRow, readFeatureIdx, (double)100);
        dataTable.set(firstReadRow + 1, readFeatureIdx, (double)101);
        dataTable.set(firstReadRow + 2, readFeatureIdx, (double)102);

        /* Read a feature (column) */
        DoubleBuffer dataDoubleFeatures = DoubleBuffer.allocate((int) nVectorsMatrix);
        dataDoubleFeatures = dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nVectorsMatrix, dataDoubleFeatures);
        printDoubleBuffer(dataDoubleFeatures, 1, nVectorsMatrix, "Print the third feature of matrix:");
        dataTable.releaseBlockOfColumnValues(readFeatureIdx, firstReadRow, nVectorsMatrix, dataDoubleFeatures);

        /* Get a pointer to the inner array for Matrix. This pointer is a pointer to the array data */
        double[] dataFromNumericTable = (double[])dataTable.getDataObject();
        Service.printMatrix(dataFromNumericTable, nFeaturesMatrix, nVectorsMatrix, "Data from getDataObject:");

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
