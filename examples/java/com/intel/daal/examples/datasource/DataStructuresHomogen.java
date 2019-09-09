/* file: DataStructuresHomogen.java */
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
 //     Java example of using homogeneous data structures
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DATASTRUCTURESHOMOGEN">
 * @example DataStructuresHomogen.java
 */

package com.intel.daal.examples.datasource;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class DataStructuresHomogen {
    private static final int nVectorsHomogen  = 10;
    private static final int nFeaturesHomogen = 11;
    private static final int firstReadRow     = 0;
    private static final int nRead            = 3;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) {
        System.out.println("Homogeneous numeric table example\n");
        int readFeatureIdx;

        double[] data = {
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5,
            5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 1,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 2,
            7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 3,
            8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 4,
            9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 5
        };

        HomogenNumericTable dataTable = new HomogenNumericTable(context, data, nFeaturesHomogen, nVectorsHomogen);

        /* Read a block of rows */
        DoubleBuffer dataDouble = DoubleBuffer.allocate(nRead * nFeaturesHomogen);

        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);
        System.out.printf("%d rows are read\n",nRead);
        printDoubleBuffer(dataDouble, nFeaturesHomogen, nRead, "Print 3 rows from homogeneous data array as float:");
        dataTable.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        /* Read a feature (column) */
        DoubleBuffer dataDoubleFeatures = DoubleBuffer.allocate((int) nVectorsHomogen);
        readFeatureIdx = 2;
        dataDoubleFeatures = dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nVectorsHomogen, dataDoubleFeatures);
        printDoubleBuffer(dataDoubleFeatures, 1, nVectorsHomogen, "Print the third feature of homogeneous data:");
        dataTable.releaseBlockOfColumnValues(readFeatureIdx, firstReadRow, nVectorsHomogen, dataDoubleFeatures);

        /* Get a pointer to the inner array for HomogenNumericTable. This pointer is a pointer to the array data */
        data[0] = 999;
        double[] dataFromNumericTable = dataTable.getDoubleArray();
        Service.printMatrix(dataFromNumericTable, nFeaturesHomogen, nVectorsHomogen, "Data from getArray:");

        int nNewVectors = 3;
        float[] newData =
        {
            1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f,
            3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f,
            5.0f, 6.0f, 5.0f, 6.0f, 5.0f, 6.0f, 5.0f, 6.0f, 5.0f, 6.0f, 5.0f,
        };

        /* Set new data to HomogenNumericTable. It mush have the same type as the numeric table. */
        HomogenNumericTable newDataTable = new HomogenNumericTable(context, newData, nFeaturesHomogen, nNewVectors);
        /* Ensure the data has changed */
        readFeatureIdx = 1;
        FloatBuffer dataFloat = FloatBuffer.allocate((int) nNewVectors);;

        dataFloat = newDataTable.getBlockOfColumnValues((long)readFeatureIdx, (long)firstReadRow, (long)nNewVectors, dataFloat);
        printFloatBuffer(dataFloat, 1, nNewVectors, "\nPrint the second feature of new data:");
        newDataTable.releaseBlockOfColumnValues(readFeatureIdx, firstReadRow, nNewVectors, dataFloat);

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

    private static void printFloatBuffer(FloatBuffer buf, long nColumns, long nRows, String message) {
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
