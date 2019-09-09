/* file: DataStructuresMerged.java */
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

/**
 * <a name="DAAL-EXAMPLE-JAVA-DATASTRUCTURESMERGED">
 * @example DataStructuresMerged.java
 */

package com.intel.daal.examples.datasource;

import java.nio.DoubleBuffer;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/*
// Merged data structures example
*/

class DataStructuresMerged {
    private static final int nVectors = 5;
    private static final int nFeatures1 = 5;
    private static final int nFeatures2 = 6;
    private static final int firstReadRow = 3;
    private static final int nRead = 1;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) {
        System.out.println("Merged numeric table example\n");
        int readFeatureIdx;

        float[] data1 = {
            0.0f, 0.1f, 0.2f, 0.3f, 0.4f,
            1.0f, 1.1f, 1.2f, 1.3f, 1.4f,
            2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
            3.0f, 3.1f, 3.2f, 3.3f, 3.4f,
            4.0f, 4.1f, 4.2f, 4.3f, 4.4f,
        };

        float[] data2 = {
            0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1,
            1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2,
            2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3,
            3.5f, 3.6f, 3.7f, 3.8f, 3.9f, 4,
            4.5f, 4.6f, 4.7f, 4.8f, 4.9f, 5,
        };

        /* Create two homogen numeric tables from data arrays */
        HomogenNumericTable dataTable1 = new HomogenNumericTable(context, data1, nFeatures1, nVectors);
        HomogenNumericTable dataTable2 = new HomogenNumericTable(context, data2, nFeatures2, nVectors);

        /* Create merged numeric table consisting of two homogen numeric tables */
        MergedNumericTable dataTable = new MergedNumericTable(context);
        dataTable.addNumericTable(dataTable1);
        dataTable.addNumericTable(dataTable2);

        DoubleBuffer dataDouble = DoubleBuffer.allocate(0);

        /* Read one row from merged numeric table */
        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);
        printDoubleBuffer(dataDouble, nFeatures1 + nFeatures2, nRead, "Print 1 row from merged numeric table as float:");

        /* Modify row of the merged numeric table */
        for (int i = 0; i < nFeatures1 + nFeatures2; i++) dataDouble.put(i, dataDouble.get(i) * dataDouble.get(i));
        dataTable.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        /* Read the same row from homogen numeric tables */
        dataDouble = dataTable1.getBlockOfRows(firstReadRow, nRead, dataDouble);
        printDoubleBuffer(dataDouble, nFeatures1, nRead, "Print 1 row from first homogen numeric table as float:");
        dataTable1.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        dataDouble = dataTable2.getBlockOfRows(firstReadRow, nRead, dataDouble);
        printDoubleBuffer(dataDouble, nFeatures2, nRead, "Print 1 row from second homogen numeric table as float:");
        dataTable2.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        context.dispose();
    }

    private static void printDoubleBuffer (DoubleBuffer buf, int nColumns, int nRows, String message) {
        int step = (int)nColumns;
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
