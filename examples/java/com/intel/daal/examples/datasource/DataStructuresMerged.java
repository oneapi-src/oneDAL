/* file: DataStructuresMerged.java */
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
    private static final int nFeatures1 = 6;
    private static final int nFeatures2 = 5;
    private static final int firstReadRow = 3;
    private static final int nRead = 1;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) {
        System.out.println("Merged numeric table example");
        int readFeatureIdx;

        double[] data1 = {
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5,
        };

        double[] data2 = {
            0.6, 0.7, 0.8, 0.9, 1,
            1.6, 1.7, 1.8, 1.9, 2,
            2.6, 2.7, 2.8, 2.9, 3,
            3.6, 3.7, 3.8, 3.9, 4,
            4.6, 4.7, 4.8, 4.9, 5,
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
        printDoubleBuffer(dataDouble, nFeatures1 + nFeatures2, nRead, "Print 1 row from merged numeric table:");

        /* Modify row of the merged numeric table */
        for (int i = 0; i < nFeatures1 + nFeatures2; i++) dataDouble.put(i, dataDouble.get(i) * dataDouble.get(i));
        dataTable.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        /* Read the same row from homogen numeric tables */
        dataDouble = dataTable1.getBlockOfRows(firstReadRow, nRead, dataDouble);
        printDoubleBuffer(dataDouble, nFeatures1, nRead, "Print 1 row from first homogen numeric table:");
        dataTable1.releaseBlockOfRows(firstReadRow, nRead, dataDouble);

        dataDouble = dataTable2.getBlockOfRows(firstReadRow, nRead, dataDouble);
        printDoubleBuffer(dataDouble, nFeatures2, nRead, "Print 1 row from second homogen numeric table:");
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
