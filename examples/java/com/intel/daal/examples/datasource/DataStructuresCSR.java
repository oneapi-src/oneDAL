/* file: DataStructuresCSR.java */
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

/**
 * <a name="DAAL-EXAMPLE-JAVA-DATASTRUCTURESCSR">
 * @example DataStructuresCSR.java
 */

package com.intel.daal.examples.datasource;

import java.nio.DoubleBuffer;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/*
// Compressed spares rows (CSR) data structures example
*/

class DataStructuresCSR {
    private static final long nObservations = 5;
    private static final long nFeatures     = 5;
    private static final long firstReadRow  = 1;
    private static final long nRead         = 3;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) {
        System.out.println("Compressed spares rows (CSR) numeric table example");
        int readFeatureIdx;

        double values[]   = {1, -1, -3, -2,  5,  4,  6,  4, -4,  2,  7,  8, -5};
        long colIndices[] = {1,  2,  4,  1,  2,  3,  4,  5,  1,  3,  4,  2,  5};
        long rowOffsets[] = {1,          4,      6,          9,         12,     14};

        CSRNumericTable dataTable = new CSRNumericTable(context, values, colIndices, rowOffsets, nFeatures, nObservations);

        /* Read block of rows in dense format */
        DoubleBuffer dataDouble = DoubleBuffer.allocate((int)(nRead * nFeatures));
        dataDouble = dataTable.getBlockOfRows(firstReadRow, nRead, dataDouble);

        printDoubleBuffer(dataDouble, nFeatures, nRead,
                          "Print 3 rows from CSR data array as dense double array:");
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
