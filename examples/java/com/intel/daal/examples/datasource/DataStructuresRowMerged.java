/* file: DataStructuresRowMerged.java */
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
 * <a name="DAAL-EXAMPLE-JAVA-DATASTRUCTURESROWMERGED">
 * @example DataStructuresRowMerged.java
 */

package com.intel.daal.examples.datasource;

import java.nio.FloatBuffer;
import com.intel.daal.data_management.data.DataDictionary;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.RowMergedNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/*
// RowMerged data structures example
*/

class DataStructuresRowMerged {
    private static final int nVectors1 = 5;
    private static final int nVectors2 = 6;
    private static final int nFeatures = 5;
    private static final int firstReadRow = 3;
    private static final int nRead = 6;
    private static final int featureIdx = 2;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) {
        System.out.println("Row merged numeric table example\n");
        int readFeatureIdx;

        float[] data1 = {
            0.0f, 0.1f, 0.2f, 0.3f, 0.4f,
            1.0f, 1.1f, 1.2f, 1.3f, 1.4f,
            2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
            3.0f, 3.1f, 3.2f, 3.3f, 3.4f,
            4.0f, 4.1f, 4.2f, 4.3f, 4.4f,
        };

        float[] data2 = {
            0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
            1.5f, 1.6f, 1.7f, 1.8f, 1.9f,
            2.5f, 2.6f, 2.7f, 2.8f, 2.9f,
            3.5f, 3.6f, 3.7f, 3.8f, 3.9f,
            4.5f, 4.6f, 4.7f, 4.8f, 4.9f,
            5.5f, 5.6f, 5.7f, 5.8f, 5.9f,
        };

        /* Create two homogen numeric tables from data arrays */
        HomogenNumericTable dataTable1 = new HomogenNumericTable(context, DataDictionary.FeaturesEqual.equal, data1, nFeatures, nVectors1);
        HomogenNumericTable dataTable2 = new HomogenNumericTable(context, DataDictionary.FeaturesEqual.equal, data2, nFeatures, nVectors2);

        /* Create rowmerged numeric table consisting of two homogen numeric tables */
        RowMergedNumericTable dataTable = new RowMergedNumericTable(context);
        dataTable.addNumericTable(dataTable1);
        dataTable.addNumericTable(dataTable2);

        FloatBuffer dataFloat = FloatBuffer.allocate(0);

        /* Read one row from rowmerged numeric table */
        dataFloat = dataTable.getBlockOfRows(0, nVectors1 + nVectors2, dataFloat);
        printFloatBuffer(dataFloat, nFeatures, nVectors1 + nVectors2, "Print rows from row merged numeric table as float:");

        /* Modify row of the rowmerged numeric table */
        for (int i = 0; i < nVectors1 + nVectors2; i++) dataFloat.put(i*nFeatures+featureIdx, dataFloat.get(i*nFeatures+featureIdx) * dataFloat.get(i*nFeatures+featureIdx));
        dataTable.releaseBlockOfRows(0, nVectors1 + nVectors2, dataFloat);

        /* Read rows from homogen numeric tables */
        dataFloat = dataTable.getBlockOfRows(0, nVectors1 + nVectors2, dataFloat);
        printFloatBuffer(dataFloat, nFeatures, nVectors1 + nVectors2, "Print rows from row merged numeric table as float:");
        dataTable.releaseBlockOfRows(0, nVectors1 + nVectors2, dataFloat);

        dataFloat = dataTable.getBlockOfRows(0, nVectors1 + nVectors2, dataFloat);
        printFloatBuffer(dataFloat, nFeatures, nVectors1 + nVectors2, "Row merged table converted to homogen numeric table");
        dataTable.releaseBlockOfRows(0, nVectors1 + nVectors2, dataFloat);

        context.dispose();
    }

    private static void printFloatBuffer (FloatBuffer buf, int nColumns, int nRows, String message) {
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
