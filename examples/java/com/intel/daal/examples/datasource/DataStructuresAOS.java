/* file: DataStructuresAOS.java */
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
 //     Java example of using an array of structures (AOS)
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DATASTRUCTURESAOS">
 * @example DataStructuresAOS.java
 */

package com.intel.daal.examples.datasource;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import com.intel.daal.algorithms.cosdistance.*;
import com.intel.daal.data_management.data.AOSNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.examples.utils.Service;
import java.io.Serializable;
import com.intel.daal.services.DaalContext;

class DataStructuresAOS {
    private static final int nVectors = 5;
    private static final long nFeatures = 4;
    private static final long firstReadRow = 0;

    public static class PointType implements Serializable {
        public float x;
        public float y;
        public int    categ;
        public double  value;

        public PointType(float _x, float _y, int _categ, double _value) {
            x = _x;
            y = _y;
            categ = _categ;
            value = _value;
        }
    }

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        System.out.println("Array of structures (AOS) numeric table example\n");
        PointType points[] = new PointType[nVectors];
        points[0] = new PointType(0.5f, -1.3f, 1, 100.1);
        points[1] = new PointType(2.5f, -3.3f, 2, 200.0);
        points[2] = new PointType(4.5f, -5.3f, 2, 350.3);
        points[3] = new PointType(6.5f, -7.3f, 0, 470.4);
        points[4] = new PointType(8.5f, -9.3f, 1, 270.5);

        /*
         * Construct AOS numericTable for a data array with nFeaturesAOS fields
         * and nPoints elements
         */
        AOSNumericTable dataTable = new AOSNumericTable(context, points);

        /* Create an algorithm to compute a cosine distance matrix using the defaultDense method */
        Batch alg = new Batch(context, Float.class, Method.defaultDense);

        alg.input.set(InputId.data, dataTable);
        Result result = alg.compute();

        NumericTable res = result.get(ResultId.cosineDistance);

        /* Read a block of rows */
        DoubleBuffer dataDouble = DoubleBuffer.allocate((int) (nVectors * nFeatures));
        dataDouble = res.getBlockOfRows(firstReadRow, nVectors, dataDouble);
        printDoubleBuffer(dataDouble, nFeatures, nVectors, "Print AOS data structures as double:");
        res.releaseBlockOfRows(firstReadRow, nVectors, dataDouble);

        int readFeatureIdx = 2;
        IntBuffer dataInt = IntBuffer.allocate((int) (nVectors * nFeatures));
        dataInt = res.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nVectors, dataInt);
        printIntBuffer(dataInt, 1, nVectors, "Print the third feature of AOS:");
        res.releaseBlockOfRows(firstReadRow, nVectors, dataInt);


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

        private static void printIntBuffer(IntBuffer buf, long nColumns, long nRows, String message) {
        int step = (int) nColumns;
        System.out.println(message);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nColumns; j++) {
                System.out.format("%3d   ", buf.get(i * step + j));
            }
            System.out.println("");
        }
        System.out.println("");
    }
}
