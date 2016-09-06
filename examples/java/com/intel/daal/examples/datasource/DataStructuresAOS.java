/* file: DataStructuresAOS.java */
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
    private static final long nFeatures = 5;
    private static final long firstReadRow = 0;

    public static class PointType implements Serializable {
        public double x;
        public double y;
        public int    categ;
        public float  value;

        public PointType(double _x, double _y, int _categ, float _value) {
            x = _x;
            y = _y;
            categ = _categ;
            value = _value;
        }
    }

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        PointType points[] = new PointType[nVectors];
        points[0] = new PointType(0.5, -1.3, 1, 1.10f);
        points[1] = new PointType(2.5, -3.3, 2, 2.20f);
        points[2] = new PointType(4.5, -5.3, 2, 3.35f);
        points[3] = new PointType(6.5, -7.3, 0, 4.47f);
        points[4] = new PointType(8.5, -9.3, 1, 2.57f);

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

        Service.printNumericTable("Matrix of Distances:", res);

        /* Read a block of rows */
        DoubleBuffer dataDouble = DoubleBuffer.allocate((int) (nVectors * nFeatures));
        dataDouble = res.getBlockOfRows(firstReadRow, nVectors, dataDouble);
        printDoubleBuffer(dataDouble, nFeatures, nVectors, "Print AOS data structures as double:");
        res.releaseBlockOfRows(firstReadRow, nVectors, dataDouble);

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
