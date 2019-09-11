/* file: Matrix.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @ingroup numeric_tables
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__MATRIX"></a>
 * @brief A derivative class of the NumericTable class, that provides methods to
 *        access the data that is stored as a contiguous array of homogeneous
 *        feature vectors. Table rows contain feature vectors, and columns
 *        contain values of individual features.
 */
public class Matrix extends HomogenNumericTable {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs homogeneous numeric table using implementation provided by user
     *
     * @param context   Context to manage created matrix
     * @param impl      Implementation of matrix
     */
    public Matrix(DaalContext context, HomogenNumericTableImpl impl) {
        super(context, impl);
    }

    /**
     * Constructs homogeneous numeric table from the array of doubles
     *
     * @param context   Context to manage created matrix
     * @param data      Array of size nVectors x nFeatures
     * @param nFeatures Number of features in numeric table
     * @param nVectors  Number of feature vectors in numeric table
     */
    public Matrix(DaalContext context, double[] data, long nFeatures, long nVectors) {
        super(context, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of doubles
     *
     * @param context        Context to manage created matrix
     * @param featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     * @param data           Array of size nVectors x nFeatures
     * @param nFeatures      Number of features in numeric table
     * @param nVectors       Number of feature vectors in numeric table
     */
    public Matrix(DaalContext context, DataDictionary.FeaturesEqual featuresEqual, double[] data, long nFeatures, long nVectors) {
        super(context, featuresEqual, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of floats
     *
     * @param context   Context to manage created matrix
     * @param data      Array of size nVectors x nFeatures
     * @param nFeatures Number of features in numeric table
     * @param nVectors  Number of feature vectors in numeric table
     */
    public Matrix(DaalContext context, float[] data, long nFeatures, long nVectors) {
        super(context, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of floats
     *
     * @param context        Context to manage created matrix
     * @param featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     * @param data           Array of size nVectors x nFeatures
     * @param nFeatures      Number of features in numeric table
     * @param nVectors       Number of feature vectors in numeric table
     */
    public Matrix(DaalContext context, DataDictionary.FeaturesEqual featuresEqual, float[] data, long nFeatures, long nVectors) {
        super(context, featuresEqual, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of longs
     *
     * @param context   Context to manage created matrix
     * @param data      Array of size nVectors x nFeatures
     * @param nFeatures Number of features in numeric table
     * @param nVectors  Number of feature vectors in numeric table
     */
    public Matrix(DaalContext context, long[] data, long nFeatures, long nVectors) {
        super(context, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of longs
     *
     * @param context        Context to manage created matrix
     * @param featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     * @param data           Array of size nVectors x nFeatures
     * @param nFeatures      Number of features in numeric table
     * @param nVectors       Number of feature vectors in numeric table
     */
    public Matrix(DaalContext context, DataDictionary.FeaturesEqual featuresEqual, long[] data, long nFeatures, long nVectors) {
        super(context, featuresEqual, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of integers
     *
     * @param context   Context to manage created matrix
     * @param data      Array of size nVectors x nFeatures
     * @param nFeatures Number of features in numeric table
     * @param nVectors  Number of feature vectors in numeric table
     */
    public Matrix(DaalContext context, int[] data, long nFeatures, long nVectors) {
        super(context, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of integers
     *
     * @param context        Context to manage created matrix
     * @param featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     * @param data           Array of size nVectors x nFeatures
     * @param nFeatures      Number of features in numeric table
     * @param nVectors       Number of feature vectors in numeric table
     */
    public Matrix(DaalContext context, DataDictionary.FeaturesEqual featuresEqual, int[] data, long nFeatures, long nVectors) {
        super(context, featuresEqual, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from C++ homogeneous numeric
     *        table
     * @param context   Context to manage created matrix
     * @param cTable    Pointer to C++ numeric table
     */
    public Matrix(DaalContext context, long cTable) {
        super(context, cTable);
    }

    public Matrix(DaalContext context, Class<? extends Number> cls, long nColumns) {
        super(context, cls, nColumns);
    }

    public Matrix(DaalContext context, DataDictionary.FeaturesEqual featuresEqual, Class<? extends Number> cls, long nColumns) {
        super(context, featuresEqual, cls, nColumns);
    }

    public Matrix(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            AllocationFlag allocFlag) {
        super(context, cls, nColumns, nRows, allocFlag);
    }

    public Matrix(DaalContext context, DataDictionary.FeaturesEqual featuresEqual, Class<? extends Number> cls, long nColumns, long nRows,
            AllocationFlag allocFlag) {
        super(context, featuresEqual, cls, nColumns, nRows, allocFlag);
    }

    public void set(long row, long column, double value) {
        ((HomogenNumericTableImpl)tableImpl).set(row, column, value);
    }

    public void set(long row, long column, float value) {
        ((HomogenNumericTableImpl)tableImpl).set(row, column, value);
    }

    public void set(long row, long column, long value) {
        ((HomogenNumericTableImpl)tableImpl).set(row, column, value);
    }

    public void set(long row, long column, int value) {
        ((HomogenNumericTableImpl)tableImpl).set(row, column, value);
    }

    public double getDouble(long row, long column) {
        return ((HomogenNumericTableImpl)tableImpl).getDouble(row, column);
    }

    public float getFloat(long row, long column) {
        return ((HomogenNumericTableImpl)tableImpl).getFloat(row, column);
    }

    public long getLong(long row, long column) {
        return ((HomogenNumericTableImpl)tableImpl).getLong(row, column);
    }

    public int getInt(long row, long column) {
        return ((HomogenNumericTableImpl)tableImpl).getInt(row, column);
    }
}
/** @} */
