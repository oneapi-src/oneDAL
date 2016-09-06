/* file: HomogenNumericTable.java */
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

package com.intel.daal.data_management.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__HOMOGENNUMERICTABLE"></a>
 * @brief A derivative class of the NumericTable class, that provides methods to
 *        access the data that is stored as a contiguous array of homogeneous
 *        feature vectors. Table rows contain feature vectors, and columns
 *        contain values of individual features.
 */
public class HomogenNumericTable extends NumericTable {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs homogeneous numeric table using implementation provided by user
     *
     * @param context   Context to manage created homogeneous numeric table
     * @param impl      Implementation of homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, HomogenNumericTableImpl impl) {
        super(context);
        tableImpl = impl;
    }

    /**
     * Constructs homogeneous numeric table from the array of doubles
     *
     * @param context   Context to manage created homogeneous numeric table
     * @param data      Array of size nVectors x nFeatures
     * @param nFeatures Number of features in numeric table
     * @param nVectors  Number of feature vectors in numeric table
     */
    public HomogenNumericTable(DaalContext context, double[] data, long nFeatures, long nVectors) {
        super(context);
        tableImpl = new HomogenNumericTableArrayImpl(context, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of floats
     *
     * @param context   Context to manage created homogeneous numeric table
     * @param data      Array of size nVectors x nFeatures
     * @param nFeatures Number of features in numeric table
     * @param nVectors  Number of feature vectors in numeric table
     */
    public HomogenNumericTable(DaalContext context, float[] data, long nFeatures, long nVectors) {
        super(context);
        tableImpl = new HomogenNumericTableArrayImpl(context, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of longs
     *
     * @param context   Context to manage created homogeneous numeric table
     * @param data      Array of size nVectors x nFeatures
     * @param nFeatures Number of features in numeric table
     * @param nVectors  Number of feature vectors in numeric table
     */
    public HomogenNumericTable(DaalContext context, long[] data, long nFeatures, long nVectors) {
        super(context);
        tableImpl = new HomogenNumericTableArrayImpl(context, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of integers
     *
     * @param context   Context to manage created homogeneous numeric table
     * @param data      Array of size nVectors x nFeatures
     * @param nFeatures Number of features in numeric table
     * @param nVectors  Number of feature vectors in numeric table
     */
    public HomogenNumericTable(DaalContext context, int[] data, long nFeatures, long nVectors) {
        super(context);
        tableImpl = new HomogenNumericTableArrayImpl(context, data, nFeatures, nVectors);
    }

    /**
     * Constructs homogeneous numeric table from the array of doubles and filling the table with a constant
     *
     * @param context    Context to manage created homogeneous numeric table
     * @param data       Array of size nVectors x nFeatures
     * @param nFeatures  Number of features in numeric table
     * @param nVectors   Number of feature vectors in numeric table
     * @param constValue Constant to initialize entries of the homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, double[] data, long nFeatures, long nVectors, double constValue) {
        super(context);
        tableImpl = new HomogenNumericTableArrayImpl(context, data, nFeatures, nVectors, constValue);
    }

    /**
     * Constructs homogeneous numeric table from the array of floats and filling the table with a constant
     *
     * @param context    Context to manage created homogeneous numeric table
     * @param data       Array of size nVectors x nFeatures
     * @param nFeatures  Number of features in numeric table
     * @param nVectors   Number of feature vectors in numeric table
     * @param constValue Constant to initialize entries of the homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, float[] data, long nFeatures, long nVectors, float constValue) {
        super(context);
        tableImpl = new HomogenNumericTableArrayImpl(context, data, nFeatures, nVectors, constValue);
    }

    /**
     * Constructs homogeneous numeric table from the array of longs and filling the table with a constant
     *
     * @param context    Context to manage created homogeneous numeric table
     * @param data       Array of size nVectors x nFeatures
     * @param nFeatures  Number of features in numeric table
     * @param nVectors   Number of feature vectors in numeric table
     * @param constValue Constant to initialize entries of the homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, long[] data, long nFeatures, long nVectors, long constValue) {
        super(context);
        tableImpl = new HomogenNumericTableArrayImpl(context, data, nFeatures, nVectors, constValue);
    }

    /**
     * Constructs homogeneous numeric table from the array of integers and filling the table with a constant
     *
     * @param context    Context to manage created homogeneous numeric table
     * @param data       Array of size nVectors x nFeatures
     * @param nFeatures  Number of features in numeric table
     * @param nVectors   Number of feature vectors in numeric table
     * @param constValue Constant to initialize entries of the homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, int[] data, long nFeatures, long nVectors, int constValue) {
        super(context);
        tableImpl = new HomogenNumericTableArrayImpl(context, data, nFeatures, nVectors, constValue);
    }

    /**
     * Constructs homogeneous numeric table from C++ homogeneous numeric
     *        table
     * @param context   Context to manage created homogeneous numeric table
     * @param cTable    Pointer to C++ numeric table
     */
    public HomogenNumericTable(DaalContext context, long cTable) {
        super(context);
        tableImpl = new HomogenNumericTableByteBufferImpl(context, cTable);
    }

    /**
     * Constructs homogeneous numeric table without memory allocation
     *
     * @param context                 Context to manage created homogeneous numeric table
     * @param cls                     Numeric type of values in the table
     * @param nColumns                Number of columns in the table
     */
    public HomogenNumericTable(DaalContext context, Class<? extends Number> cls, long nColumns) {
        super(context);
        tableImpl = new HomogenNumericTableByteBufferImpl(context, cls, nColumns);
    }

    /**
     * Constructs homogeneous numeric table with memory allocation controlled via a flag
     *
     * @param context                 Context to manage created homogeneous numeric table
     * @param cls                     Numeric type of values in the table
     * @param nColumns                Number of columns in the table
     * @param nRows                   Number of rows in the table
     * @param allocFlag               Flag that controls internal memory allocation for data in the numeric table
     */
    public HomogenNumericTable(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            AllocationFlag allocFlag) {
        super(context);
        tableImpl = new HomogenNumericTableByteBufferImpl(context, cls, nColumns, nRows, allocFlag);
    }

    /**
     * Constructs homogeneous numeric table with memory allocation controlled via a flag and filling the table with a constant
     *
     * @param context                 Context to manage created homogeneous numeric table
     * @param cls                     Numeric type of values in the table
     * @param nColumns                Number of columns in the table
     * @param nRows                   Number of rows in the table
     * @param allocFlag               Flag that controls internal memory allocation for data in the numeric table
     * @param constValue              Constant to initialize entries of the homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            AllocationFlag allocFlag, double constValue) {
        super(context);
        tableImpl = new HomogenNumericTableByteBufferImpl(context, cls, nColumns, nRows, allocFlag, constValue);
    }

    /**
     * Constructs homogeneous numeric table with memory allocation controlled via a flag and filling the table with a constant
     *
     * @param context                 Context to manage created homogeneous numeric table
     * @param cls                     Numeric type of values in the table
     * @param nColumns                Number of columns in the table
     * @param nRows                   Number of rows in the table
     * @param allocFlag               Flag that controls internal memory allocation for data in the numeric table
     * @param constValue              Constant to initialize entries of the homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            AllocationFlag allocFlag, float constValue) {
        super(context);
        tableImpl = new HomogenNumericTableByteBufferImpl(context, cls, nColumns, nRows, allocFlag, constValue);
    }

    /**
     * Constructs homogeneous numeric table with memory allocation controlled via a flag and filling the table with a constant
     *
     * @param context                 Context to manage created homogeneous numeric table
     * @param cls                     Numeric type of values in the table
     * @param nColumns                Number of columns in the table
     * @param nRows                   Number of rows in the table
     * @param allocFlag               Flag that controls internal memory allocation for data in the numeric table
     * @param constValue              Constant to initialize entries of the homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            AllocationFlag allocFlag, long constValue) {
        super(context);
        tableImpl = new HomogenNumericTableByteBufferImpl(context, cls, nColumns, nRows, allocFlag, constValue);
    }

    /**
     * Constructs homogeneous numeric table with memory allocation controlled via a flag and filling the table with a constant
     *
     * @param context                 Context to manage created homogeneous numeric table
     * @param cls                     Numeric type of values in the table
     * @param nColumns                Number of columns in the table
     * @param nRows                   Number of rows in the table
     * @param allocFlag               Flag that controls internal memory allocation for data in the numeric table
     * @param constValue              Constant to initialize entries of the homogeneous numeric table
     */
    public HomogenNumericTable(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            AllocationFlag allocFlag, int constValue) {
        super(context);
        tableImpl = new HomogenNumericTableByteBufferImpl(context, cls, nColumns, nRows, allocFlag, constValue);
    }

    /**
     * Constructs an empty Numeric Table with a predefined Data Dictionary
     *
     * @param context                 Context to manage created homogeneous numeric table
     * @param cls                     Numeric type of values in the table
     * @param dict                    Predefined Data Dictionary
     */
    public HomogenNumericTable(DaalContext context, Class<? extends Number> cls, DataDictionary dict) {
        super(context);
        tableImpl = new HomogenNumericTableByteBufferImpl(context, cls, dict);
    }

    /**
     * Fills a numeric table with a constant
     *
     * @param constValue  Constant to initialize entries of the homogeneous numeric table
     */
    public void assign(long constValue) {
        ((HomogenNumericTableImpl)tableImpl).assign(constValue);
    }

    /** @copydoc HomogenNumericTable::assign(long) */
    public void assign(int constValue) {
        ((HomogenNumericTableImpl)tableImpl).assign(constValue);
    }

    /** @copydoc HomogenNumericTable::assign(long) */
    public void assign(double constValue) {
        ((HomogenNumericTableImpl)tableImpl).assign(constValue);
    }

    /** @copydoc HomogenNumericTable::assign(long) */
    public void assign(float constValue) {
        ((HomogenNumericTableImpl)tableImpl).assign(constValue);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        return ((HomogenNumericTableImpl)tableImpl).getBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        return ((HomogenNumericTableImpl)tableImpl).getBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        return ((HomogenNumericTableImpl)tableImpl).getBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        return ((HomogenNumericTableImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        return ((HomogenNumericTableImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        return ((HomogenNumericTableImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        ((HomogenNumericTableImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        ((HomogenNumericTableImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        ((HomogenNumericTableImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        ((HomogenNumericTableImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        ((HomogenNumericTableImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        ((HomogenNumericTableImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /**
     * Gets data as an array of doubles
     * @return Table data as an array of double
     */
    public double[] getDoubleArray() {
        return ((HomogenNumericTableImpl)tableImpl).getDoubleArray();
    }

    /**
     * Gets data as an array of floats
     * @return Table data as an array of floats
     */
    public float[] getFloatArray() {
        return ((HomogenNumericTableImpl)tableImpl).getFloatArray();
    }

    /**
     * Gets data as an array of longs
     * @return Table data as an array of longs
     */
    public long[] getLongArray() {
        return ((HomogenNumericTableImpl)tableImpl).getLongArray();
    }

    /**
     * Gets data as an Object
     * @return Table data as an Object
     */
    public Object getDataObject() {
        return ((HomogenNumericTableImpl)tableImpl).getDataObject();
    }

    /**
     * Gets numeric type of data stored in numeric table
     * @return Numeric type of table data
     */
    public Class<? extends Number> getNumericType() {
        return ((HomogenNumericTableImpl)tableImpl).getNumericType();
    }
}
