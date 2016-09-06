/* file: PackedTriangularMatrix.java */
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
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__PACKEDTRIANGULARMATRIX"></a>
 * @brief Class that provides methods to access triangular matrices.
 */
public class PackedTriangularMatrix extends NumericTable {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs packed triangular matrix using implementation provided by user
     *
     * @param context   Context to manage created packed triangular matrix
     * @param impl      Implementation of packed triangular matrix
     */
    public PackedTriangularMatrix(DaalContext context, PackedTriangularMatrixImpl impl) {
        super(context);
        tableImpl = impl;
    }

    /**
     * Constructs packed triangular matrix from the array of doubles
     *
     * @param context   Context to manage created packed triangular matrix
     * @param data      Array of size nVectors x nDim
     * @param nDim      Number of features in numeric table
     * @param layout    Data layout of the numeric table, NumericTable.StorageLayout.upperPackedTriangularMatrix or
     *                  NumericTable.StorageLayout.lowerPackedTriangularMatrix
     */
    public PackedTriangularMatrix(DaalContext context, double[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        if (layout == NumericTable.StorageLayout.upperPackedTriangularMatrix || layout == NumericTable.StorageLayout.lowerPackedTriangularMatrix) {
            tableImpl = new PackedTriangularMatrixArrayImpl(context, data, nDim, layout);
        } else {
            throw new IllegalArgumentException("wrong layout parameter in packed triangular matrix");
        }
    }

    /**
     * Constructs packed triangular matrix from the array of floats
     *
     * @param context   Context to manage created packed triangular matrix
     * @param data      Array of size nVectors x nDim
     * @param nDim      Number of features in numeric table
     * @param layout    Data layout of the numeric table, NumericTable.StorageLayout.upperPackedTriangularMatrix or
     *                  NumericTable.StorageLayout.lowerPackedTriangularMatrix
     */
    public PackedTriangularMatrix(DaalContext context, float[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        if (layout == NumericTable.StorageLayout.upperPackedTriangularMatrix || layout == NumericTable.StorageLayout.lowerPackedTriangularMatrix) {
            tableImpl = new PackedTriangularMatrixArrayImpl(context, data, nDim, layout);
        } else {
            throw new IllegalArgumentException("wrong layout parameter in packed triangular matrix");
        }
    }

    /**
     * Constructs packed triangular matrix from the array of longs
     *
     * @param context   Context to manage created packed triangular matrix
     * @param data      Array of size nVectors x nDim
     * @param nDim      Number of features in numeric table
     * @param layout    Data layout of the numeric table, NumericTable.StorageLayout.upperPackedTriangularMatrix or
     *                  NumericTable.StorageLayout.lowerPackedTriangularMatrix
     */
    public PackedTriangularMatrix(DaalContext context, long[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        if (layout == NumericTable.StorageLayout.upperPackedTriangularMatrix || layout == NumericTable.StorageLayout.lowerPackedTriangularMatrix) {
            tableImpl = new PackedTriangularMatrixArrayImpl(context, data, nDim, layout);
        } else {
            throw new IllegalArgumentException("wrong layout parameter in packed triangular matrix");
        }
    }

    /**
     * Constructs packed triangular matrix from the array of integers
     *
     * @param context   Context to manage created packed triangular matrix
     * @param data      Array of size nVectors x nDim
     * @param nDim      Number of features in numeric table
     * @param layout    Data layout of the numeric table, NumericTable.StorageLayout.upperPackedTriangularMatrix or
     *                  NumericTable.StorageLayout.lowerPackedTriangularMatrix
     */
    public PackedTriangularMatrix(DaalContext context, int[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        if (layout == NumericTable.StorageLayout.upperPackedTriangularMatrix || layout == NumericTable.StorageLayout.lowerPackedTriangularMatrix) {
            tableImpl = new PackedTriangularMatrixArrayImpl(context, data, nDim, layout);
        } else {
            throw new IllegalArgumentException("wrong layout parameter in packed triangular matrix");
        }
    }

    /**
     * Constructs packed triangular matrix from C++ packed triangular matrix
     * @param context   Context to manage created packed triangular matrix
     * @param cTable    Pointer to C++ numeric table
     */
    public PackedTriangularMatrix(DaalContext context, long cTable) {
        super(context);
        tableImpl = new PackedTriangularMatrixByteBufferImpl(context, cTable);
    }

    public PackedTriangularMatrix(DaalContext context, Class<? extends Number> cls, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        if (layout == NumericTable.StorageLayout.upperPackedTriangularMatrix || layout == NumericTable.StorageLayout.lowerPackedTriangularMatrix) {
            tableImpl = new PackedTriangularMatrixByteBufferImpl(context, cls, nDim, layout);
        } else {
            throw new IllegalArgumentException("wrong layout parameter in packed triangular matrix");
        }
    }

    public PackedTriangularMatrix(DaalContext context, Class<? extends Number> cls, long nDim, NumericTable.StorageLayout layout,
            AllocationFlag allocFlag) {
        super(context);
        if (layout == NumericTable.StorageLayout.upperPackedTriangularMatrix || layout == NumericTable.StorageLayout.lowerPackedTriangularMatrix) {
            tableImpl = new PackedTriangularMatrixByteBufferImpl(context, cls, nDim, layout, allocFlag);
        } else {
            throw new IllegalArgumentException("wrong layout parameter in packed triangular matrix");
        }
    }

    /**
     * Fills a numeric table with a constant
     *
     * @param constValue  Constant to initialize entries of the packed triangular matrix
     */
    public void assign(long constValue) {
        ((PackedTriangularMatrixImpl)tableImpl).assign(constValue);
    }

    /** @copydoc PackedTriangularMatrix::assign(long) */
    public void assign(int constValue) {
        ((PackedTriangularMatrixImpl)tableImpl).assign(constValue);
    }

    /** @copydoc PackedTriangularMatrix::assign(long) */
    public void assign(double constValue) {
        ((PackedTriangularMatrixImpl)tableImpl).assign(constValue);
    }

    /** @copydoc PackedTriangularMatrix::assign(long) */
    public void assign(float constValue) {
        ((PackedTriangularMatrixImpl)tableImpl).assign(constValue);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /**
     * Gets the whole packed array and returns it to java.nio.DoubleBuffer
     *
     * @param buf         Buffer to store results
     *
     * return DoubleBuffer containing whole packed array
     */
    public DoubleBuffer getPackedArray(DoubleBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getPackedArray(buf);
    }

    /**
     * Gets the whole packed array and returns it to java.nio.FloatBuffer
     *
     * @param buf         Buffer to store results
     *
     * return FloatBuffer containing whole packed array
     */
    public FloatBuffer getPackedArray(FloatBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getPackedArray(buf);
    }

    /**
     * Gets the whole packed array and returns it to java.nio.IntBuffer
     *
     * @param buf         Buffer to store results
     *
     * return IntBuffer containing whole packed array
     */
    public IntBuffer getPackedArray(IntBuffer buf) {
        return ((PackedTriangularMatrixImpl)tableImpl).getPackedArray(buf);
    }

    /**
     * Release a packed array from the input DoubleBuffer
     *
     * @param buf         Input DoubleBuffer with the capacity nDim * (nDim + 1) / 2, where
     *                    nDim is the matrix dimension
     */
    public void releasePackedArray(DoubleBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releasePackedArray(buf);
    }

    /**
     * Release a packed array from the input FloatBuffer
     *
     * @param buf         Input FloatBuffer with the capacity nDim * (nDim + 1) / 2, where
     *                    nDim is the matrix dimension
     */
    public void releasePackedArray(FloatBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releasePackedArray(buf);
    }

    /**
     * Release a packed array from the input IntBuffer
     *
     * @param buf         Input IntBuffer with the capacity nDim * (nDim + 1) / 2, where
     *                    nDim is the matrix dimension
     */
    public void releasePackedArray(IntBuffer buf) {
        ((PackedTriangularMatrixImpl)tableImpl).releasePackedArray(buf);
    }

    /**
     * Gets data as an Object
     * @return Table data as an Object
     */
    public Object getDataObject() {
        return ((PackedTriangularMatrixImpl)tableImpl).getDataObject();
    }

    /**
     * Gets numeric type of data stored in numeric table
     * @return Numeric type of table data
     */
    public Class<? extends Number> getNumericType() {
        return ((PackedTriangularMatrixImpl)tableImpl).getNumericType();
    }
}
