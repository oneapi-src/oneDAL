/* file: PackedSymmetricMatrixArrayImpl.java */
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

import com.intel.daal.services.DaalContext;
import com.intel.daal.SerializationTag;

class PackedSymmetricMatrixArrayImpl extends PackedSymmetricMatrixImpl {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @copydoc PackedSymmetricMatrix::PackedSymmetricMatrix(DaalContext,double[],long) */
    public PackedSymmetricMatrixArrayImpl(DaalContext context, double[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initialize(context, Double.class, data, nDim, layout);
    }

    /** @copydoc PackedSymmetricMatrix::PackedSymmetricMatrix(DaalContext,float[],long) */
    public PackedSymmetricMatrixArrayImpl(DaalContext context, float[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initialize(context, Float.class, data, nDim, layout);
    }

    /** @copydoc PackedSymmetricMatrix::PackedSymmetricMatrix(DaalContext,long[],long,long) */
    public PackedSymmetricMatrixArrayImpl(DaalContext context, long[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initialize(context, Long.class, data, nDim, layout);
    }

    /** @copydoc PackedSymmetricMatrix::PackedSymmetricMatrix(DaalContext,int[],long,long) */
    public PackedSymmetricMatrixArrayImpl(DaalContext context, int[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initialize(context, Integer.class, data, nDim, layout);
    }

    /** @copydoc PackedSymmetricMatrix::assign(long) */
    @Override
    public void assign(long constValue) {
        if (type != Long.class) {
            throw new IllegalArgumentException("can not assign Long type to table of " + type);
        }
        else {
            int nDim = (int) getNumberOfRows();
            int nSize = (nDim * (nDim + 1)) / 2;
            long[] data = (long[])jData;
            for(int i = 0; i < nSize; i++) {
                data[i] = constValue;
            }
        }
    }

    /** @copydoc PackedSymmetricMatrix::assign(int) */
    @Override
    public void assign(int constValue) {
        if (type != Integer.class) {
            throw new IllegalArgumentException("can not assign Integer type to table of " + type);
        }
        else {
            int nDim = (int) getNumberOfRows();
            int nSize = (nDim * (nDim + 1)) / 2;
            int[] data = (int[])jData;
            for(int i = 0; i < nSize; i++) {
                data[i] = constValue;
            }
        }
    }

    /** @copydoc PackedSymmetricMatrix::assign(double) */
    @Override
    public void assign(double constValue) {
        if (type != Double.class) {
            throw new IllegalArgumentException("can not assign Double type to table of " + type);
        }
        else {
            int nDim = (int) getNumberOfRows();
            int nSize = (nDim * (nDim + 1)) / 2;
            double[] data = (double[])jData;
            for(int i = 0; i < nSize; i++) {
                data[i] = constValue;
            }
        }
    }

    /** @copydoc PackedSymmetricMatrix::assign(float) */
    @Override
    public void assign(float constValue) {
        if (type != Float.class) {
            throw new IllegalArgumentException("can not assign Float type to table of " + type);
        }
        else {
            int nDim = (int) getNumberOfRows();
            int nSize = (nDim * (nDim + 1)) / 2;
            float[] data = (float[])jData;
            for(int i = 0; i < nSize; i++) {
                data[i] = constValue;
            }
        }
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricUpCastIface symmetricUpCast = PackedSymmetricMatrixUtils.SymmetricUpCast.getCast(type, double.class);
        symmetricUpCast.upCast((int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricUpCastIface symmetricUpCast = PackedSymmetricMatrixUtils.SymmetricUpCast.getCast(type, float.class);
        symmetricUpCast.upCast((int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricUpCastIface symmetricUpCast = PackedSymmetricMatrixUtils.SymmetricUpCast.getCast(type, int.class);
        symmetricUpCast.upCast((int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricUpCastIface symmetricUpCast = PackedSymmetricMatrixUtils.SymmetricUpCast.getCast(type, double.class);
        symmetricUpCast.upCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricUpCastIface symmetricUpCast = PackedSymmetricMatrixUtils.SymmetricUpCast.getCast(type, float.class);
        symmetricUpCast.upCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricUpCastIface symmetricUpCast = PackedSymmetricMatrixUtils.SymmetricUpCast.getCast(type, int.class);
        symmetricUpCast.upCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricDownCastIface symmetricDownCast = PackedSymmetricMatrixUtils.SymmetricDownCast.getCast(double.class, type);
        symmetricDownCast.downCast((int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricDownCastIface symmetricDownCast = PackedSymmetricMatrixUtils.SymmetricDownCast.getCast(float.class, type);
        symmetricDownCast.downCast((int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricDownCastIface symmetricDownCast = PackedSymmetricMatrixUtils.SymmetricDownCast.getCast(int.class, type);
        symmetricDownCast.downCast((int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricDownCastIface symmetricDownCast = PackedSymmetricMatrixUtils.SymmetricDownCast.getCast(double.class, type);
        symmetricDownCast.downCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricDownCastIface symmetricDownCast = PackedSymmetricMatrixUtils.SymmetricDownCast.getCast(float.class, type);
        symmetricDownCast.downCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedSymmetricMatrixUtils.SymmetricDownCastIface symmetricDownCast = PackedSymmetricMatrixUtils.SymmetricDownCast.getCast(int.class, type);
        symmetricDownCast.downCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc PackedSymmetricMatrix::getPackedArray(DoubleBuffer) */
    @Override
    public DoubleBuffer getPackedArray(DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (int) (nDim * (nDim + 1)) /  2;

        // Copies data into NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, double.class);
        vectorUpCast.upCast(bufferSize, 0, jData, buf);

        return buf;
    }

    /** @copydoc PackedSymmetricMatrix::getPackedArray(FloatBuffer) */
    @Override
    public FloatBuffer getPackedArray(FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (int) (nDim * (nDim + 1)) /  2;

        // Copies data into NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, float.class);
        vectorUpCast.upCast(bufferSize, 0, jData, buf);

        return buf;
    }

    /** @copydoc PackedSymmetricMatrix::getPackedArray(IntBuffer) */
    @Override
    public IntBuffer getPackedArray(IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (int) (nDim * (nDim + 1)) /  2;

        // Copies data into NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, int.class);
        vectorUpCast.upCast(bufferSize, 0, jData, buf);

        return buf;
    }

    /** @copydoc PackedSymmetricMatrix::releasePackedArray(DoubleBuffer) */
    @Override
    public void releasePackedArray(DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (int) (nDim * (nDim + 1)) / 2;

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(double.class, df.type);
        vectorDownCast.downCast(bufferSize, 0, buf, jData);
    }

    /** @copydoc PackedSymmetricMatrix::releasePackedArray(FloatBuffer) */
    @Override
    public void releasePackedArray(FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (int) (nDim * (nDim + 1)) / 2;

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(float.class, df.type);
        vectorDownCast.downCast(bufferSize, 0, buf, jData);
    }

    /** @copydoc PackedSymmetricMatrix::releasePackedArray(IntBuffer) */
    @Override
    public void releasePackedArray(IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (int) (nDim * (nDim + 1)) / 2;

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(int.class, df.type);
        vectorDownCast.downCast(bufferSize, 0, buf, jData);
    }

    /** @copydoc PackedSymmetricMatrix::getDataObject */
    @Override
    public Object getDataObject() {
        return jData;
    }

    /** @copydoc PackedSymmetricMatrix::getNumericType */
    @Override
    public Class<? extends Number> getNumericType() {
        return type;
    }

    /** @copydoc NumericTable::allocateDataMemory() */
    @Override
    public void allocateDataMemory() {
        throw new IllegalArgumentException("can not allocate data memory in Numeric Table with data on Java side");
    }

    /** @copydoc NumericTable::freeDataMemory() */
    @Override
    public void freeDataMemory() {
    }

    private void initialize(DaalContext context, Class<? extends Number> cls, Object data, long nDim,
                            NumericTable.StorageLayout layout) {
        type = cls;
        cObject = newJavaNumericTable(nDim, nDim, layout, DataDictionary.FeaturesEqual.notEqual,
                                      SerializationTag.SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID);
        dict = new DataDictionary(context, nDim, cGetCDataDictionary(cObject));
        for (int i = 0; i < nDim; i++) {
            dict.setFeature(cls, i);
        }
        nJavaFeatures = nDim;
        nJavaVectors = nDim;
        jData = data;
        dataAllocatedInJava = true;

        packedLayout = layout;
    }

    @Override
    protected void onUnpack(DaalContext context) {
        if (cObject == 0) {
            cObject = newJavaNumericTable(nJavaFeatures, nJavaVectors, packedLayout, DataDictionary.FeaturesEqual.notEqual,
                                          SerializationTag.SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID);
        }
        dict = new DataDictionary(context, nJavaFeatures, cGetCDataDictionary(cObject));
        for (int i = 0; i < nJavaFeatures; i++) {
            dict.setFeature(type, i);
        }
    }

    /** @private */
    NumericTable.StorageLayout packedLayout;
}
/** @} */
