/* file: PackedTriangularMatrixArrayImpl.java */
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

class PackedTriangularMatrixArrayImpl extends PackedTriangularMatrixImpl {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /** @copydoc PackedTriangularMatrix::PackedTriangularMatrix(DaalContext,double[],long) */
    public PackedTriangularMatrixArrayImpl(DaalContext context, double[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initialize(context, Double.class, data, nDim, layout);
    }

    /** @copydoc PackedTriangularMatrix::PackedTriangularMatrix(DaalContext,float[],long) */
    public PackedTriangularMatrixArrayImpl(DaalContext context, float[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initialize(context, Float.class, data, nDim, layout);
    }

    /** @copydoc PackedTriangularMatrix::PackedTriangularMatrix(DaalContext,long[],long,long) */
    public PackedTriangularMatrixArrayImpl(DaalContext context, long[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initialize(context, Long.class, data, nDim, layout);
    }

    /** @copydoc PackedTriangularMatrix::PackedTriangularMatrix(DaalContext,int[],long,long) */
    public PackedTriangularMatrixArrayImpl(DaalContext context, int[] data, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initialize(context, Integer.class, data, nDim, layout);
    }

    /** @copydoc PackedTriangularMatrix::assign(long) */
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

    /** @copydoc PackedTriangularMatrix::assign(int) */
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

    /** @copydoc PackedTriangularMatrix::assign(double) */
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

    /** @copydoc PackedTriangularMatrix::assign(float) */
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
        PackedTriangularMatrixUtils.TriangularUpCastIface triangularUpCast = PackedTriangularMatrixUtils.TriangularUpCast.getCast(type, double.class);
        triangularUpCast.upCast((int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularUpCastIface triangularUpCast = PackedTriangularMatrixUtils.TriangularUpCast.getCast(type, float.class);
        triangularUpCast.upCast((int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularUpCastIface triangularUpCast = PackedTriangularMatrixUtils.TriangularUpCast.getCast(type, int.class);
        triangularUpCast.upCast((int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularUpCastIface triangularUpCast = PackedTriangularMatrixUtils.TriangularUpCast.getCast(type, double.class);
        triangularUpCast.upCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularUpCastIface triangularUpCast = PackedTriangularMatrixUtils.TriangularUpCast.getCast(type, float.class);
        triangularUpCast.upCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularUpCastIface triangularUpCast = PackedTriangularMatrixUtils.TriangularUpCast.getCast(type, int.class);
        triangularUpCast.upCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, jData, buf, packedLayout);
        return buf;
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularDownCastIface triangularDownCast = PackedTriangularMatrixUtils.TriangularDownCast.getCast(double.class, type);
        triangularDownCast.downCast((int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularDownCastIface triangularDownCast = PackedTriangularMatrixUtils.TriangularDownCast.getCast(float.class, type);
        triangularDownCast.downCast((int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularDownCastIface triangularDownCast = PackedTriangularMatrixUtils.TriangularDownCast.getCast(int.class, type);
        triangularDownCast.downCast((int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularDownCastIface triangularDownCast = PackedTriangularMatrixUtils.TriangularDownCast.getCast(double.class, type);
        triangularDownCast.downCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularDownCastIface triangularDownCast = PackedTriangularMatrixUtils.TriangularDownCast.getCast(float.class, type);
        triangularDownCast.downCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        PackedTriangularMatrixUtils.TriangularDownCastIface triangularDownCast = PackedTriangularMatrixUtils.TriangularDownCast.getCast(int.class, type);
        triangularDownCast.downCastFeature((int)featureIndex, (int)vectorIndex, (int)vectorNum, nDim, buf, jData, packedLayout);
    }

    /** @copydoc PackedTriangularMatrix::getPackedArray(DoubleBuffer) */
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

    /** @copydoc PackedTriangularMatrix::getPackedArray(FloatBuffer) */
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

    /** @copydoc PackedTriangularMatrix::getPackedArray(IntBuffer) */
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

    /** @copydoc PackedTriangularMatrix::releasePackedArray(DoubleBuffer) */
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

    /** @copydoc PackedTriangularMatrix::releasePackedArray(FloatBuffer) */
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

    /** @copydoc PackedTriangularMatrix::releasePackedArray(IntBuffer) */
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

    /** @copydoc PackedTriangularMatrix::getDataObject */
    @Override
    public Object getDataObject() {
        return jData;
    }

    /** @copydoc PackedTriangularMatrix::getNumericType */
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
        cObject = newJavaNumericTable(nDim, nDim, layout);
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
        cObject = newJavaNumericTable(nJavaFeatures, nJavaVectors, packedLayout);
        dict = new DataDictionary(context, nJavaFeatures, cGetCDataDictionary(cObject));
        for (int i = 0; i < nJavaFeatures; i++) {
            dict.setFeature(type, i);
        }
    }

    /** @private */
    NumericTable.StorageLayout packedLayout;
}
