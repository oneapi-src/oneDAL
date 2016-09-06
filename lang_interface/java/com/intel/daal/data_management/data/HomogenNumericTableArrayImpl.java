/* file: HomogenNumericTableArrayImpl.java */
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
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.lang.reflect.Array;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA__HOMOGENNUMERICTABLEARRAYIMPL__HOMOGENNUMERICTABLEARRAYIMPL"></a>
 * @brief A derivative class of the HomogenNumericTableImpl class, that provides implementation
 *        of a homogen numeric table with data stored as array of primitives
 */
class HomogenNumericTableArrayImpl extends HomogenNumericTableImpl {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,double[],long,long) */
    public HomogenNumericTableArrayImpl(DaalContext context, double[] data, long nFeatures, long nVectors) {
        super(context);
        initialize(context, Double.class, data, nFeatures, nVectors);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,float[],long,long) */
    public HomogenNumericTableArrayImpl(DaalContext context, float[] data, long nFeatures, long nVectors) {
        super(context);
        initialize(context, Float.class, data, nFeatures, nVectors);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,long[],long,long) */
    public HomogenNumericTableArrayImpl(DaalContext context, long[] data, long nFeatures, long nVectors) {
        super(context);
        initialize(context, Long.class, data, nFeatures, nVectors);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,int[],long,long) */
    public HomogenNumericTableArrayImpl(DaalContext context, int[] data, long nFeatures, long nVectors) {
        super(context);
        initialize(context, Integer.class, data, nFeatures, nVectors);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,double[],long,long,double) */
    public HomogenNumericTableArrayImpl(DaalContext context, double[] data, long nFeatures, long nVectors, double constValue) {
        super(context);
        initialize(context, Double.class, data, nFeatures, nVectors);
        assign (constValue);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,float[],long,long,float) */
    public HomogenNumericTableArrayImpl(DaalContext context, float[] data, long nFeatures, long nVectors, float constValue) {
        super(context);
        initialize(context, Float.class, data, nFeatures, nVectors);
        assign (constValue);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,long[],long,long,long) */
    public HomogenNumericTableArrayImpl(DaalContext context, long[] data, long nFeatures, long nVectors, long constValue) {
        super(context);
        initialize(context, Long.class, data, nFeatures, nVectors);
        assign (constValue);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,int[],long,long,int) */
    public HomogenNumericTableArrayImpl(DaalContext context, int[] data, long nFeatures, long nVectors, int constValue) {
        super(context);
        initialize(context, Integer.class, data, nFeatures, nVectors);
        assign (constValue);
    }

    /** @copydoc HomogenNumericTable::assign(long) */
    @Override
    public void assign(long constValue) {
        int nRows = (int) getNumberOfRows();
        int nColumns = (int) getNumberOfColumns();
        if (type != Long.class) {
            for(int i = 0; i < nRows * nColumns; i++) {
                java.lang.reflect.Array.setLong(jData, i, constValue);
            }
        }
        else {
            long[] data = (long[])jData;
            for(int i = 0; i < nRows * nColumns; i++) {
                data[i] = constValue;
            }
        }
    }

    /** @copydoc HomogenNumericTable::assign(int) */
    @Override
    public void assign(int constValue) {
        int nRows = (int) getNumberOfRows();
        int nColumns = (int) getNumberOfColumns();
        if (type != Integer.class) {
            for(int i = 0; i < nRows * nColumns; i++) {
                java.lang.reflect.Array.setInt(jData, i, constValue);
            }
        }
        else {
            int[] data = (int[])jData;
            for(int i = 0; i < nRows * nColumns; i++) {
                data[i] = constValue;
            }
        }
    }

    /** @copydoc HomogenNumericTable::assign(double) */
    @Override
    public void assign(double constValue) {
        int nRows = (int) getNumberOfRows();
        int nColumns = (int) getNumberOfColumns();
        if (type != Double.class) {
            for(int i = 0; i < nRows * nColumns; i++) {
                java.lang.reflect.Array.setDouble(jData, i, constValue);
            }
        }
        else {
            double[] data = (double[])jData;
            for(int i = 0; i < nRows * nColumns; i++) {
                data[i] = constValue;
            }
        }
    }

    /** @copydoc HomogenNumericTable::assign(float) */
    @Override
    public void assign(float constValue) {
        int nRows = (int) getNumberOfRows();
        int nColumns = (int) getNumberOfColumns();
        if (type != Float.class) {
            for(int i = 0; i < nRows * nColumns; i++) {
                java.lang.reflect.Array.setFloat(jData, i, constValue);
            }
        }
        else {
            float[] data = (float[])jData;
            for(int i = 0; i < nRows * nColumns; i++) {
                data[i] = constValue;
            }
        }
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);
        int shift = (int) (vectorIndex * nColumns);

        // Copies data into NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, double.class);
        vectorUpCast.upCast(bufferSize, shift, jData, buf);

        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);
        int shift = (int) (vectorIndex * nColumns);

        // Copies data into NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, float.class);
        vectorUpCast.upCast(bufferSize, shift, jData, buf);

        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);
        int shift = (int) (vectorIndex * nColumns);

        // Copies data into NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, int.class);
        vectorUpCast.upCast(bufferSize, shift, jData, buf);

        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;
        int shift = (int) (vectorIndex * nColumns + featureIndex);

        // Copies data to the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature((int)featureIndex);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, double.class);
        vectorUpCast.upCastWithStride(bufferSize, shift, nColumns, jData, buf);

        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;
        int shift = (int) (vectorIndex * nColumns + featureIndex);

        // Copies data to the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature((int)featureIndex);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, float.class);
        vectorUpCast.upCastWithStride(bufferSize, shift, nColumns, jData, buf);

        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;
        int shift = (int) (vectorIndex * nColumns + featureIndex);

        // Copies data to the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature((int)featureIndex);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, int.class);
        vectorUpCast.upCastWithStride(bufferSize, shift, nColumns, jData, buf);

        return buf;
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);
        int shift = (int) (vectorIndex * nColumns);

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(double.class, df.type);
        vectorDownCast.downCast(bufferSize, shift, buf, jData);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);
        int shift = (int) (vectorIndex * nColumns);

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(float.class, df.type);
        vectorDownCast.downCast(bufferSize, shift, buf, jData);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);
        int shift = (int) (vectorIndex * nColumns);

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(int.class, df.type);
        vectorDownCast.downCast(bufferSize, shift, buf, jData);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum);
        int shift = (int) (vectorIndex * nColumns + featureIndex);

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature((int)featureIndex);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(double.class, df.type);
        vectorDownCast.downCastWithStride(bufferSize, shift, nColumns, buf, jData);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum);
        int shift = (int) (vectorIndex * nColumns + featureIndex);

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature((int)featureIndex);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(float.class, df.type);
        vectorDownCast.downCastWithStride(bufferSize, shift, nColumns, buf, jData);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum);
        int shift = (int) (vectorIndex * nColumns + featureIndex);

        // Copies results from the NIO buffer
        DataDictionary dict = getDictionary();
        DataFeature df = dict.getFeature((int)featureIndex);
        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(int.class, df.type);
        vectorDownCast.downCastWithStride(bufferSize, shift, nColumns, buf, jData);
    }

    /** @copydoc HomogenNumericTable::getDoubleArray() */
    @Override
    public double[] getDoubleArray() {
        int nRows = (int) getNumberOfRows();
        int nColumns = (int) getNumberOfColumns();
        DoubleBuffer doubleBuffer = DoubleBuffer.allocate(nRows * nColumns);
        doubleBuffer = getBlockOfRows(0, nRows, doubleBuffer);

        double[] buffer;
        buffer = new double[doubleBuffer.capacity()];
        doubleBuffer.get(buffer);

        return buffer;
    }

    /** @copydoc HomogenNumericTable::getFloatArray() */
    @Override
    public float[] getFloatArray() {
        int nRows = (int) getNumberOfRows();
        int nColumns = (int) getNumberOfColumns();
        FloatBuffer floatBuffer = FloatBuffer.allocate(nRows * nColumns);
        floatBuffer = getBlockOfRows(0, nRows, floatBuffer);

        float[] buffer;
        buffer = new float[floatBuffer.capacity()];
        floatBuffer.get(buffer);

        return buffer;
    }

    /** @copydoc HomogenNumericTable::getLongArray() */
    @Override
    public long[] getLongArray() {
        int nRows = (int) getNumberOfRows();
        int nColumns = (int) getNumberOfColumns();
        IntBuffer intBuffer = IntBuffer.allocate(nRows * nColumns);
        intBuffer = getBlockOfRows(0, nRows, intBuffer);

        int[] intArray;
        intArray = new int[intBuffer.capacity()];
        intBuffer.get(intArray);

        long[] buffer;
        buffer = new long[intBuffer.capacity()];

        for (int i = 0; i < intBuffer.capacity(); i++) {
            buffer[i] = (long)intArray[i];
        }

        return buffer;
    }

    /** @copydoc HomogenNumericTable::getDataObject() */
    @Override
    public Object getDataObject() {
        return jData;
    }

    /** @copydoc HomogenNumericTable::getNumericType() */
    @Override
    public Class<? extends Number> getNumericType() {
        return type;
    }

    /**
     *  Sets a data dictionary in the Numeric Table
     *  @param ddict Pointer to the data dictionary
     */
    public void setDictionary(DataDictionary ddict) {
        dict = ddict;
        if (cObject != 0) {
            cSetCDataDictionary(cObject, ddict.getCObject());
        }
    }

    /**
     *  Sets the value of the element at position (row, column) to a given double value
     *  @param row      Row of the element
     *  @param column   Column of the element
     *  @param value    New value of the element
     */
    public void set(long row, long column, double value) {
        java.lang.reflect.Array.setDouble(jData, (int)(row * getNumberOfColumns() + column), value);
    }

    /**
     *  Sets the value of the element at position (row, column) to a given float value
     *  @param row      Row of the element
     *  @param column   Column of the element
     *  @param value    New value of the element
     */
    public void set(long row, long column, float value) {
        java.lang.reflect.Array.setFloat(jData, (int)(row * getNumberOfColumns() + column), value);
    }

    /**
     *  Sets the value of the element at position (row, column) to a given long value
     *  @param row      Row of the element
     *  @param column   Column of the element
     *  @param value    New value of the element
     */
    public void set(long row, long column, long value) {
        java.lang.reflect.Array.setLong(jData, (int)(row * getNumberOfColumns() + column), value);
    }

    /**
     *  Sets the value of the element at position (row, column) to a given int value
     *  @param row      Row of the element
     *  @param column   Column of the element
     *  @param value    New value of the element
     */
    public void set(long row, long column, int value) {
        java.lang.reflect.Array.setInt(jData, (int)(row * getNumberOfColumns() + column), value);
    }

    /**
     *  Returns the value of the element at position (row, column) as double
     *  @param row      Row of the element
     *  @param column   Column of the element
     */
    public double getDouble(long row, long column) {
        return java.lang.reflect.Array.getDouble(jData, (int)(row * getNumberOfColumns() + column));
    }

    /**
     *  Returns the value of the element at position (row, column) as double
     *  @param row      Row of the element
     *  @param column   Column of the element
     */
    public float getFloat(long row, long column) {
        return java.lang.reflect.Array.getFloat(jData, (int)(row * getNumberOfColumns() + column));
    }

    /**
     *  Returns the value of the element at position (row, column) as double
     *  @param row      Row of the element
     *  @param column   Column of the element
     */
    public long getLong(long row, long column) {
        return java.lang.reflect.Array.getLong(jData, (int)(row * getNumberOfColumns() + column));
    }

    /**
     *  Returns the value of the element at position (row, column) as double
     *  @param row      Row of the element
     *  @param column   Column of the element
     */
    public int getInt(long row, long column) {
        return java.lang.reflect.Array.getInt(jData, (int)(row * getNumberOfColumns() + column));
    }

    /** @copydoc NumericTable::allocateDataMemory() */
    @Override
    public void allocateDataMemory() {
        throw new IllegalArgumentException("can not allocate data memory in Homogen Numeric Table with data on Java side");
    }

    /** @copydoc NumericTable::freeDataMemory() */
    @Override
    public void freeDataMemory() {
    }

    /** @copydoc NumericTableImpl::getNumberOfColumns() */
    @Override
    public long getNumberOfColumns() {
        return nJavaFeatures;
    }

    /** @copydoc NumericTableImpl::getNumberOfRows() */
    @Override
    public long getNumberOfRows() {
        return nJavaVectors;
    }

    /** @copydoc NumericTableImpl::setNumberOfColumns() */
    @Override
    public void setNumberOfRows(long nRow) {
        throw new IllegalArgumentException("can not change number of rows in Homogen Numeric Table with data on Java side");
    }

    /** @copydoc NumericTableImpl::setNumberOfRows() */
    @Override
    public void setNumberOfColumns(long nCol) {
        throw new IllegalArgumentException("can not change number of columns in Homogen Numeric Table with data on Java side");
    }

    private void initialize(DaalContext context, Class<? extends Number> cls, Object data, long nFeatures, long nVectors) {
        type = cls;
        cObject = newJavaNumericTable(nFeatures, nVectors, NumericTable.StorageLayout.aos);
        dict = new DataDictionary(context, nFeatures, cGetCDataDictionary(cObject));
        for (int i = 0; i < nFeatures; i++) {
            dict.setFeature(cls, i);
        }
        nJavaFeatures = nFeatures;
        nJavaVectors = nVectors;
        jData = data;
        dataAllocatedInJava = true;
    }

    @Override
    protected void onUnpack(DaalContext context) {
        cObject = newJavaNumericTable(nJavaFeatures, nJavaVectors, NumericTable.StorageLayout.aos);
        dict = new DataDictionary(context, nJavaFeatures, cGetCDataDictionary(cObject));
        for (int i = 0; i < nJavaFeatures; i++) {
            dict.setFeature(type, i);
        }
    }
}
