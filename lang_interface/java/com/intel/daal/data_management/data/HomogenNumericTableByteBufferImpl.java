/* file: HomogenNumericTableByteBufferImpl.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/**
 * <a name="DAAL-CLASS-DATA__HOMOGENNUMERICTABLEBYTEBUFFERIMPL__HOMOGENNUMERICTABLEBYTEBUFFERIMPL"></a>
 * @brief A derivative class of the HomogenNumericTableImpl class, that provides implementation
 *        of a homogen numeric table with data stored in a native C++ numeric table
 */
class HomogenNumericTableByteBufferImpl extends HomogenNumericTableImpl {

    private static final long maxBufferSize = 2147483647;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,long) */
    public HomogenNumericTableByteBufferImpl(DaalContext context, long cTable) {
        super(context);
        cObject = cTable;
        int indexType = getIndexType(cTable);
        dict = new DataDictionary(context, (long)0, cGetCDataDictionary(cTable));
        dataAllocatedInJava = false;
        if (indexType == DataFeatureUtils.IndexNumType.DAAL_FLOAT32.getType()) {
            type = Float.class;
        } else if (indexType == DataFeatureUtils.IndexNumType.DAAL_FLOAT64.getType()) {
            type = Double.class;
        } else if (indexType == DataFeatureUtils.IndexNumType.DAAL_INT64_S.getType() ||
                indexType == DataFeatureUtils.IndexNumType.DAAL_INT64_U.getType()) {
            type = Long.class;
        } else if (indexType == DataFeatureUtils.IndexNumType.DAAL_INT32_S.getType() ||
                indexType == DataFeatureUtils.IndexNumType.DAAL_INT32_U.getType()) {
            type = Integer.class;
        } else {
            throw new IllegalArgumentException("type unsupported");
        }
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,Class<? extends Number>,long,DataDictionary.FeaturesEqual) */
    public HomogenNumericTableByteBufferImpl(DaalContext context, Class<? extends Number> cls, long nColumns, DataDictionary.FeaturesEqual featuresEqual) {
        super(context);
        initHomogenNumericTable(context, cls, nColumns, 0, NumericTable.AllocationFlag.DoNotAllocate, featuresEqual);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,Class<? extends Number>,long,long,NumericTable.AllocationFlag,DataDictionary.FeaturesEqual) */
    public HomogenNumericTableByteBufferImpl(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            NumericTable.AllocationFlag allocFlag, DataDictionary.FeaturesEqual featuresEqual) {
        super(context);
        initHomogenNumericTable(context, cls, nColumns, nRows, allocFlag, featuresEqual);
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,Class<? extends Number>,long,long,NumericTable.AllocationFlag,double,DataDictionary.FeaturesEqual) */
    public HomogenNumericTableByteBufferImpl(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            NumericTable.AllocationFlag allocFlag, double constValue, DataDictionary.FeaturesEqual featuresEqual) {
        super(context);
        initHomogenNumericTable(context, cls, nColumns, nRows, allocFlag, featuresEqual);
        if (allocFlag.ordinal() == NumericTable.AllocationFlag.DoAllocate.ordinal()) {
            assign(constValue);
        }
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,Class<? extends Number>,long,long,NumericTable.AllocationFlag,float,DataDictionary.FeaturesEqual) */
    public HomogenNumericTableByteBufferImpl(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            NumericTable.AllocationFlag allocFlag, float constValue, DataDictionary.FeaturesEqual featuresEqual) {
        super(context);
        initHomogenNumericTable(context, cls, nColumns, nRows, allocFlag, featuresEqual);
        if (allocFlag.ordinal() == NumericTable.AllocationFlag.DoAllocate.ordinal()) {
            assign(constValue);
        }
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,Class<? extends Number>,long,long,NumericTable.AllocationFlag,long,DataDictionary.FeaturesEqual) */
    public HomogenNumericTableByteBufferImpl(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            NumericTable.AllocationFlag allocFlag, long constValue, DataDictionary.FeaturesEqual featuresEqual) {
        super(context);
        initHomogenNumericTable(context, cls, nColumns, nRows, allocFlag, featuresEqual);
        if (allocFlag.ordinal() == NumericTable.AllocationFlag.DoAllocate.ordinal()) {
            assign(constValue);
        }
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,Class<? extends Number>,long,long,NumericTable.AllocationFlag,int,DataDictionary.FeaturesEqual) */
    public HomogenNumericTableByteBufferImpl(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            NumericTable.AllocationFlag allocFlag, int constValue, DataDictionary.FeaturesEqual featuresEqual) {
        super(context);
        initHomogenNumericTable(context, cls, nColumns, nRows, allocFlag, featuresEqual);
        if (allocFlag.ordinal() == NumericTable.AllocationFlag.DoAllocate.ordinal()) {
            assign(constValue);
        }
    }

    /** @copydoc HomogenNumericTable::HomogenNumericTable(DaalContext,Class<? extends Number>,DataDictionary) */
    public HomogenNumericTableByteBufferImpl(DaalContext context, Class<? extends Number> cls, DataDictionary dict) {
        super(context);
        initHomogenNumericTable(context, cls, dict);
    }

    /** @copydoc HomogenNumericTable::assign(long) */
    @Override
    public void assign(long constValue) {
        checkCObject();
        assignLong(getCObject(), constValue);
    }

    /** @copydoc HomogenNumericTable::assign(int) */
    @Override
    public void assign(int constValue) {
        checkCObject();
        assignInt(getCObject(), constValue);
    }

    /** @copydoc HomogenNumericTable::assign(double) */
    @Override
    public void assign(double constValue) {
        checkCObject();
        assignDouble(getCObject(), constValue);
    }

    /** @copydoc HomogenNumericTable::assign(float) */
    @Override
    public void assign(float constValue) {
        checkCObject();
        assignFloat(getCObject(), constValue);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        checkCObject();

        long nColumns = getNumberOfColumns();
        long bufferSize = vectorNum * nColumns;

        // Gets data from C++ NumericTable object
        if (bufferSize * 8 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8) /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getDoubleBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
        return byteBuf.asDoubleBuffer();
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        checkCObject();

        long nColumns = getNumberOfColumns();
        long bufferSize = vectorNum * nColumns;

        // Gets data from C++ NumericTable object
        if (bufferSize * 4 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4) /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getFloatBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
        return byteBuf.asFloatBuffer();
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        checkCObject();

        long nColumns = getNumberOfColumns();
        long bufferSize = vectorNum * nColumns;

        // Gets data from C++ NumericTable object
        if (bufferSize * 4> maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4) /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getIntBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
        return byteBuf.asIntBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        checkCObject();

        long bufferSize = vectorNum;

        // Gets data from C++ NumericTable object
        if (bufferSize * 8 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of column values cannot exceed 2 gigabytes");
        }
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8) /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getDoubleColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asDoubleBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        checkCObject();

        long bufferSize = vectorNum;

        // Gets data from C++ NumericTable object
        if (bufferSize * 4 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of column values cannot exceed 2 gigabytes");
        }
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4) /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getFloatColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asFloatBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        checkCObject();

        long bufferSize = vectorNum;

        // Gets data from C++ NumericTable object
        if (bufferSize * 4 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of column values cannot exceed 2 gigabytes");
        }
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4) /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getIntColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asIntBuffer();
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        checkCObject();

        long nColumns = getNumberOfColumns();
        long bufferSize = vectorNum * nColumns;

        if (bufferSize * 8 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        double[] data = new double[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8) /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(data);
        releaseDoubleBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        checkCObject();

        long nColumns = getNumberOfColumns();
        long bufferSize = vectorNum * nColumns;

        if (bufferSize * 4 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        float[] data = new float[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4) /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(data);
        releaseFloatBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        checkCObject();

        long nColumns = getNumberOfColumns();
        long bufferSize = vectorNum * nColumns;

        if (bufferSize * 4 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        int[] data = new int[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4) /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(data);
        releaseIntBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        checkCObject();

        long bufferSize = vectorNum;

        if (bufferSize * 8 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of column values cannot exceed 2 gigabytes");
        }

        double[] data = new double[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8) /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(data);
        releaseDoubleColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        checkCObject();

        long bufferSize = vectorNum;

        if (bufferSize * 4 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of column values cannot exceed 2 gigabytes");
        }

        float[] data = new float[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4) /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(data);
        releaseFloatColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        checkCObject();

        long bufferSize = vectorNum;

        if (bufferSize * 4 > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of column values cannot exceed 2 gigabytes");
        }

        int[] data = new int[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4) /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(data);
        releaseIntColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc HomogenNumericTable::getDoubleArray() */
    @Override
    public double[] getDoubleArray() {
        checkCObject();

        ByteBuffer byteBuffer = getDoubleBuffer(getCObject());
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer doubleBuffer = byteBuffer.asDoubleBuffer();

        double[] buffer;
        buffer = new double[doubleBuffer.capacity()];
        doubleBuffer.get(buffer);

        return buffer;
    }

    /** @copydoc HomogenNumericTable::getFloatArray() */
    @Override
    public float[] getFloatArray() {
        checkCObject();

        ByteBuffer byteBuffer = getFloatBuffer(getCObject());
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

        float[] buffer;
        buffer = new float[floatBuffer.capacity()];
        floatBuffer.get(buffer);

        return buffer;
    }

    /** @copydoc HomogenNumericTable::getLongArray() */
    @Override
    public long[] getLongArray() {
        checkCObject();

        ByteBuffer byteBuffer = getLongBuffer(getCObject());
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        LongBuffer longBuffer = byteBuffer.asLongBuffer();

        long[] buffer;
        buffer = new long[longBuffer.capacity()];
        longBuffer.get(buffer);

        return buffer;
    }

    /** @copydoc HomogenNumericTable::getDataObject() */
    @Override
    public Object getDataObject() {
        return null;
    }

    /** @copydoc HomogenNumericTable::getNumericType() */
    @Override
    public Class<? extends Number> getNumericType() {
        return null;
    }

    /** @copydoc NumericTableImpl::getNumberOfColumns() */
    @Override
    public long getNumberOfColumns() {
        if (cObject != 0) {
            return cGetNumberOfColumns();
        } else
        if (serializedCObject != null) {
            return nSerializedFeatures;
        } else {
            throw new IllegalArgumentException("number of columns is undefined");
        }
    }

    /** @copydoc NumericTableImpl::getNumberOfRows() */
    @Override
    public long getNumberOfRows() {
        if (cObject != 0) {
            return cGetNumberOfRows();
        } else
        if (serializedCObject != null) {
            return nSerializedVectors;
        } else {
            throw new IllegalArgumentException("number of rows is undefined");
        }
    }

    /**
     *  Sets the value of the element at position (row, column) to a given double value
     *  @param row      Row of the element
     *  @param column   Column of the element
     *  @param value    New value of the element
     */
    public void set(long row, long column, double value) {
        checkCObject();
        cSetDouble(row, column, value);
    }

    /**
     *  Sets the value of the element at position (row, column) to a given float value
     *  @param row      Row of the element
     *  @param column   Column of the element
     *  @param value    New value of the element
     */
    public void set(long row, long column, float value) {
        checkCObject();
        cSetFloat(row, column, value);
    }

    /**
     *  Sets the value of the element at position (row, column) to a given long value
     *  @param row      Row of the element
     *  @param column   Column of the element
     *  @param value    New value of the element
     */
    public void set(long row, long column, long value) {
        checkCObject();
        cSetLong(row, column, value);
    }

    /**
     *  Sets the value of the element at position (row, column) to a given int value
     *  @param row      Row of the element
     *  @param column   Column of the element
     *  @param value    New value of the element
     */
    public void set(long row, long column, int value) {
        checkCObject();
        cSetInt(row, column, value);
    }

    /**
     *  Returns the value of the element at position (row, column) as double
     *  @param row      Row of the element
     *  @param column   Column of the element
     */
    public double getDouble(long row, long column) {
        checkCObject();
        return cGetDouble(row, column);
    }

    /**
     *  Returns the value of the element at position (row, column) as float
     *  @param row      Row of the element
     *  @param column   Column of the element
     */
    public float getFloat(long row, long column) {
        checkCObject();
        return cGetFloat(row, column);
    }

    /**
     *  Returns the value of the element at position (row, column) as long
     *  @param row      Row of the element
     *  @param column   Column of the element
     */
    public long getLong(long row, long column) {
        checkCObject();
        return cGetLong(row, column);
    }

    /**
     *  Returns the value of the element at position (row, column) as int
     *  @param row      Row of the element
     *  @param column   Column of the element
     */
    public int getInt(long row, long column) {
        checkCObject();
        return cGetInt(row, column);
    }

    /** @copydoc NumericTable::allocateDataMemory() */
    @Override
    public void allocateDataMemory() {
        checkCObject();
        if (type == Double.class) {
            cAllocateDataMemoryDouble(getCObject());
        } else if (type == Float.class) {
            cAllocateDataMemoryFloat(getCObject());
        } else if (type == Long.class) {
            cAllocateDataMemoryLong(getCObject());
        } else if (type == Integer.class) {
            cAllocateDataMemoryInt(getCObject());
        } else {
            throw new IllegalArgumentException("type unsupported");
        }
    }

    /** @copydoc NumericTable::freeDataMemory() */
    @Override
    public void freeDataMemory() {
        checkCObject();
        cFreeDataMemory();
    }

    private void initHomogenNumericTable(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
            NumericTable.AllocationFlag allocFlag, DataDictionary.FeaturesEqual featuresEqual) {
        if (cls == Double.class) {
            cObject = dInit(nColumns, featuresEqual.ordinal());
        } else if (cls == Float.class) {
            cObject = sInit(nColumns, featuresEqual.ordinal());
        } else if (cls == Long.class) {
            cObject = lInit(nColumns, featuresEqual.ordinal());
        } else if (cls == Integer.class) {
            cObject = iInit(nColumns, featuresEqual.ordinal());
        } else {
            throw new IllegalArgumentException("type unsupported");
        }
        dict = new DataDictionary(context, nColumns, cGetCDataDictionary(cObject));
        if (dict.getFeaturesEqual().ordinal() == DataDictionary.FeaturesEqual.equal.ordinal()) {
            dict.setFeature(cls, 0);
        } else {
            for (int i = 0; i < nColumns; i++) {
                dict.setFeature(cls, i);
            }
        }
        type = cls;
        dataAllocatedInJava = false;
        if (nRows > 0) {
            setNumberOfRows(nRows);
        }
        if (allocFlag.ordinal() == NumericTable.AllocationFlag.DoAllocate.ordinal()) {
            allocateDataMemory();
        }
    }

    private void initHomogenNumericTable(DaalContext context, Class<? extends Number> cls, DataDictionary dict) {
        this.dict = dict;
        cObject = dictInit(dict.getCObject());
        type = cls;
        dataAllocatedInJava = false;
    }

    /* Creates C++ HomogenNumericTable object */
    private native long dInit(long nColumns, int featuresEqual);
    private native long sInit(long nColumns, int featuresEqual);
    private native long lInit(long nColumns, int featuresEqual);
    private native long iInit(long nColumns, int featuresEqual);
    private native long dictInit(long cObject);

    private native void cAllocateDataMemoryDouble(long cObject);
    private native void cAllocateDataMemoryFloat(long cObject);
    private native void cAllocateDataMemoryLong(long cObject);
    private native void cAllocateDataMemoryInt(long cObject);

    /* Gets index type of the C++ HomogenNumericTable object */
    private native int getIndexType(long cObject);

    /* Gets NIO buffer containing data of the C++ table */
    private native ByteBuffer getDoubleBuffer(long cObject);
    private native ByteBuffer getFloatBuffer(long cObject);
    private native ByteBuffer getLongBuffer(long cObject);
    private native ByteBuffer getIntBuffer(long cObject);

    private native ByteBuffer getDoubleBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native ByteBuffer getFloatBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native ByteBuffer getIntBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native void releaseDoubleBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native void releaseFloatBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native void releaseIntBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native void assignLong(long cObject, long constValue);
    private native void assignInt(long cObject, int constValue);
    private native void assignDouble(long cObject, double constValue);
    private native void assignFloat(long cObject, float constValue);

    private native ByteBuffer getDoubleColumnBuffer(long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native ByteBuffer getFloatColumnBuffer (long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native ByteBuffer getIntColumnBuffer   (long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native void releaseDoubleColumnBuffer(long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native void releaseFloatColumnBuffer (long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native void releaseIntColumnBuffer   (long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native void cSetDouble(long row, long column, double value);
    private native void cSetFloat(long row, long column, float value);
    private native void cSetLong(long row, long column, long value);
    private native void cSetInt(long row, long column, int value);

    private native double cGetDouble(long row, long column);
    private native float cGetFloat(long row, long column);
    private native long cGetLong(long row, long column);
    private native int cGetInt(long row, long column);
}
/** @} */
