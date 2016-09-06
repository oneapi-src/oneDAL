/* file: PackedTriangularMatrixByteBufferImpl.java */
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

class PackedTriangularMatrixByteBufferImpl extends PackedTriangularMatrixImpl {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /** @copydoc PackedTriangularMatrix::PackedTriangularMatrix(DaalContext,long) */
    public PackedTriangularMatrixByteBufferImpl(DaalContext context, long cTable) {
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

    public PackedTriangularMatrixByteBufferImpl(DaalContext context, Class<? extends Number> cls, long nDim, NumericTable.StorageLayout layout) {
        super(context);
        initPackedTriangularMatrix(context, cls, nDim, NumericTable.AllocationFlag.NotAllocate, layout);
    }

    public PackedTriangularMatrixByteBufferImpl(DaalContext context, Class<? extends Number> cls, long nDim, NumericTable.StorageLayout layout,
            NumericTable.AllocationFlag allocFlag) {
        super(context);
        initPackedTriangularMatrix(context, cls, nDim, allocFlag, layout);
    }

    /** @copydoc PackedTriangularMatrix::assign(long) */
    @Override
    public void assign(long constValue) {
        assignLong(getCObject(), constValue);
    }

    /** @copydoc PackedTriangularMatrix::assign(int) */
    @Override
    public void assign(int constValue) {
        assignInt(getCObject(), constValue);
    }

    /** @copydoc PackedTriangularMatrix::assign(double) */
    @Override
    public void assign(double constValue) {
        assignDouble(getCObject(), constValue);
    }

    /** @copydoc PackedTriangularMatrix::assign(float) */
    @Override
    public void assign(float constValue) {
        assignFloat(getCObject(), constValue);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getDoubleBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
        return byteBuf.asDoubleBuffer();
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getFloatBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
        return byteBuf.asFloatBuffer();
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getIntBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
        return byteBuf.asIntBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getDoubleColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asDoubleBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getFloatColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asFloatBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getIntColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asIntBuffer();
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);

        double[] data = new double[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(data);
        releaseDoubleBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);

        float[] data = new float[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(data);
        releaseFloatBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) (vectorNum * nColumns);

        int[] data = new int[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(data);
        releaseIntBlockBuffer(getCObject(), vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int bufferSize = (int) (vectorNum);

        double[] data = new double[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(data);
        releaseDoubleColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int bufferSize = (int) (vectorNum);

        float[] data = new float[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(data);
        releaseFloatColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int bufferSize = (int) (vectorNum);

        int[] data = new int[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(data);
        releaseIntColumnBuffer(getCObject(), featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc PackedTriangularMatrix::getPackedArray(DoubleBuffer) */
    @Override
    public DoubleBuffer getPackedArray(DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (nDim * (nDim + 1)) / 2;

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getDoublePackedBuffer(getCObject(), bufferSize, byteBuf);
        return byteBuf.asDoubleBuffer();
    }

    /** @copydoc PackedTriangularMatrix::getPackedArray(FloatBuffer) */
    @Override
    public FloatBuffer getPackedArray(FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (nDim * (nDim + 1)) / 2;

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getFloatPackedBuffer(getCObject(), bufferSize, byteBuf);
        return byteBuf.asFloatBuffer();
    }

    /** @copydoc PackedTriangularMatrix::getPackedArray(IntBuffer) */
    @Override
    public IntBuffer getPackedArray(IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (nDim * (nDim + 1)) / 2;

        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getIntPackedBuffer(getCObject(), bufferSize, byteBuf);
        return byteBuf.asIntBuffer();
    }

    /** @copydoc PackedTriangularMatrix::releasePackedArray(DoubleBuffer) */
    @Override
    public void releasePackedArray(DoubleBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (nDim * (nDim + 1)) / 2;

        double[] data = new double[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(data);
        releaseDoublePackedBuffer(getCObject(), bufferSize, byteBuf);
    }

    /** @copydoc PackedTriangularMatrix::releasePackedArray(FloatBuffer) */
    @Override
    public void releasePackedArray(FloatBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (nDim * (nDim + 1)) / 2;

        float[] data = new float[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(data);
        releaseFloatPackedBuffer(getCObject(), bufferSize, byteBuf);
    }

    /** @copydoc PackedTriangularMatrix::releasePackedArray(IntBuffer) */
    @Override
    public void releasePackedArray(IntBuffer buf) {
        int nDim = (int) getNumberOfColumns();
        int bufferSize = (nDim * (nDim + 1)) / 2;

        int[] data = new int[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(data);
        releaseIntPackedBuffer(getCObject(), bufferSize, byteBuf);
    }

    /** @copydoc PackedTriangularMatrix::getDataObject */
    @Override
    public Object getDataObject() {
        return null;
    }

    /** @copydoc PackedTriangularMatrix::getNumericType */
    @Override
    public Class<? extends Number> getNumericType() {
        return null;
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

    private void initPackedTriangularMatrix(DaalContext context, Class<? extends Number> cls, long nDim,
            NumericTable.AllocationFlag allocFlag, NumericTable.StorageLayout layout) {
        if (cls == Double.class) {
            cObject = dInit(nDim, layout.ordinal());
        } else if (cls == Float.class) {
            cObject = sInit(nDim, layout.ordinal());
        } else if (cls == Long.class) {
            cObject = lInit(nDim, layout.ordinal());
        } else if (cls == Integer.class) {
            cObject = iInit(nDim, layout.ordinal());
        } else {
            throw new IllegalArgumentException("type unsupported");
        }
        dict = new DataDictionary(context, nDim, cGetCDataDictionary(cObject));
        for (int i = 0; i < nDim; i++) {
            dict.setFeature(cls, i);
        }
        type = cls;
        dataAllocatedInJava = false;
        if (nDim > 0) {
            setNumberOfRows(nDim);
        }
        if (allocFlag.ordinal() == NumericTable.AllocationFlag.DoAllocate.ordinal()) {
            allocateDataMemory();
        }
    }

    /* Creates C++ PackedTriangularMatrix object */
    private native long dInit(long nDim, int layout);
    private native long sInit(long nDim, int layout);
    private native long lInit(long nDim, int layout);
    private native long iInit(long nDim, int layout);

    private native void cAllocateDataMemoryDouble(long cObject);
    private native void cAllocateDataMemoryFloat(long cObject);
    private native void cAllocateDataMemoryLong(long cObject);
    private native void cAllocateDataMemoryInt(long cObject);

    /* Gets index type of the C++ PackedTriangularMatrix object */
    private native int getIndexType(long cObject);

    /* Gets NIO buffer containing data of the C++ table */

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

    private native ByteBuffer getDoublePackedBuffer(long cObject, long bufferSize, ByteBuffer buffer);
    private native ByteBuffer getFloatPackedBuffer(long cObject, long bufferSize, ByteBuffer buffer);
    private native ByteBuffer getIntPackedBuffer(long cObject, long bufferSize, ByteBuffer buffer);

    private native void releaseDoublePackedBuffer(long cObject, long bufferSize, ByteBuffer buffer);
    private native void releaseFloatPackedBuffer(long cObject, long bufferSize, ByteBuffer buffer);
    private native void releaseIntPackedBuffer(long cObject, long bufferSize, ByteBuffer buffer);
}
