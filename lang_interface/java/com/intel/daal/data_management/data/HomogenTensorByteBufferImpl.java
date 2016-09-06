/* file: HomogenTensorByteBufferImpl.java */
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
 * @brief A derivative class of the HomogenTensorImpl class, that provides implementation
 *        of a homogen tensor with data stored in a native C++ tensor
 */
class HomogenTensorByteBufferImpl extends HomogenTensorImpl {

    private static final long maxBufferSize = 2147483647;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long) */
    public HomogenTensorByteBufferImpl(DaalContext context, long cTensor) {
        super(context);
        cObject = cTensor;
        int indexType = getIndexType(cTensor);
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

    /** @copydoc Tensor::getSubtensor(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf) {
        checkCObject();

        long[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);
        long bufferSize = sizeShift[0] * 8;

        // Gets data from C++ Tensor object
        if (bufferSize > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getDoubleSubtensorBuffer(getCObject(), fixedDims, rangeDimIdx, rangeDimNum, byteBuf);
        return byteBuf.asDoubleBuffer();
    }

    /** @copydoc Tensor::getSubtensor(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf) {
        checkCObject();

        long[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);
        long bufferSize = sizeShift[0] * 4;

        // Gets data from C++ Tensor object
        if (bufferSize > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getFloatSubtensorBuffer(getCObject(), fixedDims, rangeDimIdx, rangeDimNum, byteBuf);
        return byteBuf.asFloatBuffer();
    }

    /** @copydoc Tensor::getSubtensor(long,long,IntBuffer) */
    @Override
    public IntBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf) {
        checkCObject();

        long[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);
        long bufferSize = sizeShift[0] * 4;

        // Gets data from C++ Tensor object
        if (bufferSize > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getIntSubtensorBuffer(getCObject(), fixedDims, rangeDimIdx, rangeDimNum, byteBuf);
        return byteBuf.asIntBuffer();
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,DoubleBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf) {
        checkCObject();

        long[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);
        long bufferSize = sizeShift[0] * 8;

        // Gets data from C++ Tensor object
        if (bufferSize > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        double[] data = new double[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ Tensor object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 8));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(data);
        releaseDoubleSubtensorBuffer(getCObject(), fixedDims, rangeDimIdx, rangeDimNum, byteBuf);
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,FloatBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf) {
        checkCObject();

        long[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);
        long bufferSize = sizeShift[0] * 8;

        // Gets data from C++ Tensor object
        if (bufferSize > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        float[] data = new float[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ Tensor object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(data);
        releaseFloatSubtensorBuffer(getCObject(), fixedDims, rangeDimIdx, rangeDimNum, byteBuf);
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,IntBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf) {
        checkCObject();

        long[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);
        long bufferSize = sizeShift[0] * 8;

        // Gets data from C++ Tensor object
        if (bufferSize > maxBufferSize) {
            throw new IllegalArgumentException("size of the block of rows cannot exceed 2 gigabytes");
        }

        int[] data = new int[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ Tensor object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect((int)(bufferSize * 4));
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(data);
        releaseIntSubtensorBuffer(getCObject(), fixedDims, rangeDimIdx, rangeDimNum, byteBuf);
    }

    /** @copydoc HomogenTensor::getDataObject() */
    @Override
    public Object getDataObject() {
        return null;
    }

    /** @copydoc HomogenTensor::getNumericType() */
    @Override
    public Class<? extends Number> getNumericType() {
        return null;
    }

    protected long[] getOffsets() {
        long[] dims = getDimensions();
        long[] offsets = new long[dims.length];
        for(int i=0;i<dims.length;i++) offsets[i] = 1;
        for(int i=0;i<dims.length;i++) {
            for(int j=0;j<i;j++) {
                offsets[j] *= dims[i];
            }
        }
        return offsets;
    }

    private long[] sizeAndShift(long[] fixedDims, long rangeDimIdx, long rangeDimNum) {
        long[]  dims     = getDimensions();
        long[]  offsets  = getOffsets();
        long    nDim     = dims.length;
        long    size     = 1;

        long shift = 0;
        for( int i=0; i<fixedDims.length; i++ )
        {
            shift += fixedDims[i] * offsets[i];
        }
        if( fixedDims.length != nDim )
        {
            shift += rangeDimIdx * offsets[fixedDims.length];

            size = rangeDimNum;
            for( int i=(int)(rangeDimIdx+1); i<nDim; i++ )
            {
                size *= dims[i];
            }
        }

        return new long[]{size,shift};
    }

    /* Gets index type of the C++ HomogenTensor object */
    private native int getIndexType(long cObject);

    /* Gets NIO buffer containing data of the C++ tensor */
    private native ByteBuffer getDoubleSubtensorBuffer(long cObject, long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buffer);
    private native ByteBuffer getFloatSubtensorBuffer(long cObject, long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buffer);
    private native ByteBuffer getIntSubtensorBuffer(long cObject, long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buffer);

    private native void releaseDoubleSubtensorBuffer(long cObject, long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buffer);
    private native void releaseFloatSubtensorBuffer(long cObject, long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buffer);
    private native void releaseIntSubtensorBuffer(long cObject, long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buffer);
}
