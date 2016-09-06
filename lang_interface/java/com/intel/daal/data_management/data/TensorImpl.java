/* file: TensorImpl.java */
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

/**
 * \brief Contains classes that implement the data management component
 *        responsible for representaion of the tensor data
 */
package com.intel.daal.data_management.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__TENSORIMPL"></a>
 *  @brief  Class for the data management component responsible for the representation of the data in a numerical format.
 */
abstract public class TensorImpl extends SerializableBase {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    protected TensorImpl(DaalContext context) {
        super(context);
        serializedDims = null;
        jData = null;
        dataAllocatedInJava = false;
    }

    /** @copydoc Tensor::allocateDataMemory() */
    public void allocateDataMemory() {
        checkCObject();
        cAllocateDataMemory( getCObject() );
    }

    protected native void cAllocateDataMemory(long cObject);

    /** @copydoc Tensor::freeDataMemory() */
    public void freeDataMemory() {
        checkCObject();
        cFreeDataMemory( getCObject() );
    }

    protected native void cFreeDataMemory(long cObject);

    @Override
    protected void onPack() {
        serializedDims = getDimensions();
    }

    @Override
    protected void onUnpack(DaalContext context) {
        deserializeCObject();
    }

    DoubleBuffer getDoubleSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer dBuf = getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf.asDoubleBuffer());
        return dBuf;
    }

    FloatBuffer getFloatSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer sBuf = getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf.asFloatBuffer());
        return sBuf;
    }

    IntBuffer getIntSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer iBuf = getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf.asIntBuffer());
        return iBuf;
    }

    void releaseDoubleSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf.asDoubleBuffer());
    }

    void releaseFloatSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf.asFloatBuffer());
    }

    void releaseIntSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf.asIntBuffer());
    }

    /* True if data for the tensor allocated on Java side */
    protected boolean dataAllocatedInJava;

    /* Java data associated with this tensor */
    protected Object jData;

    protected long[] serializedDims;

    public long[] getDimensions() {
        checkCObject();
        return cGetDimensions( getCObject() );
    }

    protected native long[] cGetDimensions(long cObject);

    public void setDimensions(long[] newDims) {
        checkCObject();
        cSetDimensions( getCObject(), newDims );
    }

    protected native void cSetDimensions(long cObject, long[] newDims);

    protected long newJavaTensor(long[] newDims) {
        return cNewJavaTensor(newDims);
    }

    private native long cNewJavaTensor(long[] newDims);

    @Override
    protected boolean onSerializeCObject() {
        return !dataAllocatedInJava;
    }

    public long getSize() {
        long[] dims = getDimensions();
        if(dims.length == 0) return 0;
        long size=1;
        for(int i=0; i<dims.length; i++) {
            size *= dims[i];
        }
        return size;
    }

    abstract public DoubleBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf);

    abstract public FloatBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf);

    abstract public IntBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf);

    abstract public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf);

    abstract public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf);

    abstract public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf);
}
