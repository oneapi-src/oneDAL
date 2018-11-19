/* file: TensorImpl.java */
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
 * @defgroup tensor Numeric Tensors
 * @brief Contains classes for a data management component responsible for representation of data in the n-dimensions numeric format.
 * @ingroup data_management
 * @{
 */
/**
 * @brief Contains classes that implement the data management component
 *        responsible for representaion of the tensor data
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.intel.daal.services.DaalContext;
import com.intel.daal.SerializationTag;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__TENSORIMPL"></a>
 *  @brief  Class for the data management component responsible for the representation of the data in a numerical format.
 */
abstract public class TensorImpl extends SerializableBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
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

    protected long newJavaTensor(long[] newDims, SerializationTag tag) {
        return cNewJavaTensor(newDims, tag.getValue());
    }

    private native long cNewJavaTensor(long[] newDims, int tag);

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
/** @} */
