/* file: HomogenTensorArrayImpl.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @ingroup tensor
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.lang.reflect.Array;

import com.intel.daal.services.DaalContext;
import com.intel.daal.SerializationTag;

/**
 * @brief A derivative class of the HomogenTensorImpl class, that provides implementation
 *        of a homogen tensor with data stored as array of primitives
 */
class HomogenTensorArrayImpl extends HomogenTensorImpl {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long[],double[]) */
    public HomogenTensorArrayImpl(DaalContext context, long[] dims,  double[] data) {
        super(context);
        initialize(context, Double.class, dims, data);
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long[],float[]) */
    public HomogenTensorArrayImpl(DaalContext context, long[] dims,  float[] data) {
        super(context);
        initialize(context, Float.class, dims, data);
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long[],int[]) */
    public HomogenTensorArrayImpl(DaalContext context, long[] dims,  int[] data) {
        super(context);
        initialize(context, Integer.class, dims, data);
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long[],long[]) */
    public HomogenTensorArrayImpl(DaalContext context, long[] dims,  long[] data) {
        super(context);
        initialize(context, Long.class, dims, data);
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long[],double[],double) */
    public HomogenTensorArrayImpl(DaalContext context, long[] dims,  double[] data, double constValue) {
        super(context);
        initialize(context, Double.class, dims, data);
        assign(constValue);
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long[],float[],float) */
    public HomogenTensorArrayImpl(DaalContext context, long[] dims,  float[] data, float constValue) {
        super(context);
        initialize(context, Float.class, dims, data);
        assign(constValue);
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long[],int[],int) */
    public HomogenTensorArrayImpl(DaalContext context, long[] dims,  int[] data, int constValue) {
        super(context);
        initialize(context, Integer.class, dims, data);
        assign(constValue);
    }

    /** @copydoc HomogenTensor::HomogenTensor(DaalContext,long[],long[],long) */
    public HomogenTensorArrayImpl(DaalContext context, long[] dims,  long[] data, long constValue) {
        super(context);
        initialize(context, Long.class, dims, data);
        assign(constValue);
    }

    /** @copydoc Tensor::getSubtensor(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf) {
        int[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);

        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(type, double.class);
        vectorUpCast.upCast(sizeShift[0], sizeShift[1], jData, buf);

        return buf;
    }

    /** @copydoc Tensor::getSubtensor(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf) {
        int[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);

        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(type, float.class);
        vectorUpCast.upCast(sizeShift[0], sizeShift[1], jData, buf);

        return buf;
    }

    /** @copydoc Tensor::getSubtensor(long,long,IntBuffer) */
    @Override
    public IntBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf) {
        int[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);

        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(type, int.class);
        vectorUpCast.upCast(sizeShift[0], sizeShift[1], jData, buf);

        return buf;
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,DoubleBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf) {
        int[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);

        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(double.class, type);
        vectorDownCast.downCast(sizeShift[0], sizeShift[1], buf, jData);
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,FloatBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf) {
        int[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);

        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(float.class, type);
        vectorDownCast.downCast(sizeShift[0], sizeShift[1], buf, jData);
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,IntBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf) {
        int[] sizeShift = sizeAndShift(fixedDims, rangeDimIdx, rangeDimNum);

        DataFeatureUtils.VectorDownCastIface vectorDownCast = DataFeatureUtils.VectorDownCast.getCast(int.class, type);
        vectorDownCast.downCast(sizeShift[0], sizeShift[1], buf, jData);
    }

    /** @copydoc HomogenTensor::getDataObject() */
    @Override
    public Object getDataObject() {
        return jData;
    }

    /** @copydoc HomogenTensor::getNumericType() */
    @Override
    public Class<? extends Number> getNumericType() {
        return type;
    }

    /** @copydoc Tensor::allocateDataMemory() */
    @Override
    public void allocateDataMemory() {
        throw new IllegalArgumentException("can not allocate data memory in Homogen Tensor with data on Java side");
    }

    /** @copydoc Tensor::freeDataMemory() */
    @Override
    public void freeDataMemory() {
    }

    private void initialize(DaalContext context, Class<? extends Number> cls, long[] dims, Object data) {
        type = cls;
        cObject = newJavaTensor(dims, SerializationTag.SERIALIZATION_JAVANIO_HOMOGEN_TENSOR_ID);
        jData = data;
        dataAllocatedInJava = true;
    }

    @Override
    protected void onUnpack(DaalContext context) {
        if (this.cObject == 0) {
            cObject = newJavaTensor(serializedDims, SerializationTag.SERIALIZATION_JAVANIO_HOMOGEN_TENSOR_ID);
        }
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

    private int[] sizeAndShift(long[] fixedDims, long rangeDimIdx, long rangeDimNum) {
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
            for( int i=fixedDims.length+1; i<nDim; i++ )
            {
                size *= dims[i];
            }
        }

        return new int[]{(int)size,(int)shift};
    }

    private void assign(long constValue) {
        long[] dims = getDimensions();
        int nSize = 1;
        for (int i = 0; i < dims.length; i++) {
            nSize *= dims[i];
        }
        if (type != Long.class) {
            for(int i = 0; i < nSize; i++) {
                java.lang.reflect.Array.setLong(jData, i, constValue);
            }
        }
        else {
            long[] data = (long[])jData;
            for(int i = 0; i < nSize; i++) {
                data[i] = constValue;
            }
        }
    }

    private void assign(int constValue) {
        long[] dims = getDimensions();
        int nSize = 1;
        for (int i = 0; i < dims.length; i++) {
            nSize *= dims[i];
        }
        if (type != Integer.class) {
            for(int i = 0; i < nSize; i++) {
                java.lang.reflect.Array.setInt(jData, i, constValue);
            }
        }
        else {
            int[] data = (int[])jData;
            for(int i = 0; i < nSize; i++) {
                data[i] = constValue;
            }
        }
    }

    private void assign(double constValue) {
        long[] dims = getDimensions();
        int nSize = 1;
        for (int i = 0; i < dims.length; i++) {
            nSize *= dims[i];
        }
        if (type != Double.class) {
            for(int i = 0; i < nSize; i++) {
                java.lang.reflect.Array.setDouble(jData, i, constValue);
            }
        }
        else {
            double[] data = (double[])jData;
            for(int i = 0; i < nSize; i++) {
                data[i] = constValue;
            }
        }
    }

    private void assign(float constValue) {
        long[] dims = getDimensions();
        int nSize = 1;
        for (int i = 0; i < dims.length; i++) {
            nSize *= dims[i];
        }
        if (type != Float.class) {
            for(int i = 0; i < nSize; i++) {
                java.lang.reflect.Array.setFloat(jData, i, constValue);
            }
        }
        else {
            float[] data = (float[])jData;
            for(int i = 0; i < nSize; i++) {
                data[i] = constValue;
            }
        }
    }
}
/** @} */
