/* file: HomogenTensorArrayImpl.java */
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
 * @brief A derivative class of the HomogenTensorImpl class, that provides implementation
 *        of a homogen tensor with data stored as array of primitives
 */
class HomogenTensorArrayImpl extends HomogenTensorImpl {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
        cObject = newJavaTensor(dims);
        jData = data;
        dataAllocatedInJava = true;
    }

    @Override
    protected void onUnpack(DaalContext context) {
        cObject = newJavaTensor(serializedDims);
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
