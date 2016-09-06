/* file: HomogenTensor.java */
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
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__HOMOGENTENSOR"></a>
 * @brief A derivative class of the Tensor class, that provides methods to
 *        access the data that is stored as a contiguous homogeneous array.
 */
public class HomogenTensor extends Tensor {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs homogeneous tensor using implementation provided by user
     *
     * @param context   Context to manage created homogeneous tensor
     * @param impl      Implementation of homogeneous tensor
     */
    public HomogenTensor(DaalContext context, HomogenTensorImpl impl) {
        super(context);
        tensorImpl = impl;
    }

    /**
     * Constructs homogeneous tensor from the array of doubles
     *
     * @param context   Context to manage created homogeneous tensor
     * @param dims      Dimensions of the tensor
     * @param data      Array with the tensor data
     */
    public HomogenTensor(DaalContext context, long[] dims, double[] data) {
        super(context);
        tensorImpl = new HomogenTensorArrayImpl(context, dims, data);
    }

    /**
     * Constructs homogeneous tensor from the array of floats
     *
     * @param context   Context to manage created homogeneous tensor
     * @param dims      Dimensions of the tensor
     * @param data      Array with the tensor data
     */
    public HomogenTensor(DaalContext context, long[] dims, float[] data) {
        super(context);
        tensorImpl = new HomogenTensorArrayImpl(context, dims, data);
    }

    /**
     * Constructs homogeneous tensor from the array of integers
     *
     * @param context   Context to manage created homogeneous tensor
     * @param dims      Dimensions of the tensor
     * @param data      Array with the tensor data
     */
    public HomogenTensor(DaalContext context, long[] dims, int[] data) {
        super(context);
        tensorImpl = new HomogenTensorArrayImpl(context, dims, data);
    }

    /**
     * Constructs homogeneous tensor from the array of longs
     *
     * @param context   Context to manage created homogeneous tensor
     * @param dims      Dimensions of the tensor
     * @param data      Array with the tensor data
     */
    public HomogenTensor(DaalContext context, long[] dims, long[] data) {
        super(context);
        tensorImpl = new HomogenTensorArrayImpl(context, dims, data);
    }

    /**
     * Constructs homogeneous tensor from the array of doubles and filling the tensor with a constant
     *
     * @param context    Context to manage created homogeneous tensor
     * @param dims       Dimensions of the tensor
     * @param data       Array with the tensor data
     * @param constValue Constant to initialize entries of the homogeneous tensor
     */
    public HomogenTensor(DaalContext context, long[] dims, double[] data, double constValue) {
        super(context);
        tensorImpl = new HomogenTensorArrayImpl(context, dims, data, constValue);
    }

    /**
     * Constructs homogeneous tensor from the array of floats and filling the tensor with a constant
     *
     * @param context    Context to manage created homogeneous tensor
     * @param dims       Dimensions of the tensor
     * @param data       Array with the tensor data
     * @param constValue Constant to initialize entries of the homogeneous tensor
     */
    public HomogenTensor(DaalContext context, long[] dims, float[] data, float constValue) {
        super(context);
        tensorImpl = new HomogenTensorArrayImpl(context, dims, data, constValue);
    }

    /**
     * Constructs homogeneous tensor from the array of integers and filling the tensor with a constant
     *
     * @param context    Context to manage created homogeneous tensor
     * @param dims       Dimensions of the tensor
     * @param data       Array with the tensor data
     * @param constValue Constant to initialize entries of the homogeneous tensor
     */
    public HomogenTensor(DaalContext context, long[] dims, int[] data, int constValue) {
        super(context);
        tensorImpl = new HomogenTensorArrayImpl(context, dims, data, constValue);
    }

    /**
     * Constructs homogeneous tensor from the array of longs and filling the tensor with a constant
     *
     * @param context    Context to manage created homogeneous tensor
     * @param dims       Dimensions of the tensor
     * @param data       Array with the tensor data
     * @param constValue Constant to initialize entries of the homogeneous tensor
     */
    public HomogenTensor(DaalContext context, long[] dims, long[] data, long constValue) {
        super(context);
        tensorImpl = new HomogenTensorArrayImpl(context, dims, data, constValue);
    }

    /**
     * Constructs homogeneous tensor from C++ homogeneous tensor
     * @param context   Context to manage created homogeneous tensor
     * @param cObject   Pointer to C++ tensor
     */
    public HomogenTensor(DaalContext context, long cObject) {
        super(context);
        tensorImpl = new HomogenTensorByteBufferImpl(context, cObject);
    }

    /** @copydoc Tensor::getSubtensor(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf) {
        return ((HomogenTensorImpl)tensorImpl).getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /** @copydoc Tensor::getSubtensor(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf) {
        return ((HomogenTensorImpl)tensorImpl).getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /** @copydoc Tensor::getSubtensor(long,long,IntBuffer) */
    @Override
    public IntBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf) {
        return ((HomogenTensorImpl)tensorImpl).getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,FloatBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf) {
        ((HomogenTensorImpl)tensorImpl).releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,DoubleBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf) {
        ((HomogenTensorImpl)tensorImpl).releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /** @copydoc Tensor::releaseSubtensor(long,long,IntBuffer) */
    @Override
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf) {
        ((HomogenTensorImpl)tensorImpl).releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /**
     * Gets data as an Object
     * @return Tensor data as an Object
     */
    public Object getDataObject() {
        return ((HomogenTensorImpl)tensorImpl).getDataObject();
    }

    /**
     * Gets numeric type of data stored in the tensor
     * @return Numeric type of the tensor data
     */
    public Class<? extends Number> getNumericType() {
        return ((HomogenTensorImpl)tensorImpl).getNumericType();
    }
}
