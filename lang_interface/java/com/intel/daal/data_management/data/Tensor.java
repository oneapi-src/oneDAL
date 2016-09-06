/* file: Tensor.java */
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
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__TENSOR"></a>
 *  @anchor Tensor
 *  @brief  Class for the data management component responsible for the representation of the tensor data.
 */
abstract public class Tensor extends SerializableBase implements TensorDenseIface {
    protected TensorImpl tensorImpl;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    protected Tensor(DaalContext context) {
        super(context);
    }

    /** Specifies whether the Tensor allocates memory */
    static public class DataLayout {
        private static final int _defaultLayout = 0;

        public static final DataLayout defaultLayout = new DataLayout(_defaultLayout);

        private final int _value;

        DataLayout(final int value) {
            this._value = value;
        }

        public int ordinal() {
            return _value;
        }
    }

    /** Specifies whether the Tensor allocates memory */
    static public class AllocationFlag {
        private static final int _notAllocate = 1;
        private static final int _doAllocate  = 2;

        /**  Tensor does not allocate memory */
        public static final AllocationFlag NotAllocate = new AllocationFlag(_notAllocate);
        /** Tensor allocates memory when needed */
        public static final AllocationFlag DoAllocate  = new AllocationFlag(_doAllocate);

        private final int _value;

        AllocationFlag(final int value) {
            this._value = value;
        }

        public int ordinal() {
            return _value;
        }
    }

    /**
     * Reads subtensor from the tensor and returns it to
     * java.nio.DoubleBuffer. This method needs to be defined by user
     * in the subclass of this class.
     *
     * @param  fixedDims    The number fixed dimensions and values at which dimensions are fixed
     * @param  rangeDimIdx  Values for the next dimension after fixed to get data from
     * @param  rangeDimNum  Range for dimension values to get data from
     * @param  buf          Buffer to store results
     *
     * @return Subtensor packed into DoubleBuffer
     */
    public DoubleBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf) {
        return tensorImpl.getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /**
     * Reads subtensor from the tensor and returns it to
     * java.nio.FloatBuffer. This method needs to be defined by user in
     * the subclass of this class.
     *
     * @param  fixedDims    The number fixed dimensions and values at which dimensions are fixed
     * @param  rangeDimIdx  Values for the next dimension after fixed to get data from
     * @param  rangeDimNum  Range for dimension values to get data from
     * @param  buf          Buffer to store results
     *
     * @return Subtensor packed into FloatBuffer
     */
    public FloatBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf) {
        return tensorImpl.getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /**
     * Reads subtensor from the tensor and returns it to
     * java.nio.IntBuffer. This method needs to be defined by user in
     * the subclass of this class.
     *
     * @param  fixedDims    The number fixed dimensions and values at which dimensions are fixed
     * @param  rangeDimIdx  Values for the next dimension after fixed to get data from
     * @param  rangeDimNum  Range for dimension values to get data from
     * @param  buf          Buffer to store results
     *
     * @return Subtensor packed into IntBuffer
     */
    public IntBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf) {
        return tensorImpl.getSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /**
     * Transfers the data from the input DoubleBuffer into subtemsor of the tensor
     * This function needs to be defined by user in the subclass of
     * this class.
     *
     * @param  fixedDims    The number fixed dimensions and values at which dimensions are fixed
     * @param  rangeDimIdx  Values for the next dimension after fixed to get data from
     * @param  rangeDimNum  Range for dimension values to get data from
     * @param buf         Input DoubleBuffer with the subtensor data
     */
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf) {
        tensorImpl.releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /**
     * Transfers the data from the input FloatBuffer into subtemsor of the tensor
     * This function needs to be defined by user in the subclass of
     * this class.
     *
     * @param  fixedDims    The number fixed dimensions and values at which dimensions are fixed
     * @param  rangeDimIdx  Values for the next dimension after fixed to get data from
     * @param  rangeDimNum  Range for dimension values to get data from
     * @param buf         Input FloatBuffer with the subtensor data
     */
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf) {
        tensorImpl.releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /**
     * Transfers the data from the input IntBuffer into subtemsor of the tensor
     * This function needs to be defined by user in the subclass of
     * this class.
     *
     * @param  fixedDims    The number fixed dimensions and values at which dimensions are fixed
     * @param  rangeDimIdx  Values for the next dimension after fixed to get data from
     * @param  rangeDimNum  Range for dimension values to get data from
     * @param buf         Input IntBuffer with the subtensor data
     */
    public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf) {
        tensorImpl.releaseSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    /**
     *  Allocates memory for a data set
     */
    public void allocateDataMemory() {
        tensorImpl.allocateDataMemory();
    }

    /**
     *  Deallocates the memory allocated for a data set
     */
    public void freeDataMemory() {
        tensorImpl.freeDataMemory();
    }

    /**
     *  Gets dimensions of the tensor
     */
    public long[] getDimensions() {
        return tensorImpl.getDimensions();
    }

    /**
     *  Sets dimensions of the tensor
     */
    public void   setDimensions(long[] newDims) {
        tensorImpl.setDimensions(newDims);
    }

    /**
     *  Gets size of the tensor
     */
    public long getSize() {
        return tensorImpl.getSize();
    }

    /** @copydoc SerializableBase::getCObject() */
    @Override
    public long getCObject() {
        return tensorImpl.getCObject();
    }

    @Override
    protected boolean onSerializeCObject() {
        return false;
    }

    @Override
    protected void onPack() {
        if (tensorImpl != null) {
            tensorImpl.pack();
        }
    }

    @Override
    protected void onUnpack(DaalContext context) {
        if (tensorImpl != null) {
            tensorImpl.unpack(context);
        }
    }

    DoubleBuffer getDoubleSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        return tensorImpl.getDoubleSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    FloatBuffer getFloatSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        return tensorImpl.getFloatSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    IntBuffer getIntSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        return tensorImpl.getIntSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    void releaseDoubleSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        tensorImpl.releaseDoubleSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    void releaseFloatSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        tensorImpl.releaseFloatSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }

    void releaseIntSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, ByteBuffer buf) {
        tensorImpl.releaseIntSubtensor(fixedDims, rangeDimIdx, rangeDimNum, buf);
    }
}
