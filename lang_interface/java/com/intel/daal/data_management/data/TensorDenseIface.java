/* file: TensorDenseIface.java */
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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

interface TensorDenseIface {

    /**
     * Reads subtensor the tensor and returns it to
     *        java.nio.DoubleBuffer. This method needs to be defined by user
     *        in the subclass of this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Buffer to store results
     *
     * @return Subtensor packed into DoubleBuffer
     */
    abstract public DoubleBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf);

    /**
     * Reads subtensor the tensor and returns it to
     *        java.nio.FloatBuffer. This method needs to be defined by user in
     *        the subclass of this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Buffer to store results
     *
     * @return Subtensor packed into FloatBuffer
     */
    abstract public FloatBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf);

    /**
     * Reads subtensor the tensor and returns it to
     *        java.nio.IntBuffer. This method needs to be defined by user in
     *        the subclass of this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Buffer to store results
     *
     * @return Subtensor packed into IntBuffer
     */
    abstract public IntBuffer getSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf);

    /**
     * Transfers the data from the input DoubleBuffer subtensor of the tensor
     * This function needs to be defined by user in the subclass of
     * this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Input DoubleBuffer with the subtensor data
     */
    abstract public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, DoubleBuffer buf);

    /**
     * Transfers the data from the input FloatBuffer subtensor of the tensor
     * This function needs to be defined by user in the subclass of
     * this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Input FloatBuffer with the subtensor data
     */
    abstract public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, FloatBuffer buf);

    /**
     * Transfers the data from the input IntBuffer subtensor of the tensor
     * This function needs to be defined by user in the subclass of
     * this class.
     *
     * @param vectorIndex Index of the first row to include into the block
     * @param vectorNum   Number of rows in the block
     * @param buf         Input IntBuffer with the subtensor data
     */
    abstract public void releaseSubtensor(long[] fixedDims, long rangeDimIdx, long rangeDimNum, IntBuffer buf);
}
/** @} */
