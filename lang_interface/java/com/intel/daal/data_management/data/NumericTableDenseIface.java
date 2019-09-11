/* file: NumericTableDenseIface.java */
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
 * @ingroup numeric_tables
 * @{
 */
package com.intel.daal.data_management.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

interface NumericTableDenseIface {

    /**
     * Reads block of rows from the table and returns it to
     *        java.nio.DoubleBuffer. This method needs to be defined by user
     *        in the subclass of this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Buffer to store results
     *
     * @return Block of table rows packed into DoubleBuffer
     */
    abstract public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf);

    /**
     * Reads block of rows from the table and returns it to
     *        java.nio.FloatBuffer. This method needs to be defined by user in
     *        the subclass of this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Buffer to store results
     *
     * @return Block of table rows packed into FloatBuffer
     */
    abstract public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf);

    /**
     * Reads block of rows from the table and returns it to
     *        java.nio.IntBuffer. This method needs to be defined by user in
     *        the subclass of this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Buffer to store results
     *
     * @return Block of table rows packed into IntBuffer
     */
    abstract public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf);

    /**
     * Transfers the data from the input DoubleBuffer into a block of table
     *        rows. This function needs to be defined by user in the subclass of
     *        this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Input DoubleBuffer with the capacity vectorNum * nColumns, where
     *                         nColumns is the number of columns in the table
     */
    abstract public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf);

    /**
     * Transfers the data from the input FloatBuffer into a block of table
     * rows. This function needs to be defined by user in the subclass of
     * this class.
     *
     * @param  vectorIndex Index of the first row to include into the block
     * @param  vectorNum   Number of rows in the block
     * @param buf         Input FloatBuffer with the capacity vectorNum * nColumns, where
     *                         nColumns is the number of columns in the table
     */
    abstract public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf);

    /**
     * Transfers the data from the input IntBuffer into a block of table
     * rows. This function needs to be defined by user in the subclass of
     * this class.
     *
     * @param vectorIndex Index of the first row to include into the block
     * @param vectorNum   Number of rows in the block
     * @param buf         Input IntBuffer with the capacity vectorNum * nColumns, where
     *                    nColumns is the number of columns in the table
     */
    abstract public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf);

    /**
     * Gets block of values for a given feature and returns it to
     * java.nio.DoubleBuffer. This function needs to be defined by user
     * in the subclass of this class.
     *
     * @param  featureIndex Index of the feature
     * @param  vectorIndex  Index of the first row to include into the block
     * @param  vectorNum    Number of values in the block
     * @param buf          Buffer to store results
     *
     * @return Block of values of the feature packed into the DoubleBuffer
     */
    abstract public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            DoubleBuffer buf);

    /**
     * Gets block of values for a given feature and returns it to
     *        java.nio.FloatBuffer. This function needs to be defined by user in
     *        the subclass of this class.
     *
     * @param  featureIndex Index of the feature
     * @param  vectorIndex  Index of the first row to include into the block
     * @param  vectorNum    Number of values in the block
     * @param buf          Buffer to store results
     *
     * @return Block of values of the feature packed into the FloatBuffer
     */
    abstract public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            FloatBuffer buf);

    /**
     * Gets block of values for a given feature and returns it to
     *        java.nio.IntBuffer. This function needs to be defined by user in
     *        the subclass of this class.
     *
     * @param  featureIndex Index of the feature
     * @param  vectorIndex  Index of the first row to include into the block
     * @param  vectorNum    Number of values in the block
     * @param  buf          Buffer to store results
     *
     * @return Block of values of the feature packed into the IntBuffer
     */
    abstract public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            IntBuffer buf);

    /**
     * Transfers the values of a given feature from the input DoubleBuffer
     *        into a block of values of the feature in the table. This function needs
     *        to be defined by user in the subclass of this class.
     *
     * @param featureIndex Index of the feature
     * @param vectorIndex  Index of the first row to include into the block
     * @param vectorNum    Number of values in the block
     * @param buf          Input DoubleBuffer of size vectorNum
     */
    abstract public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            DoubleBuffer buf);

    /**
     * Transfers the values of a given feature from the input FloatBuffer
     *        into a block of values of the feature in the table. This function needs
     *        to be defined by user in the subclass of this class.
     *
     * @param featureIndex Index of the feature
     * @param vectorIndex  Index of the first row to include into the block
     * @param vectorNum    Number of values in the block
     * @param buf          Input FloatBuffer of size vectorNum
     */
    abstract public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            FloatBuffer buf);

    /**
     * Transfers the values of a given feature from the input IntBuffer
     *        into a block of values of the feature in the table. This function needs
     *        to be defined by user in the subclass of this class.
     *
     * @param featureIndex Index of the feature
     * @param vectorIndex  Index of the first row to include into the block
     * @param vectorNum    Number of values in the block
     * @param buf          Input IntBuffer of size vectorNum
     */
    abstract public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf);
}
/** @} */
