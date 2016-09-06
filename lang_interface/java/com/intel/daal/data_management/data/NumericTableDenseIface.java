/* file: NumericTableDenseIface.java */
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
