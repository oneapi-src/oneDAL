/* file: MergedNumericTableImpl.java */
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

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Vector;

import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__MERGEDNUMERICTABLEIMPL"></a>
 *  @brief Class that provides methods to access a collection of numeric tables as if they are joined by columns
 */
public class MergedNumericTableImpl extends NumericTableImpl {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs empty merged numeric table
     *
     * @param context   Context to manage created merged numeric table
     */
    public MergedNumericTableImpl(DaalContext context) {
        super(context);
        dataAllocatedInJava = false;
        this.cObject = cNewMergedNumericTable();
        dict = new DataDictionary(getContext(), (long)0, cGetCDataDictionary(cObject));
    }

    /**
     * Constructs merged numeric table consisting of one table
     *
     * @param context   Context to manage created merged numeric table
     * @param table     Pointer to the Numeric Table
     */
    public MergedNumericTableImpl(DaalContext context, NumericTable table) {
        super(context);
        dataAllocatedInJava = false;
        this.cObject = cNewMergedNumericTable();
        dict = new DataDictionary(getContext(), (long)0, cGetCDataDictionary(cObject));
        addNumericTable(table);
    }

    /**
     * Constructs merged numeric table consisting of two tables
     *
     * @param context   Context to manage created merged numeric table
     * @param first     Pointer to the first Numeric Table
     * @param second    Pointer to the second Numeric Table
     */
    public MergedNumericTableImpl(DaalContext context, NumericTable first, NumericTable second) {
        super(context);
        dataAllocatedInJava = false;
        this.cObject = cNewMergedNumericTable();
        dict = new DataDictionary(getContext(), (long)0, cGetCDataDictionary(cObject));
        addNumericTable(first);
        addNumericTable(second);
    }

    /**
     *  Adds the table to the right of the merged numeric table
     *  \param table    Pointer to the Numeric Table
     */
    public void addNumericTable(NumericTable table) {
        if ((table.getDataLayout().ordinal() & NumericTable.StorageLayout.csrArray.ordinal()) != 0) {
            throw new IllegalArgumentException("can not join numeric table in csr format to merged numeric table");
        } else {
            cAddNumericTable(this.cObject, table.getCObject());

            int ncols = (int)(getNumberOfColumns());
            int cols = (int)(table.getNumberOfColumns());

            dict.setNumberOfFeatures(ncols);

            for (int i = 0; i < cols; i++) {
                DataFeature f = table.getDictionary().getFeature(i);
                dict.setFeature(f, ncols - cols + i);
            }
        }
    }

    /** @copydoc NumericTable::getNumberOfColumns() */
    @Override
    public long getNumberOfColumns() {
        return cGetNumberOfColumns(this.cObject);
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getDoubleBlockBuffer(this.cObject, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asDoubleBuffer();
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getFloatBlockBuffer(this.cObject, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asFloatBuffer();
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getIntBlockBuffer(this.cObject, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asIntBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getDoubleColumnBuffer(this.cObject, featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asDoubleBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getFloatColumnBuffer(this.cObject, featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asFloatBuffer();
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) getNumberOfColumns();
        int bufferSize = (int) vectorNum;
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf = getIntColumnBuffer(this.cObject, featureIndex, vectorIndex, vectorNum, byteBuf);
        return byteBuf.asIntBuffer();
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);

        float[] data = new float[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(data);
        releaseFloatBlockBuffer(this.cObject, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);

        double[] data = new double[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(data);
        releaseDoubleBlockBuffer(this.cObject, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum * nColumns);

        int[] data = new int[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(data);
        releaseIntBlockBuffer(this.cObject, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum);

        double[] data = new double[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 8 /* sizeof(double) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asDoubleBuffer().put(data);
        releaseDoubleColumnBuffer(this.cObject, featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum);

        float[] data = new float[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(float) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asFloatBuffer().put(data);
        releaseFloatColumnBuffer(this.cObject, featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int bufferSize = (int) (vectorNum);

        int[] data = new int[buf.capacity()];
        buf.position(0);
        buf.get(data);
        // Gets data from C++ NumericTable object
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(bufferSize * 4 /* sizeof(int) */);
        byteBuf.order(ByteOrder.LITTLE_ENDIAN);
        byteBuf.asIntBuffer().put(data);
        releaseIntColumnBuffer(this.cObject, featureIndex, vectorIndex, vectorNum, byteBuf);
    }

    /* Gets NIO buffer containing data of the C++ table */

    private native ByteBuffer getDoubleBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native ByteBuffer getFloatBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native ByteBuffer getIntBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native void releaseDoubleBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native void releaseFloatBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native void releaseIntBlockBuffer(long cObject, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native ByteBuffer getDoubleColumnBuffer(long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native ByteBuffer getFloatColumnBuffer (long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native ByteBuffer getIntColumnBuffer   (long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native void releaseDoubleColumnBuffer(long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native void releaseFloatColumnBuffer (long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);
    private native void releaseIntColumnBuffer   (long cObject, long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buffer);

    private native long cNewMergedNumericTable();
    private native long cGetNumberOfColumns(long cObject);
    private native void cAddNumericTable(long cObject, long numericTableAddr);
}
