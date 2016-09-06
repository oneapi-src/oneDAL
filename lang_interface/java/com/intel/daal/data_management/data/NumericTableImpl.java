/* file: NumericTableImpl.java */
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
 *        responsible for representaion of the data in numerical format
 */
package com.intel.daal.data_management.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__NUMERICTABLEIMPL"></a>
 *  @brief  Class for the data management component responsible for the representation of the data in a numerical format.
 */
abstract public class NumericTableImpl extends SerializableBase {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    protected NumericTableImpl(DaalContext context) {
        super(context);
        nJavaFeatures = 0;
        nJavaVectors = 0;
        jData = null;
        dict = null;
        dataAllocatedInJava = false;
    }

    /** @copydoc NumericTable::allocateDataMemory() */
    public void allocateDataMemory() {
        checkCObject();
        cAllocateDataMemory();
    }

    protected native void cAllocateDataMemory();

    /** @copydoc NumericTable::freeDataMemory() */
    public void freeDataMemory() {
        checkCObject();
        cFreeDataMemory();
    }

    protected native void cFreeDataMemory();

    /**
     * Gets number of columns in the table
     *
     * @return Number of columns in the table
     */
    public long getNumberOfColumns() {
        if (dataAllocatedInJava) {
            return nJavaFeatures;
        } else
        if (cObject != 0) {
            return cGetNumberOfColumns();
        } else
        if (serializedCObject != null) {
            return nSerializedFeatures;
        } else {
            throw new IllegalArgumentException("number of columns is undefined");
        }
    }

    protected native long cGetNumberOfColumns();

    /**
     * Gets number of rows in the table
     *
     * @return Number of rows in the table
     */
    public long getNumberOfRows() {
        if (dataAllocatedInJava) {
            return nJavaVectors;
        } else
        if (cObject != 0) {
            return cGetNumberOfRows();
        } else
        if (serializedCObject != null) {
            return nSerializedVectors;
        } else {
            throw new IllegalArgumentException("number of rows is undefined");
        }
    }

    protected native long cGetNumberOfRows();

    /**
     * Sets number of rows in the table
     *
     * @param nRow Number of rows
     */
    public void setNumberOfRows(long nRow) {
        checkCObject();
        cSetNumberOfRows(nRow);
    }

    private native void cSetNumberOfRows(long nRow);

    /**
     * Sets number of columns in the table
     *
     * @param nCol Number of columns
     */
    public void setNumberOfColumns(long nCol) {
        checkCObject();
        cSetNumberOfColumns(nCol);
    }

    private native void cSetNumberOfColumns(long nCol);

    /**
     * Sets the normalization flag for dataset stored in the numeric table
     *
     * @param flag Normalization flag
     * @return Previous value of the normalization flag
     */
    public NumericTable.NormalizationType setNormalizationFlag(NumericTable.NormalizationType flag) {
        checkCObject();
        return new NumericTable.NormalizationType(cSetNormalizationFlag(flag.ordinal()));
    }

    private native int cSetNormalizationFlag(int flag);

    /**
     *  Checks if dataset stored in the numeric table is normalized, according to the given normalization flag
     *  @param flag Normalization flag to check
     *  @return Check result
     */
    public boolean isNormalized(NumericTable.NormalizationType flag) {
        checkCObject();
        return cIsNormalized(flag.ordinal());
    }

    private native boolean cIsNormalized(int flag);

    /**
     * Returns the data dictionary
     *
     * @return Data dictionary
     */
    public DataDictionary getDictionary() {
        return dict;
    }

    /**
     *  Sets a data dictionary in the Numeric Table
     *  @param ddict Pointer to the data dictionary
     */
    public void setDictionary(DataDictionary ddict) {
        dict = ddict;
        checkCObject();
        cSetCDataDictionary(cObject, ddict.getCObject());
    }

    protected native void cSetCDataDictionary(long cTable, long cDictionary);

    /**
     * Return data storage layout
     *
     * @return Data storage Layout
     */
    public NumericTable.StorageLayout getDataLayout() {
        checkCObject();
        return new NumericTable.StorageLayout(cGetDataLayout(cObject));
    }

    private native int cGetDataLayout(long cObject);

    /**
     *  Return the status of the memory used by a data set connected with a Numeric Table
     *
     *  @return Status of the memory used by a data set connected with a Numeric Table
     */
    NumericTable.MemoryStatus getDataMemoryStatus() {
        checkCObject();
        return new NumericTable.MemoryStatus(cGetDataMemoryStatus(cObject));
    }

    private native int cGetDataMemoryStatus(long cObject);

    /**
     *  Returns the type of a given feature
     *  @param idx Feature index
     *
     *  @return Feature type
     */
    public DataFeature getFeatureType(int idx) {
        return dict.getFeature(idx);
    }

    /**
     *
     *
     */
    public long getNumberOfCategories(int idx) {
        checkCObject();
        return cGetNumberOfCategories(cObject, idx);
    }

    private native long cGetNumberOfCategories(long cObject, int idx);

    @Override
    protected void onPack() {
        nSerializedFeatures = getNumberOfColumns();
        nSerializedVectors = getNumberOfRows();
    }

    @Override
    protected void onUnpack(DaalContext context) {
        deserializeCObject();
        dict = new DataDictionary(context, (long)0, cGetCDataDictionary(cObject));
    }

    DoubleBuffer getDoubleBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer dBuf = getBlockOfRows(vectorIndex, vectorNum, buf.asDoubleBuffer());
        return dBuf;
    }

    FloatBuffer getFloatBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer sBuf = getBlockOfRows(vectorIndex, vectorNum, buf.asFloatBuffer());
        return sBuf;
    }

    IntBuffer getIntBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.position(0);
        IntBuffer iBuf = getBlockOfRows(vectorIndex, vectorNum, buf.asIntBuffer());
        buf.position(0);
        return iBuf;
    }

    void releaseDoubleBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        releaseBlockOfRows(vectorIndex, vectorNum, buf.asDoubleBuffer());
    }

    void releaseFloatBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        releaseBlockOfRows(vectorIndex, vectorNum, buf.asFloatBuffer());
    }

    void releaseIntBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.position(0);
        releaseBlockOfRows(vectorIndex, vectorNum, buf.asIntBuffer());
        buf.position(0);
    }

    DoubleBuffer getDoubleFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer dBuf = getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf.asDoubleBuffer());
        return dBuf;
    }

    FloatBuffer getFloatFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer sBuf = getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf.asFloatBuffer());
        return sBuf;
    }

    IntBuffer getIntFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer iBuf = getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf.asIntBuffer());
        return iBuf;
    }

    void releaseDoubleFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf.asDoubleBuffer());
    }

    void releaseFloatFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf.asFloatBuffer());
    }

    void releaseIntFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf.asIntBuffer());
    }

    /* Java data associated with this table */
    protected Object jData;

    /* True if data for the table allocation is on Java side */
    protected boolean dataAllocatedInJava;

    /* Data dictionary */
    protected DataDictionary dict;

    protected long nJavaFeatures;
    protected long nJavaVectors;
    protected long nSerializedFeatures;
    protected long nSerializedVectors;

    protected long newJavaNumericTable(long nColumns, long nRows, NumericTable.StorageLayout layout) {
        nJavaFeatures = nColumns;
        nJavaVectors = nRows;
        return cNewJavaNumericTable(nColumns, nRows, layout.ordinal());
    }

    private native long cNewJavaNumericTable(long nColumns, long nRows, int layout);
    protected native long cGetCDataDictionary(long cTable);

    @Override
    protected boolean onSerializeCObject() {
        return !dataAllocatedInJava;
    }

    abstract public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf);

    abstract public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf);

    abstract public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf);

    abstract public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf);

    abstract public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf);

    abstract public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf);

    abstract public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            DoubleBuffer buf);

    abstract public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            FloatBuffer buf);

    abstract public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            IntBuffer buf);

    abstract public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            DoubleBuffer buf);

    abstract public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            FloatBuffer buf);

    abstract public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf);
}
