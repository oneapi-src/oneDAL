/* file: NumericTable.java */
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__NUMERICTABLE"></a>
 *  @anchor NumericTable
 *  @brief  Class for the data management component responsible for the representation of the data in a numerical format.
 */
abstract public class NumericTable extends SerializableBase implements NumericTableDenseIface {
    protected NumericTableImpl tableImpl;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    protected NumericTable(DaalContext context) {
        super(context);
    }

    /** Specifies whether the Numeric Table allocates memory */
    static public class StorageLayout {
        private static final int _soa                         = 1;
        private static final int _aos                         = 2;
        private static final int _csrArray                    = 1 << 4;
        private static final int _upperPackedSymmetricMatrix  = 1 << 8;
        private static final int _lowerPackedSymmetricMatrix  = 2 << 8;
        private static final int _upperPackedTriangularMatrix = 3 << 8;
        private static final int _lowerPackedTriangularMatrix = 4 << 8;
        private static final int _layout_unknown              = 0x80000000;

        public static final StorageLayout soa                         = new StorageLayout(_soa);
        public static final StorageLayout aos                         = new StorageLayout(_aos);
        public static final StorageLayout csrArray                    = new StorageLayout(_csrArray);
        public static final StorageLayout upperPackedSymmetricMatrix  = new StorageLayout(_upperPackedSymmetricMatrix);
        public static final StorageLayout lowerPackedSymmetricMatrix  = new StorageLayout(_lowerPackedSymmetricMatrix);
        public static final StorageLayout upperPackedTriangularMatrix = new StorageLayout(_upperPackedTriangularMatrix);
        public static final StorageLayout lowerPackedTriangularMatrix = new StorageLayout(_lowerPackedTriangularMatrix);
        public static final StorageLayout layout_unknown              = new StorageLayout(_layout_unknown);

        private final int _value;

        StorageLayout(final int value) {
            this._value = value;
        }

        public int ordinal() {
            return _value;
        }
    }

    /** Specifies the status of memory related to the Numeric Table */
    static public class MemoryStatus {
        private static final int _notAllocated        = 0;
        private static final int _userAllocated       = 1;
        private static final int _internallyAllocated = 2;

        public static final MemoryStatus notAllocated = new MemoryStatus(_notAllocated);
        public static final MemoryStatus userAllocated = new MemoryStatus(_userAllocated);
        public static final MemoryStatus internallyAllocated = new MemoryStatus(_internallyAllocated);

        private final int _value;

        MemoryStatus(final int value) {
            this._value = value;
        }

        public int ordinal() {
            return _value;
        }
    }

    /** Specifies whether the Numeric Table allocates memory */
    static public class AllocationFlag {
        private static final int _notAllocate = 1;
        private static final int _doAllocate  = 2;

        /**  NumericTable does not allocate memory */
        public static final AllocationFlag NotAllocate = new AllocationFlag(_notAllocate);
        /** NumericTable allocates memory when needed */
        public static final AllocationFlag DoAllocate  = new AllocationFlag(_doAllocate);

        private final int _value;

        AllocationFlag(final int value) {
            this._value = value;
        }

        public int ordinal() {
            return _value;
        }
    }

    /** Specifies types of normalization */
    static public class NormalizationType {
        private static final int _nonNormalized           = 0;
        private static final int _standardScoreNormalized = 1;

        /**  Default: non-normalized */
        public static final NormalizationType nonNormalized           = new NormalizationType(_nonNormalized);
        /** Standard score normalization (mean=0, variance=1) */
        public static final NormalizationType standardScoreNormalized = new NormalizationType(_standardScoreNormalized);

        private final int _value;

        NormalizationType(final int value) {
            this._value = value;
        }

        public int ordinal() {
            return _value;
        }
    }

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
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        return tableImpl.getBlockOfRows(vectorIndex, vectorNum, buf);
    }

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
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        return tableImpl.getBlockOfRows(vectorIndex, vectorNum, buf);
    }

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
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        return tableImpl.getBlockOfRows(vectorIndex, vectorNum, buf);
    }

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
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        tableImpl.releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

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
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        tableImpl.releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

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
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        tableImpl.releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

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
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            DoubleBuffer buf) {
        return tableImpl.getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

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
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            FloatBuffer buf) {
        return tableImpl.getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

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
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            IntBuffer buf) {
        return tableImpl.getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

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
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            DoubleBuffer buf) {
        tableImpl.releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

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
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum,
            FloatBuffer buf) {
        tableImpl.releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

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
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        tableImpl.releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /**
     *  Allocates memory for a data set
     */
    public void allocateDataMemory() {
        tableImpl.allocateDataMemory();
    }

    /**
     *  Deallocates the memory allocated for a data set
     */
    public void freeDataMemory() {
        tableImpl.freeDataMemory();
    }

    /**
     * Gets number of columns in the table
     *
     * @return Number of columns in the table
     */
    public long getNumberOfColumns() {
        return tableImpl.getNumberOfColumns();
    }

    /**
     * Gets number of rows in the table
     *
     * @return Number of rows in the table
     */
    public long getNumberOfRows() {
        return tableImpl.getNumberOfRows();
    }

    /**
     * Sets number of rows in the table
     *
     * @param nRow Number of rows
     */
    public void setNumberOfRows(long nRow) {
        tableImpl.setNumberOfRows(nRow);
    }

    /**
     * Sets number of columns in the table
     *
     * @param nCol Number of columns
     */
    public void setNumberOfColumns(long nCol) {
        tableImpl.setNumberOfColumns(nCol);
    }

    /**
     * Sets the normalization flag for dataset stored in the numeric table
     *
     * @param flag Normalization flag
     * @return Previous value of the normalization flag
     */
    public NormalizationType setNormalizationFlag(NormalizationType flag) {
        return tableImpl.setNormalizationFlag(flag);
    }

    /**
     *  Checks if dataset stored in the numeric table is normalized, according to the given normalization flag
     *  @param flag Normalization flag to check
     *  @return Check result
     */
    public boolean isNormalized(NormalizationType flag) {
        return tableImpl.isNormalized(flag);
    }

    /**
     * Returns the data dictionary
     *
     * @return Data dictionary
     */
    public DataDictionary getDictionary() {
        return tableImpl.getDictionary();
    }

    /**
     *  Sets a data dictionary in the Numeric Table
     *  @param ddict Pointer to the data dictionary
     */
    public void setDictionary(DataDictionary ddict) {
        tableImpl.setDictionary(ddict);
    }

    /**
     * Return data storage layout
     *
     * @return Data storage Layout
     */
    public StorageLayout getDataLayout() {
        return tableImpl.getDataLayout();
    }

    /**
     *  Return the status of the memory used by a data set connected with a Numeric Table
     *
     *  @return Status of the memory used by a data set connected with a Numeric Table
     */
    MemoryStatus getDataMemoryStatus() {
        return tableImpl.getDataMemoryStatus();
    }

    /**
     *  Returns the type of a given feature
     *  @param idx Feature index
     *
     *  @return Feature type
     */
    public DataFeature getFeatureType(int idx) {
        return tableImpl.getFeatureType(idx);
    }

    /**
     *
     *
     */
    public long getNumberOfCategories(int idx) {
        return tableImpl.getNumberOfCategories(idx);
    }

    /** @copydoc SerializableBase::getCObject() */
    @Override
    public long getCObject() {
        return tableImpl.getCObject();
    }

    /** @copydoc SerializableBase::getCObject() */
    public long getCNumericTable() {
        return getCObject();
    }

    @Override
    protected boolean onSerializeCObject() {
        return false;
    }

    @Override
    protected void onPack() {
        if (tableImpl != null) {
            tableImpl.pack();
        }
    }

    @Override
    protected void onUnpack(DaalContext context) {
        if (tableImpl != null) {
            tableImpl.unpack(context);
        }
    }

    DoubleBuffer getDoubleBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        return tableImpl.getDoubleBlock(vectorIndex, vectorNum, buf);
    }

    FloatBuffer getFloatBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        return tableImpl.getFloatBlock(vectorIndex, vectorNum, buf);
    }

    IntBuffer getIntBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        return tableImpl.getIntBlock(vectorIndex, vectorNum, buf);
    }

    void releaseDoubleBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        tableImpl.releaseDoubleBlock(vectorIndex, vectorNum, buf);
    }

    void releaseFloatBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        tableImpl.releaseFloatBlock(vectorIndex, vectorNum, buf);
    }

    void releaseIntBlock(long vectorIndex, long vectorNum, ByteBuffer buf) {
        tableImpl.releaseIntBlock(vectorIndex, vectorNum, buf);
    }

    DoubleBuffer getDoubleFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        return tableImpl.getDoubleFeature(featureIndex, vectorIndex, vectorNum, buf);
    }

    FloatBuffer getFloatFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        return tableImpl.getFloatFeature(featureIndex, vectorIndex, vectorNum, buf);
    }

    IntBuffer getIntFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        return tableImpl.getIntFeature(featureIndex, vectorIndex, vectorNum, buf);
    }

    void releaseDoubleFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        tableImpl.releaseDoubleFeature(featureIndex, vectorIndex, vectorNum, buf);
    }

    void releaseFloatFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        tableImpl.getFloatFeature(featureIndex, vectorIndex, vectorNum, buf);
    }

    void releaseIntFeature(long featureIndex, long vectorIndex, long vectorNum, ByteBuffer buf) {
        tableImpl.getIntFeature(featureIndex, vectorIndex, vectorNum, buf);
    }
}
