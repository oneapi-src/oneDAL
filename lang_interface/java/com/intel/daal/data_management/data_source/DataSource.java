/* file: DataSource.java */
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
 * \brief Contains classes that implement the data source component
 *        responsible for representation of the data in a raw format
 */
package com.intel.daal.data_management.data_source;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA_SOURCE__DATASOURCE"></a>
 *  @brief Abstract class that defines the interface for the data management component responsible for the
 *  representation of the data in a raw format. This class declares the most generic methods for data access.
 */
abstract public class DataSource extends ContextClient {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /** DataSource status type */
    static public class DataSourceStatus {
        private static final int _ReadyForLoad   = 1;
        private static final int _WaitingForRows = 2;
        private static final int _EndOfData      = 3;
        private static final int _NotReady       = 4;

        /** DataSource is ready for a loadDataBlock call */
        public static final DataSourceStatus ReadyForLoad   = new DataSourceStatus(_ReadyForLoad);
        /** No data available for a loadDataBlock, but there can be available data in future */
        public static final DataSourceStatus WaitingForRows = new DataSourceStatus(_WaitingForRows);
        /** No data available for a loadDataBlock, and it will not be available in future */
        public static final DataSourceStatus EndOfData      = new DataSourceStatus(_EndOfData);
        /** DataSource is not ready to be loaded */
        public static final DataSourceStatus NotReady       = new DataSourceStatus(_NotReady);

        private final int status;

        DataSourceStatus(final int status) {
            this.status = status;
        }

        public int ordinal() {
            return status;
        }
    }

    /** Specifies whether the Data Dictionary is created from the context of the Data Source */
    static public class DictionaryCreationFlag {
        private static final int _notDictionaryFromContext = 1;
        private static final int _doDictionaryFromContext  = 2;

        /** Does not create dictionary automatically */
        public static final DictionaryCreationFlag NotDictionaryFromContext = new DictionaryCreationFlag(
                _notDictionaryFromContext);
        /** Creates dictionary when needed */
        public static final DictionaryCreationFlag DoDictionaryFromContext  = new DictionaryCreationFlag(
                _doDictionaryFromContext);

        private final int _value;

        DictionaryCreationFlag(final int value) {
            this._value = value;
        }

        public int ordinal() {
            return _value;
        }
    }

    /** Specifies whether the Numeric Table is allocated inside of the Data Source object */
    static public class NumericTableAllocationFlag {
        private static final int _notAllocateNumericTable = 1;
        private static final int _doAllocateNumericTable  = 2;

        /** Does not allocate Numeric Table automatically */
        public static final NumericTableAllocationFlag NotAllocateNumericTable = new NumericTableAllocationFlag(
                _notAllocateNumericTable);
        /** Allocates Numeric Table when needed */
        public static final NumericTableAllocationFlag DoAllocateNumericTable  = new NumericTableAllocationFlag(
                _doAllocateNumericTable);

        private final int _value;

        NumericTableAllocationFlag(final int value) {
            this._value = value;
        }

        public int ordinal() {
            return _value;
        }
    }

    /**
     * Default constructor
     */
    DataSource(DaalContext context) {
        super(context);
        cObject = 0;
    }

    /**
     *  Creates the Data Dictionary by extracting information from a Data Source
     */
    public void createDictionaryFromContext() {
        cCreateDictionaryFromContext(cObject);
    }

    private native void cCreateDictionaryFromContext(long ptr);

    /**
     *  Returns the number of columns in the Data Source
     *  @return Number of columns
     */
    public long getNumberOfColumns() {
        return cGetNumberOfColumns(cObject);
    }

    private native long cGetNumberOfColumns(long ptr);

    /**
     *  Returns the number of rows available in the Data Source
     *  @return Number of rows
     */
    public long getNumberOfAvailableRows() {
        return cGetNumberOfAvailableRows(cObject);
    }

    private native long cGetNumberOfAvailableRows(long ptr);

    /**
     *  Allocates the Numeric Table associated with the Data Source
     */
    public void allocateNumericTable() {
        cAllocateNumericTable(cObject);
    }

    private native void cAllocateNumericTable(long ptr);

    /**
     *  Deallocates a Numeric Table associated with the Data Source
     */
    public void freeNumericTable() {
        cFreeNumericTable(cObject);
    }

    private native void cFreeNumericTable(long ptr);

    /**
     *  Returns the Numeric Table associated with the Data Source
     *  @return Numeric Table associated with the Data Source
     */
    public NumericTable getNumericTable() {
        return new HomogenNumericTable(getContext(), cGetNumericTable(cObject));
    }

    private native long cGetNumericTable(long ptr);

    /**
     *  Loads a data block of the specified size into internally allocated Numeric Table
     *  @param maxRows Maximum number of rows to load from the Data Source into the Numeric Table
     *  @return Number of rows loaded from a Data Source into the Numeric Table
     */
    public long loadDataBlock(long maxRows) {
        return cLoadDataBlock(cObject, maxRows);
    }

    private native long cLoadDataBlock(long ptr, long maxRows);

    /**
     *  Loads a data block of the specified size into internally allocated Numeric Table
     *  @param maxRows   Maximum number of rows to load from a Data Source into the Numeric Table
     *  @param offset Write data starting from offset row
     *  @param fullRows  Maximum number of rows to allocate in the Numeric Table
     *  @return Number of rows loaded from a Data Source into the Numeric Table
     */
    public long loadDataBlock(long maxRows, long offset, long fullRows) {
        return cLoadDataBlock3Inputs(cObject, maxRows, offset, fullRows);
    }

    private native long cLoadDataBlock3Inputs(long ptr, long maxRows, long offset, long fullRows);

    /**
     *  Loads a data block of the specified size into provided Numeric Table
     *  @param maxRows Maximum number of rows to load from the Data Source into the Numeric Table
     *  @param nt      Pointer to the Numeric Table
     *  @return Number of rows loaded from a Data Source into the Numeric Table
     */
    public long loadDataBlock(long maxRows, NumericTable nt) {
        return cLoadDataBlockNt(cObject, maxRows, nt.getCObject());
    }

    private native long cLoadDataBlockNt(long ptr, long maxRows, long numericTable);

    /**
     *  Loads a data block into provided Numeric Table
     *  @param nt      Pointer to the Numeric Table
     *  @return Number of rows loaded from a Data Source into the Numeric Table
     */
    public long loadDataBlock(NumericTable nt) {
        return cLoadDataBlockNt1Input(cObject, nt.getCObject());
    }

    private native long cLoadDataBlockNt1Input(long ptr, long numericTable);

    /**
     *  Loads a data block into internally allocated Numeric Table
     *  @return Number of rows loaded from a Data Source into the Numeric Table
     */
    public long loadDataBlock() {
        return cLoadDataBlock0Inputs(cObject);
    }

    private native long cLoadDataBlock0Inputs(long ptr);

    /* Pointer to the C++ implementation of the DataSource */
    protected long cObject;

    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    public void pack() {
        DaalContext myContext = getContext();
        if (myContext != null) {
            myContext.remove(this);
            myContext = null;
        }
        dispose();
    }

    public void unpack(DaalContext context) {
        if (getContext() != null) {
            /* Error */
        }
        changeContext(context);
    }

    private native void cDispose(long ptr);
}
