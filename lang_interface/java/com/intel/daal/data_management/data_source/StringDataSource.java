/* file: StringDataSource.java */
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

package com.intel.daal.data_management.data_source;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;

import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA_SOURCE__STRINGDATASOURCE"></a>
 * @brief Specifies the methods for accessing the data stored as a text in java.io.Strings format
 */
public class StringDataSource extends DataSource implements Serializable {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Default constructor
     */
    public StringDataSource(DaalContext context, String data) {
        super(context);

        cObject = cInit(data, data.length());
        _dataForSerialization = data;
    }

    /**
     * Constructor
     */
    public StringDataSource(DaalContext context, String data, DictionaryCreationFlag doDict,
            NumericTableAllocationFlag doNT) {
        super(context);

        cObject = cInit(data, data.length());
        _dataForSerialization = data;

        if (doDict.ordinal() == DictionaryCreationFlag.DoDictionaryFromContext.ordinal()) {
            this.createDictionaryFromContext();
        }

        if (doNT.ordinal() == NumericTableAllocationFlag.DoAllocateNumericTable.ordinal()) {
            this.allocateNumericTable();
        }
    }

    protected native long cInit(String data, long length);

    /**
     *  Sets new string as a source for the data
     *  @param  data  Strings with the data
     */
    public void setData(String data) {
        cSetData(cObject, data, data.length());
    }

    private native void cSetData(long ptr, String data, long length);

    protected String _dataForSerialization;

    private void readObject(ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {
        aInputStream.defaultReadObject();
        cObject = cInit(_dataForSerialization, _dataForSerialization.length());
    }

    private native void cDispose(long ptr);
}
