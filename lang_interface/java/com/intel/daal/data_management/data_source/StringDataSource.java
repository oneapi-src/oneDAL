/* file: StringDataSource.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup data_sources
 * @{
 */
package com.intel.daal.data_management.data_source;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
    }

    /**
     * Default constructor
     */
    public StringDataSource(DaalContext context, String data) {
        super(context);

        cObject = cInit(data, data.length());
        featureManager = new FeatureManager(context, cGetFeatureManager(cObject));
        _dataForSerialization = data;
    }

    /**
     * Constructor
     */
    public StringDataSource(DaalContext context, String data, DictionaryCreationFlag doDict,
            NumericTableAllocationFlag doNT) {
        super(context);

        cObject = cInit(data, data.length());
        featureManager = new FeatureManager(context, cGetFeatureManager(cObject));
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
        featureManager = new FeatureManager(getContext(), cGetFeatureManager(cObject));
    }

    private native void cDispose(long ptr);
    private native long cGetFeatureManager(long cObject);
}
/** @} */
