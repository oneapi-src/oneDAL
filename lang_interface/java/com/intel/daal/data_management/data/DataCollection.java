/* file: DataCollection.java */
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
 * @defgroup data_model Data Model
 * @brief Contains classes that provide functionality of Collection container for objects derived from SerializableBase
 * @ingroup data_management
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__DATACOLLECTION"></a>
 *  @brief Class that provides functionality of the Collection container for Serializable objects
 */
public class DataCollection extends SerializableBase {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the collection
     * @param context   Context to manage the collection
     */
    public DataCollection(DaalContext context) {
        super(context);
        this.cObject = cNewDataCollection();
    }

    public DataCollection(DaalContext context, long cDataCollection) {
        super(context);
        this.cObject = cDataCollection;
        this.serializedCObject = null;
    }

    public long size() {
        return cSize(this.cObject);
    }

    public SerializableBase get(long idx) {
        return Factory.instance().createObject(getContext(), cGetValue(this.cObject, idx));
    }

    public void set(SerializableBase value, long idx) {
        cSetValue(this.cObject, value.getCObject(), idx);
    }

    private native long cNewDataCollection();

    private native long cSize(long cDataCollectionAddr);

    private native long cGetValue(long cDataCollectionAddr, long idx);

    private native void cSetValue(long cDataCollectionAddr, long cValueAddr, long idx);
}
/** @} */
