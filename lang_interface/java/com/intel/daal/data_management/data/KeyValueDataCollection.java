/* file: KeyValueDataCollection.java */
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
 * @ingroup data_model
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__KEYVALUEDATACOLLECTION"></a>
 *  @brief Class that provides functionality of the key-value container for Serializable objects
 *  with the key of integer type
 */
public class KeyValueDataCollection extends SerializableBase {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the key-value container for Serializable objects
     * @param context   Context to manage the key-value container for Serializable objects
     */
    public KeyValueDataCollection(DaalContext context) {
        super(context);
        this.cObject = cNewDataCollection();
    }

    public KeyValueDataCollection(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
        this.serializedCObject = null;
    }

    public long size() {
        return cSize(getCObject());
    }

    public SerializableBase get(int key) {
        return Factory.instance().createObject(getContext(), cGetValue(getCObject(), key));
    }

    public void set(int key, SerializableBase value) {
        cSetValue(getCObject(), key, value.getCObject());
    }

    public long getKeyByIndex(int index) {
        return cGetKeyByIndex(getCObject(), index);
    }

    public SerializableBase getValueByIndex(int index) {
        return Factory.instance().createObject(getContext(), cGetValueByIndex(getCObject(), index));
    }

    private native long cSize(long cObject);

    private native long cGetKeyByIndex  (long cObject, int index);
    private native long cGetValueByIndex(long cObject, int index);

    private native long cGetValue(long cObject, int key);
    private native void cSetValue(long cObject, int key, long cValue);

    private native long cNewDataCollection();
}
/** @} */
