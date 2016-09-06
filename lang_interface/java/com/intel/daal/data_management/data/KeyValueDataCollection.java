/* file: KeyValueDataCollection.java */
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

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__KEYVALUEDATACOLLECTION"></a>
 *  \brief Class that provides functionality of the key-value container for Serializable objects
 *  with the key of integer type
 */
public class KeyValueDataCollection extends SerializableBase {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

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
