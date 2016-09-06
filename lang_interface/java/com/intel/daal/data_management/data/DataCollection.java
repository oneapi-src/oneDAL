/* file: DataCollection.java */
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

import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__DATACOLLECTION"></a>
 *  \brief Class that provides functionality of the Collection container for Serializable objects
 */
public class DataCollection extends SerializableBase {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

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
