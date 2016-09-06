/* file: OptionalArgument.java */
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

package com.intel.daal.algorithms;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__OPTIONALARGUMENT"></a>
 *  \brief Class that provides functionality of the Collection container for Serializable objects
 */
public class OptionalArgument extends com.intel.daal.data_management.data.SerializableBase {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public OptionalArgument(DaalContext context, long size, long dummy) {
        super(context);
        this.cObject = cNewOptionalArgument(size);
    }

    public OptionalArgument(DaalContext context, long cOptionalArgument) {
        super(context);
        this.cObject = cOptionalArgument;
        this.serializedCObject = null;
    }

    public SerializableBase get(long idx) {
        return Factory.instance().createObject(getContext(), cGetValue(this.cObject, idx));
    }

    public void set(SerializableBase value, long idx) {
        cSetValue(this.cObject, value.getCObject(), idx);
    }

    private native long cNewOptionalArgument(long size);

    private native long cGetValue(long cOptionalArgumentAddr, long idx);

    private native void cSetValue(long cOptionalArgumentAddr, long cValueAddr, long idx);
}
