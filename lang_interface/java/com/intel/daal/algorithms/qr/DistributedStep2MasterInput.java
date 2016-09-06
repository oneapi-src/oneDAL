/* file: DistributedStep2MasterInput.java */
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

package com.intel.daal.algorithms.qr;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for the QR decomposition algorithm on the second step in the distributed processing mode
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep2MasterInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Adds value of input object to KeyValueDataCollection of the QR decomposition algorithm
     * @param id    Identifier of input object
     * @param key   Key to be used to retrieve data
     * @param val   Parameter value
     */
    public void add(DistributedStep2MasterInputId id, int key, DataCollection val) {
        if (id == DistributedStep2MasterInputId.inputOfStep2FromStep1) {
            cAddDataCollection(cObject, key, val.getCObject());
        }
    }

    private native void cAddDataCollection(long cInput, int key, long dcAddr);
}
