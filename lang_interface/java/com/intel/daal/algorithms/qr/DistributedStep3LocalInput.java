/* file: DistributedStep3LocalInput.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP3LOCALINPUT"></a>
 * @brief Input objects for the third step of the QR decomposition algorithm in the distributed processing mode
 */
public final class DistributedStep3LocalInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep3LocalInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets input object for the QR decomposition algorithm
     * @param id    Identifier of input object
     * @param val   Value of the input object
     */
    public void set(DistributedStep3LocalInputId id, DataCollection val) throws IllegalArgumentException {
        if (id != DistributedStep3LocalInputId.inputOfStep3FromStep1
                && id != DistributedStep3LocalInputId.inputOfStep3FromStep2) {
            throw new IllegalArgumentException("Incorrect DistributedStep3LocalInputId");
        }

        cSetDataCollection(cObject, id.getValue(), val.getCObject());
    }

    private native void cSetDataCollection(long cInput, int id, long dcAddr);
}
