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

package com.intel.daal.algorithms.ridge_regression.training;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input object for ridge regression model-based training in the second step of the distributed processing mode
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
     * Adds an input object on the master node for ridge regression model-based training
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void add(MasterInputId id, PartialResult val) {
        if (id != MasterInputId.partialModels) {
            throw new IllegalArgumentException("Incorrect MasterInputId");
        }
        cAddInput(this.cObject, id.getValue(), val.getCObject());
    }

    private native void cAddInput(long cObject, int id, long presAddr);
}
