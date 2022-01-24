/* file: InitDistributedStep2LocalInput.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

/**
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

import com.intel.daal.data_management.data.KeyValueDataCollection;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITDISTRIBUTEDSTEP2LOCALINPUT"></a>
 * @brief %Input objects for the implicit ALS initialization algorithm in the second step
 *        of the distributed processing mode
 */

public final class InitDistributedStep2LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep2LocalInput(DaalContext context, long cAlgorithm, Precision prec, InitMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets an input key-value data collection object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(InitStep2LocalInputId id, KeyValueDataCollection val) {
        if (id != InitStep2LocalInputId.inputOfStep2FromStep1) {
            throw new IllegalArgumentException("Incorrect InitStep2LocalInputId");
        }
        cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input key-value data collection object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(InitStep2LocalInputId id) {
        if (id != InitStep2LocalInputId.inputOfStep2FromStep1) {
            throw new IllegalArgumentException("Incorrect InitStep2LocalInputId"); // error processing
        }
        return new KeyValueDataCollection(getContext(), cGetDataCollection(this.cObject, id.getValue()));
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);

    private native void cSetDataCollection(long cObject, int id, long collectionAddr);
    private native long cGetDataCollection(long cObject, int id);
}
/** @} */
