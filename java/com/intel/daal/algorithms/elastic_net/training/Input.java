/* file: Input.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @defgroup elastic_net_training Training
 * @brief Contains a class for elastic net model-based training
 * @ingroup elastic_net
 * @{
 */
package com.intel.daal.algorithms.elastic_net.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ELASTIC_NET__TRAINING__INPUT"></a>
 * @brief %Input object for elastic net model-based training in the batch and
 * online processing modes and in the first step of the distributed processing mode
 */
public class Input extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Input(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets an input object for elastic net model-based training
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(TrainingInputId id, NumericTable val) {
        if ((id != TrainingInputId.data) && (id != TrainingInputId.dependentVariable)) {
            throw new IllegalArgumentException("Incorrect TrainingInputId");
        }

        cSetInput(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input object for elastic net model-based training
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(TrainingInputId id) {
        if ((id != TrainingInputId.data) && (id != TrainingInputId.dependentVariable)) {
            throw new IllegalArgumentException("id unsupported"); // error processing
        }

        return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(this.cObject, id.getValue()));
    }

    private native void cSetInput(long cObject, int id, long resAddr);

    private native long cGetInput(long cObject, int id);
}
/** @} */
