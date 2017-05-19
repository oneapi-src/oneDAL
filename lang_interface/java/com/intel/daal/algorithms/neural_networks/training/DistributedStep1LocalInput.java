/* file: DistributedStep1LocalInput.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @ingroup neural_networks_training_distributed
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDPARTIALRESULT"></a>
 * @brief Input objects for the K-Means algorithm.
 *        Represents input objects for the algorithm on local nodes.
 */
public final class DistributedStep1LocalInput extends com.intel.daal.algorithms.neural_networks.training.TrainingInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep1LocalInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets input object for the neural network training algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(DistributedStep1LocalInputId id, TrainingModel val) {
        if (id == DistributedStep1LocalInputId.inputModel) {
            cSetModel(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect DistributedStep1LocalInputId");
        }
    }

    /**
     * Gets model of the neural network training algorithm
     * @param id    Identifier of the input object
     * @return  Model of the neural network training algorithm
     */
    public TrainingModel get(DistributedStep1LocalInputId id) {
        if (id == DistributedStep1LocalInputId.inputModel) {
            return new TrainingModel(getContext(), cGetModel(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("Incorrect DistributedStep1LocalInputId");
        }
    }

    private native void cSetModel(long cObject, int id, long ntAddr);
    private native long cGetModel(long cObject, int id);
}
/** @} */
