/* file: DistributedStep2MasterInput.java */
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

import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for the neural networks training algorithm in the second step of the distributed processing mode.
 *        Represents input objects for the algorithm on the master node.
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep2MasterInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Adds a partial result computed on local nodes to the input for the neural networks training algorithm
     * in the second step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param key           Key to use to retrieve data
     * @param pres          Partial results of the algorithm obtained in the first step
     *                      of the distributed processing mode
     */
    public void add(DistributedStep2MasterInputId id, int key, PartialResult pres) {
        cAddInput(cObject, id.getValue(), key, pres.getCObject());
    }

    private native void cAddInput(long algAddr, int id, int key, long presAddr);
}
/** @} */
