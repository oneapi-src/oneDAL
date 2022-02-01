/* file: TrainingDistributedInput.java */
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
 * @ingroup multinomial_naive_bayes_training_distributed
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__MULTINOMIAL_NAIVE_BAYES__TRAININGDISTRIBUTEDINPUT"></a>
 * @brief  Input objects of the naive Bayes model training algorithm
 *         in the distributed computing mode
 */
public class TrainingDistributedInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingDistributedInput(DaalContext context, long cInput) {
        super(context, cInput);
    }

    /**
     * Adds input objects to the classifier model training algorithm on the master node
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void add(TrainingDistributedInputId id, TrainingPartialResult val) {
        if (id != TrainingDistributedInputId.partialModels) {
            throw new IllegalArgumentException("Incorrect TrainingDistributedInputId");
        }
        cAddInput(this.cObject, id.getValue(), val.getCObject());
    }

    private native void cAddInput(long cObject, int id, long presAddr);
}
/** @} */
