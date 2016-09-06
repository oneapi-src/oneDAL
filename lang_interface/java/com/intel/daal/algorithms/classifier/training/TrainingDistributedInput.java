/* file: TrainingDistributedInput.java */
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

package com.intel.daal.algorithms.classifier.training;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__TRAININGDISTRIBUTEDINPUT"></a>
 * @brief  Input objects of the classifier model training algorithm
 */
public class TrainingDistributedInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public TrainingDistributedInput(DaalContext context, long cInput) {
        super(context, cInput);
    }

    public TrainingDistributedInput(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context);
        this.cObject = cInit(cAlgorithm);
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

    private native long cInit(long cAlgorithm);

    private native void cAddInput(long cObject, int id, long presAddr);
}
