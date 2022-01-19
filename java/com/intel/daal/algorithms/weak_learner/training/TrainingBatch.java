/* file: TrainingBatch.java */
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
 * @defgroup weak_learner_training_batch Batch
 * @ingroup weak_learner_training
 * @{
 */
/**
 * @brief Contains classes for training the weak learner model
 */
package com.intel.daal.algorithms.weak_learner.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__TRAINING__TRAININGBATCH"></a>
 * @brief Base class for training the weak learner model in the batch processing mode
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 *      - TrainingResult class
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs algorithm for training the weak learner model
     * by copying input objects and parameters of another algorithm
     * @param context   Context to manage the weak learner algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context, other);
    }

    /**
     * Constructs algorithm for training the weak learner model
     * @param context   Context to manage the weak learner algorithm
     */
    public TrainingBatch(DaalContext context) {
        super(context);
    }

    /**
     * Trains the weak learner model
     * @return Structure that contains computed training results
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the newly allocated algorithm for training the weak learner model
     * with a copy of input objects and parameters of this algorithm
     * @param context   Context to manage the weak learner algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cGetResult(long algAddr);
}
/** @} */
