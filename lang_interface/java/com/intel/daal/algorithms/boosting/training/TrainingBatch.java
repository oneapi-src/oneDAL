/* file: TrainingBatch.java */
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

/**
 * @brief Contains classes for training models of %boosting classifiers
 */
package com.intel.daal.algorithms.boosting.training;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__TRAINING__TRAININGBATCH"></a>
 * @brief Base class for training models of %boosting algorithms in the batch processing mode
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResult class
 */
public abstract class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs boosting training algorithm
     * @param context   Context to manage boosting prediction algorithm
     */
    public TrainingBatch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated boosting training algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context   Context to manage boosting prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract TrainingBatch clone(DaalContext context);
}
