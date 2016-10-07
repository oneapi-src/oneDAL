/* file: PredictionBatch.java */
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
 * @brief Contains classes for making predictions based on the weak learner model
 */
package com.intel.daal.algorithms.weak_learner.prediction;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHM__WEAK_LEARNER__PREDICTION__PREDICTIONBATCH"></a>
 *  @brief Base class for making predictions based on the weak learner model
 *
 *  @par References
 *      - com.intel.daal.algorithms.classifier.prediction.NumericTableInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.ModelInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResultId class
 *      - com.intel.daal.algorithms.weak_learner.Model class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionInput class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResult class
 *
 */
public class PredictionBatch extends com.intel.daal.algorithms.classifier.prediction.PredictionBatch {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs algorithm for making predictions based on the weak learner model
     * by copying input objects and parameters of another algorithm
     * @param context   Context to manage the weak learner algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context, other);
    }

    /**
     * Constructs algorithm for making predictions based on the weak learner model
     * @param context   Context to manage the weak learner algorithm
     */
    public PredictionBatch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated algorithm for making predictions based on the weak learner model
     * with a copy of input objects and parameters of this algorithm
     * @param context   Context to manage the weak learner algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }
}
