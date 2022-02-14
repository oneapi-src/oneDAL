/* file: TrainingOnline.java */
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
 * @defgroup classifier_training_online Online
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.classifier.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.Result;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__TRAININGONLINE"></a>
 * @brief Algorithm class for the classifier model training algorithm
 *
 * @par References
 *      - InputId class
 *      - TrainingResultId class
 *      - TrainingInput class
 *      - TrainingResult class
 */
public abstract class TrainingOnline extends com.intel.daal.algorithms.TrainingOnline {
    public TrainingInput input;
    protected Precision  prec;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the base classifier training algorithm in the online processing mode
     * by copying input objects and parameters of another base classifier training algorithm
     * @param context   Context to manage the classifier training algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingOnline(DaalContext context, TrainingOnline other) {
        super(context);
        input = other.input;
        prec = other.prec;
    }

    /**
     * Constructs the base classifier model training algorithm in the online processing mode
     * @param context   Context to manage the classifier training algorithm in the online processing mode
     */
    public TrainingOnline(DaalContext context) {
        super(context);
    }

    /**
     * Computes partial results of the classifier model training algorithm
     * \return Partial results of the classifier model training algorithm
     */
    @Override
    public PartialResult compute() {
        super.compute();
        return null;
    }

    /**
     * Computes final results of the classifier model training algorithm
     * \return Results of the classifier model training algorithm
     */
    @Override
    public Result finalizeCompute() {
        super.finalizeCompute();
        return null;
    }

    /**
     * Returns the newly allocated base classifier training algorithm in the online processing mode
     * with a copy of input objects and parameters of this algorithm
     * @param context   Context to manage the classifier training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract TrainingOnline clone(DaalContext context);
}
/** @} */
