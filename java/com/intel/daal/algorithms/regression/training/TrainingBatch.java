/* file: TrainingBatch.java */
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
 * @defgroup regression_training_batch Batch
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__TRAINING__TRAININGBATCH"></a>
 * @brief Algorithm class for training the regression model
 *
 * @par References
 *      - InputId class
 *      - TrainingResultId class
 *      - TrainingInput class
 *      - TrainingResult class
 */
public abstract class TrainingBatch extends com.intel.daal.algorithms.TrainingBatch {
    public TrainingInput input;
    protected Precision  prec;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the base regression training algorithm by copying input objects and parameters
     * of another base regression training algorithm
     * @param context   Context to manage the regression training algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        input = other.input;
        prec = other.prec;
    }

    /**
     * Constructs the base regression training algorithm
     * @param context   Context to manage the regression training algorithm
     */
    public TrainingBatch(DaalContext context) {
        super(context);
    }

    /**
     * Computes results of the regression training algorithm
     * \return Results of the regression training algorithm
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        return new TrainingResult(getContext(), cGetResult(this.cObject));
    }

    /**
     * Returns the newly allocated base regression training algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context   Context to manage the regression training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract TrainingBatch clone(DaalContext context);

    private native long cGetResult(long selfPtr);
}
/** @} */
