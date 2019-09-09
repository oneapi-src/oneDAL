/* file: TrainingDistributedStep1Local.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @ingroup ridge_regression_distributed
 * @{
 */
/**
 * \brief Contains classes for ridge regression model-based training
 */
package com.intel.daal.algorithms.ridge_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__TRAININGDISTRIBUTEDSTEP1LOCAL"></a>
 * @brief Runs ridge regression model-based training in the first step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-RIDGEREGRESSION-ALGORITHM">Ridge regression algorithm description and usage models</a> -->
 *
 * @par References
 *      - Model class
 *      - ModelNormEq class
 *      - TrainingInputId class
 *      - PartialResultId class
 *      - TrainingResultId class
 */
public class TrainingDistributedStep1Local extends TrainingOnline {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a ridge regression training algorithm by copying input objects and parameters of another ridge regression training algorithm in the
     * first step of the distributed processing mode
     * @param context   Context to manage ridge regression model-based training
     * @param other     %Algorithm to use as the source to initialize the input objects and parameters of the algorithm
     */
    public TrainingDistributedStep1Local(DaalContext context, TrainingDistributedStep1Local other) {
        super(context, other);
    }

    /**
     * Constructs the ridge regression algorithm in the first step of the distributed processing mode
     * @param context   Context to manage ridge regression model-based training
     * @param cls       Data type to use in intermediate computations of ridge regression, Double.class or Float.class
     * @param method    %Algorithm computation method, @ref TrainingMethod
     */
    public TrainingDistributedStep1Local(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context, cls, method);
    }

    /**
     * Returns a newly allocated ridge regression training algorithm with a copy of the input objects and parameters of this ridge regression
     * training algorithm in the first step of the distributed processing mode
     * @param context   Context to manage ridge regression model-based training
     *
     * @return Newly allocated algorithm
     */
    @Override
    public TrainingDistributedStep1Local clone(DaalContext context) {
        return new TrainingDistributedStep1Local(context, this);
    }
}
/** @} */
