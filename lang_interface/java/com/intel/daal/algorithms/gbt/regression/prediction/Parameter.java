/* file: Parameter.java */
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
 * @ingroup gbt_regression_prediction
 */
/**
 * @brief Contains parameter for gradient boosted trees regression prediction algorithm
 */
package com.intel.daal.algorithms.gbt.regression.prediction;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__PREDICTION__PARAMETER"></a>
 * @brief Parameter of the gradient boosted trees regression prediction algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Returns number of iterations of the trained model to be used for prediction
     * by the gradient boosted trees prediction algorithm
     * @return Number of iterations
     */
    public long getNIterations() {
        return cGetNIterations(this.cObject);
    }

    /**
     * Sets the number of iterations of the trained model to be used for prediction
     * by the gradient boosted trees prediction algorithm
     * @param n Number of iterations
     */
    public void setNIterations(long n) {
        cSetNIterations(this.cObject, n);
    }

    private native long cGetNIterations(long parAddr);
    private native void cSetNIterations(long parAddr, long value);
}
/** @} */
