/* file: PredictionParameter.java */
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
 * @ingroup logistic_regression
 * @{
 */
/**
 * \brief Contains classes for computing the result of the logistic regression algorithm
 */
package com.intel.daal.algorithms.logistic_regression.prediction;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__PREDICTIONPARAMETER"></a>
 * @brief Logistic regression algorithm parameters
 */
public class PredictionParameter extends com.intel.daal.algorithms.classifier.Parameter {

    public PredictionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the 64 bit integer flag that indicates the results to compute
     * @param resultsToCompute
     */
    public void setResultsToCompute(long resultsToCompute) {
        cSetResultsToCompute(this.cObject, resultsToCompute);
    }

    /**
     * Returns the value of the resultsToCompute flag
     * @return resultsToCompute
     */
    public long getResultsToCompute() {
        return cGetResultsToCompute(this.cObject);
    }

    private native void cSetResultsToCompute(long parAddr, long resultsToCompute);
    private native long cGetResultsToCompute(long parAddr);
}
/** @} */
