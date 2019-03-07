/* file: PredictionParameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
