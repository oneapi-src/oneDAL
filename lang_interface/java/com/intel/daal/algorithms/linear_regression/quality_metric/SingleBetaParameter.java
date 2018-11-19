/* file: SingleBetaParameter.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup linear_regression_quality_metric_single_beta
 * @{
 */
/**
 * @brief Contains classes for computing the single beta metric */
package com.intel.daal.algorithms.linear_regression.quality_metric;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETAPARAMETER"></a>
 * @brief Base class for the parameters of the algorithm
 */
public class SingleBetaParameter extends com.intel.daal.algorithms.Parameter {

    /**
     * Constructs the parameters of the quality metric algorithm
     * @param context   Context to manage the parameters of the quality metric algorithm
     */
    public SingleBetaParameter(DaalContext context) {
        super(context);
    }

    public SingleBetaParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Retrieves the significance level used in the computation of beta confidence intervals
     * @return Significance level of the algorithm
     */
    public double getAlpha() {
        return cGetAlpha(this.cObject);
    }

    /**
     * Sets the significance level used in the computation of beta confidence intervals
     * @param alpha Significance level of the algorithm
     */
    public void setAlpha(double alpha) {
        cSetAlpha(this.cObject, alpha);
    }

    /**
     * Retrieves the accuracy of the algorithm (used for statistics calculation)
     * @return Accuracy of the algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the accuracy of the algorithm (used for statistics calculation)
     * @param accuracyThreshold Accuracy of the algorithm
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    private native void cSetAlpha(long parAddr, double alpha);
    private native double cGetAlpha(long parAddr);
    private native void cSetAccuracyThreshold(long parAddr, double acc);
    private native double cGetAccuracyThreshold(long parAddr);
}
/** @} */
