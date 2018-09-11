/* file: QualityMetricSetParameter.java */
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
 * @ingroup linear_regression_quality_metric_set
 * @{
 */
/**
 * @brief Contains classes for computing the quality metrics
 */
package com.intel.daal.algorithms.linear_regression.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC_SET__QUALITYMETRICSETPARAMETER"></a>
 * @brief Class for the parameter of the linear regression quality metrics set algorithm
 */

public class QualityMetricSetParameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public QualityMetricSetParameter(DaalContext context, long cParameter, long nBeta, long nBetaReducedModel) {
        super(context, cParameter);
        cSetNBeta(this.cObject, nBeta);
        cSetNBetaReducedModel(this.cObject, nBetaReducedModel);
    }

    /**
     *  Gets the number of beta coefficients
     *  @return  Number of betas
     */
    public long getNBeta() {
        return cGetNBeta(this.cObject);
    }

    /**
     *  Sets the number of beta coefficients
     *  @param nBeta Number of beta coefficients in the model
     */
    public void setNBeta(long nBeta) {
        cSetNBeta(this.cObject, nBeta);
    }

    /**
     *  Gets the number of beta coefficients in reduced model
     *  @return  Number of betas in reduced model
     */
    public long getNBetaReducedModel() {
        return cGetNBetaReducedModel(this.cObject);
    }

    /**
     *  Sets the number of beta coefficients
     *  @param nBetaReducedModel Number of beta coefficients in the model
     */
    public void setNBetaReducedModel(long nBetaReducedModel) {
        cSetNBetaReducedModel(this.cObject, nBetaReducedModel);
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

    private native void cSetNBeta(long parAddr, long nBeta);
    private native long cGetNBeta(long parAddr);
    private native void cSetNBetaReducedModel(long parAddr, long nBetaReducedModel);
    private native long cGetNBetaReducedModel(long parAddr);
    private native void cSetAlpha(long parAddr, double alpha);
    private native double cGetAlpha(long parAddr);
    private native void cSetAccuracyThreshold(long parAddr, double acc);
    private native double cGetAccuracyThreshold(long parAddr);
}
/** @} */
