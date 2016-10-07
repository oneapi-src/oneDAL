/* file: QualityMetricSetParameter.java */
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
 * @brief Contains classes for computing the quality metrics
 */
package com.intel.daal.algorithms.linear_regression.quality_metric_set;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC_SET__QUALITYMETRICSETPARAMETER"></a>
 * @brief Class for the parameter of the linear regression quality metrics set algorithm
 */

public class QualityMetricSetParameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
