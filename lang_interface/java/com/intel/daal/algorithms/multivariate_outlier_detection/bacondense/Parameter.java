/* file: Parameter.java */
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

package com.intel.daal.algorithms.multivariate_outlier_detection.bacondense;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__BACONDENSE__PARAMETER"></a>
 * @brief Parameters of the multivariate outlier detection compute() method used with the baconDense method
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Sets initialization method for the BACON multivariate outlier detection algorithm
     * @param method Initialization method
     */
    public void setInitializationMethod(InitializationMethod method) {
        cSetInitializationMethod(this.cObject, method.getValue());
    }

    /**
     * Returns initialization method of the BACON multivariate outlier detection algorithm
     * @return Initialization method
     */
    public InitializationMethod getInitializationMethod() {
        int initMethodValue = cGetInitializationMethod(this.cObject);
        return new InitializationMethod(initMethodValue);
    }

    /**
     * Sets alpha parameter of the BACON method.
     * alpha is a one-tailed probability that defines the \f$(1 - \alpha)\f$ quantile
     * of the \f$\chi^2\f$ distribution with \f$p\f$ degrees of freedom.
     * Recommended value: \f$\alpha / n\f$, where n is the number of observations.
     * @param alpha Value of the parameter alpha
     */
    public void setAlpha(double alpha) {
        cSetAlpha(this.cObject, alpha);
    }

    /**
     * Returns the parameter alpha of the BACON method.
     * @return Parameter alpha of the BACON method.
     */
    public double getAlpha() {
        return cGetAlpha(this.cObject);
    }

    /**
     * Sets the threshold for the stopping criterion of the algorithms.
     * Stopping criterion: the algorithm is terminated if the size of the basic subset
     * is changed by less than the threshold.
     * @param threshold     Threshold for the stopping criterion of the algorithm
     */
    public void setToleranceToConverge(double threshold) {
        cSetToleranceToConverge(this.cObject, threshold);
    }

    /**
     * Sets the threshold for the stopping criterion of the algorithms.
     * @return Threshold for the stopping criterion of the algorithm
     */
    public double getToleranceToConverge() {
        return cGetToleranceToConverge(this.cObject);
    }

    private native void cSetInitializationMethod(long parAddr, int method);

    private native int cGetInitializationMethod(long parAddr);

    private native void cSetAlpha(long parAddr, double alpha);

    private native double cGetAlpha(long parAddr);

    private native void cSetToleranceToConverge(long parAddr, double threshold);

    private native double cGetToleranceToConverge(long parAddr);
}
