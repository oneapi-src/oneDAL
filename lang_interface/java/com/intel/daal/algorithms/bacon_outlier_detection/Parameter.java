/* file: Parameter.java */
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
 * @ingroup bacon_outlier_detection_defaultdense
 * @{
 */
package com.intel.daal.algorithms.bacon_outlier_detection;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BACON_OUTLIER_DETECTION__DEFAULTDENSE__PARAMETER"></a>
 * @brief Parameters of the multivariate outlier detection compute() method used with the defaultDense method
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
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
/** @} */
