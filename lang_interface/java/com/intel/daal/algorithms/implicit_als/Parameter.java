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
 * @ingroup implicit_als
 * @{
 */
/**
 * \brief Contains classes for computing the results of the implicit ALS algorithm
 */
package com.intel.daal.algorithms.implicit_als;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PARAMETER"></a>
 * @brief Parameters for the compute() method of the implicit ALS algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long parAddr) {
        super(context);
        this.cObject = parAddr;
    }

    /**
     * Sets the nFactors parameter
     * @param nFactors  Number of factors
     */
    public void setNFactors(long nFactors) {
        cSetNFactors(this.cObject, nFactors);
    }

    /**
     * Gets the value of the nFactors parameter
     * @return nFactors
     */
    public long getNFactors() {
        return cGetNFactors(this.cObject);
    }

    /**
     * Sets the maxIterations parameter
     * @param maxIterations     Maximum number of iterations of the implicit ALS training algorithm
     */
    public void setMaxIterations(long maxIterations) {
        cSetMaxIterations(this.cObject, maxIterations);
    }

    /**
     * Gets the value of the maxIterations parameter
     * @return maxIterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    /**
     * Sets the confidence parameter of the implicit ALS training algorithm
     * @param alpha  Confidence parameter of the implicit ALS training algorithm
     */
    public void setAlpha(double alpha) {
        cSetAlpha(this.cObject, alpha);
    }

    /**
     * Gets the value of the confidence parameter of the implicit ALS training algorithm
     * @return Value of the confidence parameter of the implicit ALS training algorithm
     */
    public double getAlpha() {
        return cGetAlpha(this.cObject);
    }

    /**
     * Sets the regularization parameter of the implicit ALS training algorithm
     * @param lambda Regularization parameter of the implicit ALS training algorithm
     */
    public void setLambda(double lambda) {
        cSetLambda(this.cObject, lambda);
    }

    /**
     * Gets the value of the regularization parameter of the implicit ALS training algorithm
     * @return Value of the regularization parameter of the implicit ALS training algorithm
     */
    public double getLambda() {
        return cGetLambda(this.cObject);
    }

    /**
    * Sets the threshold used to define preference values
    * @param preferenceThreshold Threshold used to define preference values
    */
    public void setPreferenceThreshold(double preferenceThreshold) {
        cSetPreferenceThreshold(this.cObject, preferenceThreshold);
    }

    /**
     * Gets the value of the threshold used to define preference values
     * @return Value of the threshold used to define preference values
     */
    public double getPreferenceThreshold() {
        return cGetPreferenceThreshold(this.cObject);
    }

    private native void cSetNFactors(long algAddr, long nFactors);

    private native long cGetNFactors(long algAddr);

    private native void cSetMaxIterations(long algAddr, long maxIterations);

    private native long cGetMaxIterations(long algAddr);

    private native void cSetAlpha(long algAddr, double alpha);

    private native double cGetAlpha(long algAddr);

    private native void cSetLambda(long algAddr, double lambda);

    private native double cGetLambda(long algAddr);

    private native void cSetPreferenceThreshold(long algAddr, double preferenceThreshold);

    private native double cGetPreferenceThreshold(long algAddr);
}
/** @} */
