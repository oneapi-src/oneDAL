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
