/* file: TruncatedGaussianParameter.java */
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
 * @ingroup initializers_truncated_gaussian
 * @{
 */
package com.intel.daal.algorithms.neural_networks.initializers.truncated_gaussian;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__TRUNCATED_GAUSSIAN__TRUNCATEDGAUSSIANPARAMETER"></a>
 * @brief Class that specifies parameters of the neural network weights and biases truncated gaussian initializer
 */
public class TruncatedGaussianParameter extends com.intel.daal.algorithms.neural_networks.initializers.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private Precision _prec;

    /** @private */
    public TruncatedGaussianParameter(DaalContext context, long cObject, Precision prec) {
        super(context, cObject);
        _prec = prec;
    }
    /**
     * Returns the distribution mean
     * @return  The distribution mean
     */
    public double getMean() {
        return cGetMean(cObject, _prec.getValue());
    }

    /**
     * Sets the distribution mean
     * @param mean   The distribution mean
     */
    public void setMean(double mean) {
        cSetMean(cObject, mean, _prec.getValue());
    }

    /**
     * Returns the standard deviation of the distribution
     * @return  The standard deviation of the distribution
     */
    public double getSigma() {
        return cGetSigma(cObject, _prec.getValue());
    }

    /**
     * Sets the standard deviation of the distribution
     * @param sigma  The standard deviation of the distribution
     */
    public void setSigma(double sigma) {
        cSetSigma(cObject, sigma, _prec.getValue());
    }

    /**
     * Returns the left bound a of the truncation range from which the random values are selected
     * @return  Left bound of the truncation range
     */
    public double getA() {
        return cGetA(cObject, _prec.getValue());
    }

    /**
     * Sets the left bound a of the truncation range from which the random values are selected
     * @param a Left bound of the truncation range
     */
    public void setA(double a) {
        cSetA(cObject, a, _prec.getValue());
    }

    /**
     * Returns the right bound b of the truncation range from which the random values are selected
     * @return  Right bound of the truncation range
     */
    public double getB() {
        return cGetB(cObject, _prec.getValue());
    }

    /**
     * Sets the right bound b of the truncation range from which the random values are selected
     * @param b Right bound of the truncation range
     */
    public void setB(double b) {
        cSetB(cObject, b, _prec.getValue());
    }

    private native void cSetMean(long cObject, double mean, int prec);
    private native void cSetSigma(long cObject, double sigma, int prec);
    private native void cSetA(long cObject, double a, int prec);
    private native void cSetB(long cObject, double b, int prec);
    private native double cGetMean(long cObject, int prec);
    private native double cGetSigma(long cObject, int prec);
    private native double cGetA(long cObject, int prec);
    private native double cGetB(long cObject, int prec);
}
/** @} */
