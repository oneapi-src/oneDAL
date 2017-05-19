/* file: TruncatedGaussianParameter.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @ingroup initializers_truncated_gaussian
 * @{
 */
package com.intel.daal.algorithms.neural_networks.initializers.truncated_gaussian;

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__TRUNCATED_GAUSSIAN__TRUNCATEDGAUSSIANPARAMETER"></a>
 * @brief Class that specifies parameters of the neural network weights and biases truncated gaussian initializer
 */
public class TruncatedGaussianParameter extends com.intel.daal.algorithms.neural_networks.initializers.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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

    /**
     * Returns the seed for generating random values
     * @return The seed for generating random values
     */
    public long getSeed() {
        return cGetSeed(cObject, _prec.getValue());
    }

    /**
     * Sets the seed for generating random values
     * @param seed The seed for generating random values
     */
    public void setSeed(long seed) {
        cSetSeed(cObject, seed, _prec.getValue());
    }

    private native void cSetMean(long cObject, double mean, int prec);
    private native void cSetSigma(long cObject, double sigma, int prec);
    private native void cSetA(long cObject, double a, int prec);
    private native void cSetB(long cObject, double b, int prec);
    private native void cSetSeed(long cObject, long seed, int prec);
    private native double cGetMean(long cObject, int prec);
    private native double cGetSigma(long cObject, int prec);
    private native double cGetA(long cObject, int prec);
    private native double cGetB(long cObject, int prec);
    private native long cGetSeed(long cObject, int prec);
}
/** @} */
