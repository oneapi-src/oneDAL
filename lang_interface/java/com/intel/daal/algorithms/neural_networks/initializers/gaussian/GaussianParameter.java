/* file: GaussianParameter.java */
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
 * @ingroup initializers_gaussian
 * @{
 */
package com.intel.daal.algorithms.neural_networks.initializers.gaussian;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__GAUSSIAN__GAUSSIANPARAMETER"></a>
 * @brief Class that specifies parameters of the neural network weights and biases gaussian initializer
 */
public class GaussianParameter extends com.intel.daal.algorithms.neural_networks.initializers.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public GaussianParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the distribution mean
     * @return  The distribution mean
     */
    public double getA() {
        return cGetA(cObject);
    }

    /**
     * Sets the distribution mean
     * @param a   The distribution mean
     */
    public void setA(double a) {
        cSetA(cObject, a);
    }

    /**
     * Returns the standard deviation of the distribution
     * @return  The standard deviation of the distribution
     */
    public double getSigma() {
        return cGetSigma(cObject);
    }

    /**
     * Sets the standard deviation of the distribution
     * @param sigma  The standard deviation of the distribution
     */
    public void setSigma(double sigma) {
        cSetSigma(cObject, sigma);
    }

    private native void cSetA(long cObject, double a);
    private native void cSetSigma(long cObject, double sigma);
    private native double cGetA(long cObject);
    private native double cGetSigma(long cObject);
}
/** @} */
