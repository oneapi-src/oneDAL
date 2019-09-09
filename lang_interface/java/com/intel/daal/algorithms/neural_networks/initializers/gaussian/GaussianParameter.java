/* file: GaussianParameter.java */
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
