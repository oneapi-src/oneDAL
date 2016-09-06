/* file: UniformParameter.java */
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

package com.intel.daal.algorithms.neural_networks.initializers.uniform;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__UNIFORM__UNIFORMPARAMETER"></a>
 * @brief Class that specifies parameters of the neural network weights and biases uniform initializer
 */
public class UniformParameter extends com.intel.daal.algorithms.neural_networks.initializers.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /** @private */
    public UniformParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the left bound a of the interval from which the random values are selected
     * @return  Left bound of the interval
     */
    public double getA() {
        return cGetA(cObject);
    }

    /**
     * Sets the left bound a of the interval from which the random values are selected
     * @param a Left bound of the interval
     */
    public void setA(double a) {
        cSetA(cObject, a);
    }

    /**
     * Returns the right bound b of the interval from which the random values are selected
     * @return  Right bound of the interval
     */
    public double getB() {
        return cGetB(cObject);
    }

    /**
     * Sets the right bound b of the interval from which the random values are selected
     * @param b Right bound of the interval
     */
    public void setB(double b) {
        cSetB(cObject, b);
    }

    /**
     * Returns the seed for generating random values
     * @return The seed for generating random values
     */
    public long getSeed() {
        return cGetSeed(cObject);
    }

    /**
     * Sets the seed for generating random values
     * @param seed The seed for generating random values
     */
    public void setSeed(long seed) {
        cSetSeed(cObject, seed);
    }

    private native void cSetA(long cObject, double a);
    private native void cSetB(long cObject, double b);
    private native void cSetSeed(long cObject, long seed);
    private native double cGetA(long cObject);
    private native double cGetB(long cObject);
    private native long cGetSeed(long cObject);
}
