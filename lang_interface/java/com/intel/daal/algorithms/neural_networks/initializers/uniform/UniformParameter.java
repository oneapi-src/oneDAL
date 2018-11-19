/* file: UniformParameter.java */
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
 * @ingroup initializers_uniform
 * @{
 */
package com.intel.daal.algorithms.neural_networks.initializers.uniform;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__UNIFORM__UNIFORMPARAMETER"></a>
 * @brief Class that specifies parameters of the neural network weights and biases uniform initializer
 */
public class UniformParameter extends com.intel.daal.algorithms.neural_networks.initializers.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
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

    private native void cSetA(long cObject, double a);
    private native void cSetB(long cObject, double b);
    private native double cGetA(long cObject);
    private native double cGetB(long cObject);
}
/** @} */
