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
 * @ingroup distributions_uniform
 * @{
 */
package com.intel.daal.algorithms.distributions.uniform;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__UNIFORM__UNIFORMPARAMETER"></a>
 * @brief Class that specifies parameters of the uniform distribution
 */
public class Parameter extends com.intel.daal.algorithms.distributions.ParameterBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public Parameter(DaalContext context, long cObject) {
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
