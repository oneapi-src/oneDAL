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
 * @ingroup distributions_normal
 * @{
 */
package com.intel.daal.algorithms.distributions.normal;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__NORMAL__PARAMETER"></a>
 * @brief Class that specifies parameters of the normal distribution
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
     * Returns the mean
     * @return  Mean
     */
    public double getA() {
        return cGetA(cObject);
    }

    /**
     * Sets the mean
     * @param a Mean
     */
    public void setA(double a) {
        cSetA(cObject, a);
    }

    /**
     * Returns the standard deviation
     * @return  Standard deviation
     */
    public double getSigma() {
        return cGetSigma(cObject);
    }

    /**
     * Sets the standard deviation
     * @param sigma Standard deviation
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
