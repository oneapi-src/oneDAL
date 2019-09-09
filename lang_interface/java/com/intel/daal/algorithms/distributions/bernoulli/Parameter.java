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
 * @ingroup distributions_bernoulli
 * @{
 */
package com.intel.daal.algorithms.distributions.bernoulli;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BERNOULLI__BERNOULLIPARAMETER"></a>
 * @brief Class that specifies parameters of the bernoulli distribution
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
     * Returns the success probability of a trial, value from [0.0; 1.0]
     * @return  Success probability of a trial
     */
    public double getP() {
        return cGetP(cObject);
    }

    /**
     * Sets the success probability of a trial, value from [0.0; 1.0]
     * @param p Success probability of a trial
     */
    public void setP(double p) {
        cSetP(cObject, p);
    }

    private native void cSetP(long cObject, double p);
    private native double cGetP(long cObject);
}
/** @} */
