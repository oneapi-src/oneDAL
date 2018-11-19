/* file: Parameter.java */
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
