/* file: Parameter.java */
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
