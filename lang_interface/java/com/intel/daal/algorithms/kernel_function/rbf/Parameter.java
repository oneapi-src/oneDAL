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
 * @ingroup kernel_function_rbf
 * @{
 */
package com.intel.daal.algorithms.kernel_function.rbf;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__RBF__PARAMETER"></a>
 * \brief Parameters for computing the radial base function (RBF) kernel
 */
public class Parameter extends com.intel.daal.algorithms.kernel_function.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
    * Sets the RBF kernel coefficient
    * @param sigma   RBF kernel coefficient
    */
    public void setSigma(double sigma) {
        cSetSigma(this.cObject, sigma);
    }

    /**
    * Gets the RBF kernel coefficient
    * @return  RBF kernel coefficient
    */
    public double getSigma() {
        return cGetSigma(this.cObject);
    }

    private native void cSetSigma(long parAddr, double sigma);
    private native double cGetSigma(long parAddr);
}
/** @} */
