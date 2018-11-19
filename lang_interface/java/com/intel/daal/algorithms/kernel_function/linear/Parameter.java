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
 * @ingroup kernel_function_linear
 * @{
 */
package com.intel.daal.algorithms.kernel_function.linear;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__LINEAR__PARAMETER"></a>
 * @brief Parameters for computing the linear kernel function k(X,Y) + b
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
    * Sets the linear kernel coefficient k
    * @param k   Linear kernel coefficient k
    */
    public void setK(double k) {
        cSetK(this.cObject, k);
    }

    /**
    * Gets the linear kernel coefficient k
    * @return  Linear kernel coefficient k
    */
    public double getK() {
        return cGetK(this.cObject);
    }

    /**
    * Sets the linear kernel coefficient b
    * @param b   Linear kernel coefficient b
    */
    public void setB(double b) {
        cSetB(this.cObject, b);
    }

    /**
    * Gets the linear kernel coefficient b
    * @return  Linear kernel coefficient b
    */
    public double getB() {
        return cGetB(this.cObject);
    }

    private native void cSetK(long parAddr, double k);
    private native void cSetB(long parAddr, double b);
    private native double cGetK(long parAddr);
    private native double cGetB(long parAddr);
}
/** @} */
