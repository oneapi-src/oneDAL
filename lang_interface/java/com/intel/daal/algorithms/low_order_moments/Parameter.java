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
 * @ingroup low_order_moments
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__PARAMETER"></a>
 * @brief Parameters of the low order %moments algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets estimates to compute
     * @param id    Estimates to compute, @ref EstimatesToCompute
     */
    public void setEstimatesToCompute(EstimatesToCompute id) {
        cSetEstimatesToCompute(this.cObject, id.getValue());
    }

    /**
     * Gets estimates to compute
     * @return    Estimates to compute, @ref EstimatesToCompute
     */
    public EstimatesToCompute getEstimatesToCompute() {
        EstimatesToCompute id = new EstimatesToCompute(cGetEstimatesToCompute(this.cObject));
        return id;
    }

    private native void cSetEstimatesToCompute(long parAddr, int id);

    private native int cGetEstimatesToCompute(long parAddr);

}
/** @} */
