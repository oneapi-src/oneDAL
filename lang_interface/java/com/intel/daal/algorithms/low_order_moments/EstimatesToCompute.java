/* file: EstimatesToCompute.java */
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

package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__ESTIMATESTOCOMPUTE"></a>
 * @brief Available sets of estimates to compute of low order %Moments
 */
public final class EstimatesToCompute {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the sets of estimates object using the provided value
     * @param value     Value corresponding to the sets of estimates object
     */
    public EstimatesToCompute(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the sets of estimates object
     * @return Value corresponding to the sets of estimates object
     */
    public int getValue() {
        return _value;
    }

    private static final int EstimatesAll          = 0;
    private static final int EstimatesMinMax       = 1;
    private static final int EstimatesMeanVariance = 2;

    public static final EstimatesToCompute estimatesAll           = new EstimatesToCompute(EstimatesAll);          /*!< Default: Compute all supported moments */
    public static final EstimatesToCompute estimatesMinMax        = new EstimatesToCompute(EstimatesMinMax);       /*!< MinMAx: Compute minimum and maximum  */
    public static final EstimatesToCompute estimatesMeanVariance  = new EstimatesToCompute(EstimatesMeanVariance); /*!< MeanVariance: Compute mean and variance  */
}
