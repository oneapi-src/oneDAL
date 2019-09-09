/* file: EstimatesToCompute.java */
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

package com.intel.daal.algorithms.low_order_moments;

import java.lang.annotation.Native;

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

    @Native private static final int EstimatesAll          = 0;
    @Native private static final int EstimatesMinMax       = 1;
    @Native private static final int EstimatesMeanVariance = 2;

    public static final EstimatesToCompute estimatesAll           = new EstimatesToCompute(EstimatesAll);          /*!< Default: Compute all supported moments */
    public static final EstimatesToCompute estimatesMinMax        = new EstimatesToCompute(EstimatesMinMax);       /*!< MinMAx: Compute minimum and maximum  */
    public static final EstimatesToCompute estimatesMeanVariance  = new EstimatesToCompute(EstimatesMeanVariance); /*!< MeanVariance: Compute mean and variance  */
}
