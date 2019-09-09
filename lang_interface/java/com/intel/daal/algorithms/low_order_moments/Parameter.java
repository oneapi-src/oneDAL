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
