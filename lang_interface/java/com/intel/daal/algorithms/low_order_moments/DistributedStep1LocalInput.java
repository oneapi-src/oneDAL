/* file: DistributedStep1LocalInput.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDSTEP1LOCALINPUT"></a>
 * @brief Input objects for the low order %moments algorithm on the local node.
 */
public final class DistributedStep1LocalInput extends com.intel.daal.algorithms.low_order_moments.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep1LocalInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public DistributedStep1LocalInput(DaalContext context, long cObject, long cAlgorithm, Precision prec, Method method) {
        super(context, cObject, cAlgorithm, prec, method, ComputeMode.distributed);
    }
}
