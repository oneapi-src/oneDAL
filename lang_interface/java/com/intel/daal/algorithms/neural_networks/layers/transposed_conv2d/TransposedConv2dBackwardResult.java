/* file: TransposedConv2dBackwardResult.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @ingroup transposed_conv2d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.transposed_conv2d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TRANSPOSED_CONV2D__TRANSPOSEDCONV2DBACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward 2D transposed convolution layer
 */
public class TransposedConv2dBackwardResult extends com.intel.daal.algorithms.neural_networks.layers.BackwardResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward 2D transposed convolution layer result
     * @param context   Context to manage the backward 2D transposed convolution layer result
     */
    public TransposedConv2dBackwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public TransposedConv2dBackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    private native long cNewResult();
}
/** @} */
