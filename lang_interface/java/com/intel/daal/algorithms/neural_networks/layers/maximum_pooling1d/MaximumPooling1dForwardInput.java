/* file: MaximumPooling1dForwardInput.java */
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

package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling1d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING1D__MAXIMUMPOOLING1DFORWARDINPUT"></a>
 * @brief %Input object for the forward one-dimensional maximum pooling layer
 */
public class MaximumPooling1dForwardInput extends com.intel.daal.algorithms.neural_networks.layers.pooling1d.Pooling1dForwardInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public MaximumPooling1dForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
