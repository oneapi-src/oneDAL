/* file: LossBatch.java */
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

/**
 * @brief Contains classes of the loss layer
 */
package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSBATCH"></a>
 * @brief Provides methods for the loss layer in the batch processing mode
 * \n<a href="DAAL-REF-LOSSFORWARD-ALGORITHM">Forward loss layer description and usage models</a>
 * \n<a href="DAAL-REF-LOSSBACKWARD-ALGORITHM">Backward loss layer description and usage models</a>
 *
 * @par References
 *      - @ref LossForwardBatch class
 *      - @ref LossBackwardBatch class
 */
public class LossBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the loss layer
     * @param context    Context to manage the loss layer
     * @param cObject    Address of C++ object
     */
    public LossBatch(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
    * Constructs the loss layer
    * @param context    Context to manage the loss layer
    */
    public LossBatch(DaalContext context) {
        super(context);
    }
}
