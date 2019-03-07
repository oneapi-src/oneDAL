/* file: SoftmaxForwardInput.java */
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

/**
 * @defgroup softmax_layers_forward Forward Softmax Layer
 * @brief Contains classes of the forward softmax layer
 * @ingroup softmax_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.softmax;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__SOFTMAXFORWARDINPUT"></a>
 * @brief %Input object for the forward softmax layer
 */
public class SoftmaxForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SoftmaxForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
