/* file: SmoothreluForwardInput.java */
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
 * @defgroup smoothrelu_layers_forward Forward Smooth Rectifier Linear Unit (SmoothReLU) Layer
 * @brief Contains classes for the forward smooth relu layer
 * @ingroup smoothrelu_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.smoothrelu;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SMOOTHRELU__SMOOTHRELUFORWARDINPUT"></a>
 * @brief %Input object for the forward smoothrelu layer
 */
public class SmoothreluForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SmoothreluForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
