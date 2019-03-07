/* file: TanhForwardInput.java */
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
 * @defgroup tanh_layers_forward Forward Hyperbolic Tangent Layer
 * @brief Contains classes for the forward hyperbolic tangent layer
 * @ingroup tanh_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.tanh;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__TANHFORWARDINPUT"></a>
 * @brief %Input object for the forward hyperbolic tangent (tanh) layer
 */
public class TanhForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TanhForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
